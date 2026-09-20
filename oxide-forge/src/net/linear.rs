use crate::cuda::{self, runtime::CudaRuntime};
use cuda::container::{Matrix, Vector};
use cuda_core::CudaStream;
use serde::{Deserialize, Serialize};

use crate::graph::state::ParameterBuffer;
use crate::graph::{GraphNode, LearnConfig, MatrixConfig, NodeTrainState};
use crate::net::metadata::{
    HostData, HostDataCursor, MatrixMetadata, MetadataCursor, VectorMetadata,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    Identity,
    Gelu,
    Relu,
    Silu,
    Sigmoid,
}

/// Device-independent description of a Linear block. Parameter storage is
/// created only when its containing graph is initialized.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinearConfig {
    pub weights: MatrixConfig,
    pub bias: bool,
    pub activation: Activation,
}

impl LinearConfig {
    pub const fn new(weights: MatrixConfig, bias: bool, activation: Activation) -> Self {
        Self {
            weights,
            bias,
            activation,
        }
    }
}

impl Activation {
    const GELU_SCALE: f32 = 0.797_884_6;
    const GELU_CUBIC: f32 = 0.044_715;
    const GELU_DERIVATIVE_QUADRATIC: f32 = Self::GELU_SCALE * 3.0 * Self::GELU_CUBIC;

    #[inline(always)]
    pub fn forward(self, x: f32) -> f32 {
        match self {
            Self::Identity => x,
            Self::Gelu => {
                let inner = Self::GELU_SCALE * (x + Self::GELU_CUBIC * x * x * x);
                0.5 * x * (1.0 + inner.tanh())
            }
            Self::Relu => {
                if x > 0.0 {
                    x
                } else {
                    0.0
                }
            }
            Self::Silu => x / (1.0 + (-x).exp()),
            Self::Sigmoid => crate::cuda::sigmoid_f32(x),
        }
    }

    #[inline(always)]
    pub fn derivative(self, x: f32) -> f32 {
        match self {
            Self::Identity => 1.0,
            Self::Gelu => {
                let inner = Self::GELU_SCALE * (x + Self::GELU_CUBIC * x * x * x);
                let tanh_inner = inner.tanh();
                let inner_derivative = Self::GELU_SCALE + Self::GELU_DERIVATIVE_QUADRATIC * x * x;

                0.5 * (1.0 + tanh_inner)
                    + 0.5 * x * (1.0 - tanh_inner * tanh_inner) * inner_derivative
            }
            Self::Relu => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Silu => {
                let exp_neg_x = (-x).exp();
                1.0 / (1.0 + exp_neg_x) + x * exp_neg_x / ((1.0 + exp_neg_x) * (1.0 + exp_neg_x))
            }
            Self::Sigmoid => {
                let sigmoid = crate::cuda::sigmoid_f32(x);
                sigmoid * (1.0 - sigmoid)
            }
        }
    }
}

pub struct Linear {
    weights: Matrix,
    bias: Option<Vector>,
    activation: Activation,
}

/// Trainable graph node backed by a `Linear` layer.
///
/// The owned input is retained without a device copy until backward. For
/// non-identity activations the pre-activation is recomputed during backward,
/// avoiding a second output-sized persistent cache.
pub(crate) struct LinearTrainingNode {
    linear: Linear,
    input_cache: Option<Matrix>,
    training: NodeTrainState,
}

/// Transitional name used by compound blocks while their state ownership is
/// moved to `Graph<true>`. The storage itself is already graph-generic.
pub(crate) type LinearTrainingState = NodeTrainState;

impl Linear {
    pub(crate) fn accumulate_training(
        &self,
        state: &mut NodeTrainState,
        input: &Matrix,
        gradient: &Matrix,
        bias_gradient: Option<&Vector>,
        runtime: &mut CudaRuntime,
    ) {
        let input_transpose = runtime.matrix_transpose(input, None);
        let weight_gradient = runtime.matrix_multiply(&input_transpose, gradient, None);
        runtime.recycle_matrix(input_transpose);

        match &mut state.parameters[0].gradient {
            Some(ParameterBuffer::Matrix(total)) => {
                total.binary_assign(&weight_gradient, move |lhs, rhs| lhs + rhs, runtime, None);
                runtime.recycle_matrix(weight_gradient);
            }
            None => state.parameters[0].gradient = Some(ParameterBuffer::Matrix(weight_gradient)),
            Some(ParameterBuffer::Vector(_)) => panic!("Linear weight gradient type mismatch"),
        }

        match bias_gradient {
            Some(bias_gradient) => match &mut state.parameters[1].gradient {
                Some(ParameterBuffer::Vector(total)) => {
                    total.binary_assign(bias_gradient, runtime, move |lhs, rhs| lhs + rhs, None)
                }
                None => {
                    state.parameters[1].gradient = Some(ParameterBuffer::Vector(
                        runtime.clone_vector(bias_gradient, None),
                    ));
                }
                Some(ParameterBuffer::Matrix(_)) => {
                    panic!("Linear bias gradient type mismatch")
                }
            },
            None => assert!(
                state.parameters[1].gradient.is_none(),
                "missing bias gradient while a batch is being accumulated"
            ),
        }
    }

    pub(crate) fn learn_from_state(
        &mut self,
        state: &mut NodeTrainState,
        learning_rate: f32,
        momentum: f32,
        batch_len: usize,
        runtime: &mut CudaRuntime,
    ) {
        assert!(batch_len > 0, "optimizer batch must not be empty");
        assert!(
            learning_rate.is_finite() && learning_rate > 0.0,
            "learning rate must be finite and greater than zero"
        );
        assert!(
            momentum.is_finite() && (0.0..1.0).contains(&momentum),
            "momentum must be finite and in [0, 1)"
        );

        let inverse_batch = 1.0 / batch_len as f32;
        let mut weight_gradient = state.parameters[0]
            .gradient
            .take()
            .and_then(|buffer| match buffer {
                ParameterBuffer::Matrix(matrix) => Some(matrix),
                ParameterBuffer::Vector(_) => None,
            })
            .expect("optimizer step requires an accumulated weight gradient");
        weight_gradient.scale(inverse_batch, runtime, None);
        match &mut state.parameters[0].velocity {
            Some(ParameterBuffer::Matrix(velocity)) => {
                velocity.scale(momentum, runtime, None);
                velocity.binary_assign(&weight_gradient, move |lhs, rhs| lhs + rhs, runtime, None);
                runtime.recycle_matrix(weight_gradient);
            }
            None => state.parameters[0].velocity = Some(ParameterBuffer::Matrix(weight_gradient)),
            Some(ParameterBuffer::Vector(_)) => panic!("Linear weight velocity type mismatch"),
        }

        let ParameterBuffer::Matrix(weight_velocity) =
            state.parameters[0].velocity.as_ref().unwrap()
        else {
            unreachable!()
        };
        let mut weight_update = runtime.clone_matrix(weight_velocity, None);
        weight_update.scale(learning_rate, runtime, None);
        self.weights
            .binary_assign(&weight_update, move |lhs, rhs| lhs - rhs, runtime, None);
        runtime.recycle_matrix(weight_update);

        match (&mut self.bias, state.parameters[1].gradient.take()) {
            (Some(bias), Some(ParameterBuffer::Vector(mut bias_gradient))) => {
                bias_gradient.scale(inverse_batch, runtime, None);
                match &mut state.parameters[1].velocity {
                    Some(ParameterBuffer::Vector(velocity)) => {
                        velocity.scale(momentum, runtime, None);
                        velocity.binary_assign(
                            &bias_gradient,
                            runtime,
                            move |lhs, rhs| lhs + rhs,
                            None,
                        );
                        runtime.recycle_vector(bias_gradient);
                    }
                    None => {
                        state.parameters[1].velocity = Some(ParameterBuffer::Vector(bias_gradient));
                    }
                    Some(ParameterBuffer::Matrix(_)) => {
                        panic!("Linear bias velocity type mismatch")
                    }
                }

                let ParameterBuffer::Vector(bias_velocity) =
                    state.parameters[1].velocity.as_ref().unwrap()
                else {
                    unreachable!()
                };
                let mut bias_update = runtime.clone_vector(bias_velocity, None);
                bias_update.scale(learning_rate, runtime, None);
                bias.binary_assign(&bias_update, runtime, move |lhs, rhs| lhs - rhs, None);
                runtime.recycle_vector(bias_update);
            }
            (None, None) => {}
            (_, Some(ParameterBuffer::Matrix(_))) => panic!("Linear bias gradient type mismatch"),
            _ => panic!("bias and accumulated bias gradient do not match"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinearMetadata {
    pub input_neurons: usize,
    pub output_neurons: usize,
    pub activation: Activation,
    pub weights: MatrixMetadata,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bias: Option<VectorMetadata>,
}

impl Linear {
    pub fn input_neurons(&self) -> usize {
        self.weights.rows()
    }

    pub fn output_neurons(&self) -> usize {
        self.weights.cols()
    }

    pub fn new(weights: Matrix, bias: Option<Vector>, activation: Activation) -> Self {
        Self {
            weights,
            bias,
            activation,
        }
    }

    pub fn get_meta_data(&self, cursor: &mut MetadataCursor) -> LinearMetadata {
        LinearMetadata {
            input_neurons: self.weights.rows(),
            output_neurons: self.weights.cols(),
            activation: self.activation,
            weights: cursor.matrix(self.weights.rows(), self.weights.cols()),
            bias: self.bias.as_ref().map(|bias| cursor.vector(bias.len())),
        }
    }

    pub fn get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
        let mut data = Vec::with_capacity(1 + usize::from(self.bias.is_some()));
        data.push(HostData::new(self.weights.to_host(runtime, None)));
        if let Some(bias) = &self.bias {
            data.push(HostData::new(bias.to_host(runtime, None)));
        }
        data
    }

    pub fn set_data(&mut self, data: &mut HostDataCursor, runtime: &CudaRuntime) {
        let weights = data.take();
        self.weights
            .copy_from_host(weights.values(), runtime, None)
            .unwrap();
        if let Some(bias) = &mut self.bias {
            let values = data.take();
            bias.copy_from_host(values.values(), runtime, None).unwrap();
        }
    }

    pub fn forward(
        &self,
        input: &Matrix,
        residual: Option<&Matrix>,
        runtime: &mut CudaRuntime,
        stream: Option<&CudaStream>,
    ) -> Matrix {
        assert_eq!(input.cols(), self.weights.rows());
        let mut output = runtime.new_uninit_matrix(input.rows(), self.weights.cols());
        self.affine_into(input, residual, &mut output, runtime, stream);
        self.activate(&mut output, runtime, stream);
        output
    }

    fn affine_into(
        &self,
        input: &Matrix,
        residual: Option<&Matrix>,
        output: &mut Matrix,
        runtime: &CudaRuntime,
        stream: Option<&CudaStream>,
    ) {
        assert_eq!(input.cols(), self.weights.rows());
        assert_eq!(output.rows(), input.rows());
        assert_eq!(output.cols(), self.weights.cols());
        runtime.matrix_multiply_into(input, &self.weights, output, stream);
        if let Some(ref bias) = self.bias {
            assert_eq!(output.cols(), bias.len());
            output.binary_assign_by_rows(bias, move |lhs, rhs| lhs + rhs, runtime, stream);
        }
        if let Some(residual_matrix) = residual {
            assert_eq!(output.rows(), residual_matrix.rows());
            assert_eq!(output.cols(), residual_matrix.cols());
            output.binary_assign(residual_matrix, move |lhs, rhs| lhs + rhs, runtime, stream);
        }
    }

    pub fn affine(
        &self,
        input: &Matrix,
        residual: Option<&Matrix>,
        runtime: &mut CudaRuntime,
        stream: Option<&CudaStream>,
    ) -> Matrix {
        let mut output = runtime.new_uninit_matrix(input.rows(), self.weights.cols());
        self.affine_into(input, residual, &mut output, runtime, stream);
        output
    }

    pub fn activate(
        &self,
        output: &mut Matrix,
        runtime: &CudaRuntime,
        stream: Option<&CudaStream>,
    ) {
        if !matches!(self.activation, Activation::Identity) {
            let activation = self.activation;
            output.for_each(runtime, move |x| activation.forward(x), stream);
        }
    }

    pub fn backward(
        &self,
        pre_activation: Option<&Matrix>,
        output_gradient: &Matrix,
        runtime: &mut CudaRuntime,
        stream: Option<&CudaStream>,
    ) -> (Matrix, Option<Vector>) {
        let mut gradient = runtime.clone_matrix(output_gradient, stream);

        if !matches!(self.activation, Activation::Identity) {
            let pre_activation =
                pre_activation.expect("activation backward requires pre-activation");
            assert_eq!(pre_activation.rows(), output_gradient.rows());
            assert_eq!(pre_activation.cols(), output_gradient.cols());
            let activation = self.activation;
            let mut derivative = runtime.clone_matrix(pre_activation, stream);

            derivative.for_each(runtime, move |x| activation.derivative(x), stream);

            gradient.binary_assign(&derivative, move |lhs, rhs| lhs * rhs, runtime, stream);
            runtime.recycle_matrix(derivative);
        }

        let bias_gradient = if self.bias.is_some() {
            let transposed = runtime.matrix_transpose(&gradient, stream);
            let bias_gradient = runtime.matrix_sum_rows(&transposed, stream);
            runtime.recycle_matrix(transposed);
            Some(bias_gradient)
        } else {
            None
        };

        (gradient, bias_gradient)
    }

    pub fn needs_pre_activation(&self) -> bool {
        !matches!(self.activation, Activation::Identity)
    }

    pub fn loss_rows(
        output: &Matrix,
        target: &Matrix,
        runtime: &mut CudaRuntime,
        stream: Option<&CudaStream>,
    ) -> Vector {
        let mut loss = runtime.matrix_binary(output, target, move |lhs, rhs| lhs - rhs, stream);
        loss.for_each(runtime, |x| 0.5 * x * x, stream);

        let cols = loss.cols() as f32;
        let mut row_loss = runtime.matrix_sum_rows(&loss, stream);
        row_loss.scale(1.0 / cols, runtime, stream);
        row_loss
    }

    pub fn learn(
        &mut self,
        input: &Matrix,
        gradient: &Matrix,
        bias_gradient: Option<&Vector>,
        learning_rate: f32,
        runtime: &mut CudaRuntime,
        stream: Option<&CudaStream>,
    ) {
        let input_transpose = runtime.matrix_transpose(input, stream);
        let mut weight_gradient = runtime.matrix_multiply(&input_transpose, gradient, stream);
        weight_gradient.for_each(runtime, move |x| x * learning_rate, stream);
        self.weights
            .binary_assign(&weight_gradient, move |lhs, rhs| lhs - rhs, runtime, stream);
        if let Some(ref mut bias) = self.bias {
            let mut scaled_bias_gradient = runtime.clone_vector(
                bias_gradient.expect("missing bias gradient for biased Linear"),
                stream,
            );
            scaled_bias_gradient.scale(learning_rate, runtime, stream);
            bias.binary_assign(
                &scaled_bias_gradient,
                runtime,
                move |lhs, rhs| lhs - rhs,
                stream,
            );
        }
    }

    pub fn input_gradient(
        &self,
        gradient: &Matrix,
        runtime: &mut CudaRuntime,
        stream: Option<&CudaStream>,
    ) -> Matrix {
        assert_eq!(gradient.cols(), self.weights.cols());
        let weights_transpose = runtime.matrix_transpose(&self.weights, stream);
        let input_gradient = runtime.matrix_multiply(gradient, &weights_transpose, stream);
        runtime.recycle_matrix(weights_transpose);
        input_gradient
    }

    pub fn backward_with_res(
        &self,
        pre_activation: &Matrix,
        output_gradient: &Matrix,
        residual_gradient: &Matrix,
        runtime: &mut CudaRuntime,
        stream: Option<&CudaStream>,
    ) -> (Matrix, Option<Vector>) {
        assert_eq!(pre_activation.rows(), output_gradient.rows());
        assert_eq!(pre_activation.cols(), output_gradient.cols());
        assert_eq!(output_gradient.rows(), residual_gradient.rows());
        assert_eq!(output_gradient.cols(), residual_gradient.cols());

        let mut gradient = runtime.clone_matrix(output_gradient, stream);
        gradient.binary_assign(
            residual_gradient,
            move |lhs, rhs| lhs + rhs,
            runtime,
            stream,
        );

        if !matches!(self.activation, Activation::Identity) {
            let activation = self.activation;
            let mut derivative = runtime.clone_matrix(pre_activation, stream);

            derivative.for_each(runtime, move |x| activation.derivative(x), stream);

            gradient.binary_assign(&derivative, move |lhs, rhs| lhs * rhs, runtime, stream);
            runtime.recycle_matrix(derivative);
        }

        let bias_gradient = if self.bias.is_some() {
            let transposed = runtime.matrix_transpose(&gradient, stream);
            let bias_gradient = runtime.matrix_sum_rows(&transposed, stream);
            runtime.recycle_matrix(transposed);
            Some(bias_gradient)
        } else {
            None
        };

        (gradient, bias_gradient)
    }
}

impl LinearTrainingNode {
    pub fn new(linear: Linear) -> Self {
        Self {
            linear,
            input_cache: None,
            training: LinearTrainingState::with_parameter_count(2),
        }
    }

    pub fn get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
        self.linear.get_data(runtime)
    }

    pub fn set_data(&mut self, data: &mut HostDataCursor, runtime: &CudaRuntime) {
        self.linear.set_data(data, runtime);
    }
}

impl From<Linear> for LinearTrainingNode {
    fn from(linear: Linear) -> Self {
        Self::new(linear)
    }
}

fn take_single(mut matrices: Vec<Matrix>, operation: &str) -> Matrix {
    assert_eq!(matrices.len(), 1, "{operation} expects exactly one matrix");
    matrices.pop().unwrap()
}

impl GraphNode for Linear {
    fn create_train_state(&self) -> NodeTrainState {
        NodeTrainState::with_parameter_count(2)
    }

    fn output_configs(&self, inputs: &[MatrixConfig]) -> Vec<MatrixConfig> {
        assert_eq!(inputs.len(), 1, "Linear expects one input");
        assert_eq!(
            inputs[0].cols,
            self.input_neurons(),
            "Linear input width mismatch"
        );
        vec![MatrixConfig::new(inputs[0].rows, self.output_neurons())]
    }

    fn forward(&mut self, inputs: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        let input = take_single(inputs, "Linear forward");
        let output = Linear::forward(self, &input, None, runtime, None);
        runtime.recycle_matrix(input);
        vec![output]
    }

    fn backward(&mut self, _gradients: Vec<Matrix>, _runtime: &mut CudaRuntime) -> Vec<Matrix> {
        panic!("inference Linear does not support backward; initialize Graph<true>")
    }

    fn learn(&mut self, _config: LearnConfig, _runtime: &mut CudaRuntime) {
        panic!("inference Linear does not support optimizer steps; initialize Graph<true>")
    }

    fn clear_cache(&mut self, _runtime: &mut CudaRuntime) {}

    fn get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
        Linear::get_data(self, runtime)
    }

    fn set_data(&mut self, data: &mut HostDataCursor, runtime: &CudaRuntime) {
        Linear::set_data(self, data, runtime);
    }
}

impl GraphNode for LinearTrainingNode {
    fn output_configs(&self, inputs: &[MatrixConfig]) -> Vec<MatrixConfig> {
        self.linear.output_configs(inputs)
    }

    fn forward(&mut self, inputs: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        self.clear_cache(runtime);
        let input = take_single(inputs, "Linear training forward");
        let mut output = self.linear.affine(&input, None, runtime, None);
        self.linear.activate(&mut output, runtime, None);
        self.input_cache = Some(input);
        vec![output]
    }

    fn backward(&mut self, gradients: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        let output_gradient = take_single(gradients, "Linear training backward");
        let input = self
            .input_cache
            .take()
            .expect("Linear training forward must run before backward");
        let pre_activation = self
            .linear
            .needs_pre_activation()
            .then(|| self.linear.affine(&input, None, runtime, None));
        let (gradient, bias_gradient) =
            self.linear
                .backward(pre_activation.as_ref(), &output_gradient, runtime, None);
        let input_gradient = self.linear.input_gradient(&gradient, runtime, None);
        self.linear.accumulate_training(
            &mut self.training,
            &input,
            &gradient,
            bias_gradient.as_ref(),
            runtime,
        );

        if let Some(pre_activation) = pre_activation {
            runtime.recycle_matrix(pre_activation);
        }
        if let Some(bias_gradient) = bias_gradient {
            runtime.recycle_vector(bias_gradient);
        }
        runtime.recycle_matrix(gradient);
        runtime.recycle_matrix(output_gradient);
        runtime.recycle_matrix(input);
        vec![input_gradient]
    }

    fn learn(&mut self, config: LearnConfig, runtime: &mut CudaRuntime) {
        self.linear.learn_from_state(
            &mut self.training,
            config.learning_rate,
            config.momentum,
            config.batch_len,
            runtime,
        );
    }

    fn clear_cache(&mut self, runtime: &mut CudaRuntime) {
        if let Some(input) = self.input_cache.take() {
            runtime.recycle_matrix(input);
        }
    }

    fn get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
        LinearTrainingNode::get_data(self, runtime)
    }

    fn set_data(&mut self, data: &mut HostDataCursor, runtime: &CudaRuntime) {
        LinearTrainingNode::set_data(self, data, runtime);
    }
}

#[cfg(test)]
mod tests {
    use super::Activation;

    #[test]
    fn sigmoid_is_stable_and_has_the_expected_derivative() {
        assert_eq!(Activation::Sigmoid.forward(f32::INFINITY), 1.0);
        assert_eq!(Activation::Sigmoid.forward(f32::NEG_INFINITY), 0.0);
        assert!((Activation::Sigmoid.forward(0.0) - 0.5).abs() < 1.0e-7);
        assert!((Activation::Sigmoid.derivative(0.0) - 0.25).abs() < 1.0e-7);
    }
}
