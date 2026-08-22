use crate::cuda::{
    self,
    BinaryOp::{Add, Sub},
    runtime::CudaRuntime,
};
use cuda::container::{Matrix, Vector};
use cuda_core::CudaStream;
use serde::{Deserialize, Serialize};

use crate::net::metadata::{HostData, MatrixMetadata, MetadataCursor, VectorMetadata};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    Identity,
    Gelu,
    Relu,
    Silu,
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
        }
    }
}

pub struct Linear {
    weights: Matrix,
    bias: Option<Vector>,
    activation: Activation,
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
        data.push(HostData::new(self.weights.to_host(runtime)));
        if let Some(bias) = &self.bias {
            data.push(HostData::new(bias.to_host(runtime)));
        }
        data
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
        if let Some(stream) = stream {
            stream.join(runtime.stream()).unwrap();
            self.forward_on(input, residual, &mut output, runtime, stream);
        } else {
            self.forward_default(input, residual, &mut output, runtime);
        }
        output
    }

    fn forward_default(
        &self,
        input: &Matrix,
        residual: Option<&Matrix>,
        output: &mut Matrix,
        runtime: &CudaRuntime,
    ) {
        self.affine_on(input, residual, output, runtime, runtime.stream());
        self.activate_on(output, runtime, runtime.stream());
    }

    fn forward_on(
        &self,
        input: &Matrix,
        residual: Option<&Matrix>,
        output: &mut Matrix,
        runtime: &CudaRuntime,
        stream: &CudaStream,
    ) {
        self.affine_on(input, residual, output, runtime, stream);
        self.activate_on(output, runtime, stream);
    }

    pub fn affine(
        &self,
        input: &Matrix,
        residual: Option<&Matrix>,
        runtime: &mut CudaRuntime,
        stream: Option<&CudaStream>,
    ) -> Matrix {
        assert_eq!(input.cols(), self.weights.rows());
        let mut output = runtime.new_uninit_matrix(input.rows(), self.weights.cols());
        if let Some(stream) = stream {
            stream.join(runtime.stream()).unwrap();
            self.affine_on(input, residual, &mut output, runtime, stream);
        } else {
            self.affine_default(input, residual, &mut output, runtime);
        }
        output
    }

    fn affine_default(
        &self,
        input: &Matrix,
        residual: Option<&Matrix>,
        output: &mut Matrix,
        runtime: &CudaRuntime,
    ) {
        self.affine_on(input, residual, output, runtime, runtime.stream());
    }

    fn affine_on(
        &self,
        input: &Matrix,
        residual: Option<&Matrix>,
        output: &mut Matrix,
        runtime: &CudaRuntime,
        stream: &CudaStream,
    ) {
        assert_eq!(input.cols(), self.weights.rows());
        assert_eq!(output.rows(), input.rows());
        assert_eq!(output.cols(), self.weights.cols());
        runtime.matrix_multiply_into_on(stream, input, &self.weights, output);
        if let Some(ref bias) = self.bias {
            assert_eq!(output.cols(), bias.len());
            output.binary_assign_by_rows_on(bias, Add, runtime, stream);
        }
        if let Some(residual_matrix) = residual {
            assert_eq!(output.rows(), residual_matrix.rows());
            assert_eq!(output.cols(), residual_matrix.cols());
            output.binary_assign_on(residual_matrix, Add, runtime, stream);
        }
    }

    pub fn activate(
        &self,
        output: &mut Matrix,
        runtime: &CudaRuntime,
        stream: Option<&CudaStream>,
    ) {
        if let Some(stream) = stream {
            self.activate_on(output, runtime, stream);
        } else {
            self.activate_default(output, runtime);
        }
    }

    fn activate_default(&self, output: &mut Matrix, runtime: &CudaRuntime) {
        self.activate_on(output, runtime, runtime.stream());
    }

    fn activate_on(&self, output: &mut Matrix, runtime: &CudaRuntime, stream: &CudaStream) {
        if !matches!(self.activation, Activation::Identity) {
            let activation = self.activation;
            output.for_each_on(runtime, stream, move |x| activation.forward(x));
        }
    }

    pub fn backward(
        &self,
        pre_activation: Option<&Matrix>,
        output_gradient: &Matrix,
        runtime: &mut CudaRuntime,
        stream: Option<&CudaStream>,
    ) -> (Matrix, Option<Vector>) {
        if let Some(stream) = stream {
            self.backward_on(pre_activation, output_gradient, runtime, stream)
        } else {
            self.backward_default(pre_activation, output_gradient, runtime)
        }
    }

    fn backward_default(
        &self,
        pre_activation: Option<&Matrix>,
        output_gradient: &Matrix,
        runtime: &mut CudaRuntime,
    ) -> (Matrix, Option<Vector>) {
        let mut gradient = runtime.clone_matrix(output_gradient);

        if !matches!(self.activation, Activation::Identity) {
            let pre_activation =
                pre_activation.expect("activation backward requires pre-activation");
            assert_eq!(pre_activation.rows(), output_gradient.rows());
            assert_eq!(pre_activation.cols(), output_gradient.cols());
            let activation = self.activation;
            let mut derivative = runtime.clone_matrix(pre_activation);

            derivative.for_each(runtime, move |x| activation.derivative(x));

            gradient.binary_assign(&derivative, crate::cuda::BinaryOp::Mul, runtime);
        }

        let bias_gradient = if self.bias.is_some() {
            let transposed = runtime.matrix_transpose(&gradient);
            Some(runtime.matrix_sum_rows(&transposed))
        } else {
            None
        };

        (gradient, bias_gradient)
    }

    fn backward_on(
        &self,
        pre_activation: Option<&Matrix>,
        output_gradient: &Matrix,
        runtime: &mut CudaRuntime,
        stream: &CudaStream,
    ) -> (Matrix, Option<Vector>) {
        let mut gradient = runtime.clone_matrix_on(output_gradient, stream);

        if !matches!(self.activation, Activation::Identity) {
            let pre_activation =
                pre_activation.expect("activation backward requires pre-activation");
            assert_eq!(pre_activation.rows(), output_gradient.rows());
            assert_eq!(pre_activation.cols(), output_gradient.cols());
            let activation = self.activation;
            let mut derivative = runtime.clone_matrix_on(pre_activation, stream);

            derivative.for_each_on(runtime, stream, move |x| activation.derivative(x));
            gradient.binary_assign_on(&derivative, crate::cuda::BinaryOp::Mul, runtime, stream);
        }

        let bias_gradient = if self.bias.is_some() {
            let transposed = runtime.matrix_transpose_on(&gradient, stream);
            Some(runtime.matrix_sum_rows_on(&transposed, stream))
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
        if let Some(stream) = stream {
            return Self::loss_rows_on(output, target, runtime, stream);
        }

        Self::loss_rows_default(output, target, runtime)
    }

    fn loss_rows_default(output: &Matrix, target: &Matrix, runtime: &mut CudaRuntime) -> Vector {
        let mut loss = runtime.matrix_binary(output, target, Sub);
        loss.for_each(runtime, |x| 0.5 * x * x);

        let cols = loss.cols() as f32;
        let mut row_loss = runtime.matrix_sum_rows(&loss);
        row_loss.scale(1.0 / cols, runtime);
        row_loss
    }

    fn loss_rows_on(
        output: &Matrix,
        target: &Matrix,
        runtime: &mut CudaRuntime,
        stream: &CudaStream,
    ) -> Vector {
        let mut loss = runtime.matrix_binary_on(stream, output, target, Sub);
        loss.for_each_on(runtime, stream, |x| 0.5 * x * x);

        let cols = loss.cols() as f32;
        let mut row_loss = runtime.matrix_sum_rows_on(&loss, stream);
        row_loss.scale_on(1.0 / cols, runtime, stream);
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
        if let Some(stream) = stream {
            self.learn_on(
                input,
                gradient,
                bias_gradient,
                learning_rate,
                runtime,
                stream,
            );
        } else {
            self.learn_default(input, gradient, bias_gradient, learning_rate, runtime);
        }
    }

    fn learn_default(
        &mut self,
        input: &Matrix,
        gradient: &Matrix,
        bias_gradient: Option<&Vector>,
        learning_rate: f32,
        runtime: &mut CudaRuntime,
    ) {
        let input_transpose = runtime.matrix_transpose(input);
        let mut weight_gradient = runtime.matrix_multiply(&input_transpose, gradient);
        weight_gradient.scale(learning_rate, runtime);
        self.weights.binary_assign(&weight_gradient, Sub, runtime);
        if let Some(ref mut bias) = self.bias {
            let mut scaled_bias_gradient = runtime
                .clone_vector(bias_gradient.expect("missing bias gradient for biased Linear"));
            scaled_bias_gradient.scale(learning_rate, runtime);
            bias.binary_assign(&scaled_bias_gradient, Sub, runtime);
        }
    }

    fn learn_on(
        &mut self,
        input: &Matrix,
        gradient: &Matrix,
        bias_gradient: Option<&Vector>,
        learning_rate: f32,
        runtime: &mut CudaRuntime,
        stream: &CudaStream,
    ) {
        let input_transpose = runtime.matrix_transpose_on(input, stream);
        let mut weight_gradient = runtime.matrix_multiply_on(stream, &input_transpose, gradient);
        weight_gradient.for_each_on(runtime, stream, move |x| x * learning_rate);
        self.weights
            .binary_assign_on(&weight_gradient, Sub, runtime, stream);
        if let Some(ref mut bias) = self.bias {
            let mut scaled_bias_gradient = runtime.clone_vector_on(
                bias_gradient.expect("missing bias gradient for biased Linear"),
                stream,
            );
            scaled_bias_gradient.scale_on(learning_rate, runtime, stream);
            bias.binary_assign_on(&scaled_bias_gradient, Sub, runtime, stream);
        }
    }

    pub fn input_gradient(
        &self,
        gradient: &Matrix,
        runtime: &mut CudaRuntime,
        stream: Option<&CudaStream>,
    ) -> Matrix {
        assert_eq!(gradient.cols(), self.weights.cols());
        if let Some(stream) = stream {
            self.input_gradient_on(gradient, runtime, stream)
        } else {
            self.input_gradient_default(gradient, runtime)
        }
    }

    fn input_gradient_default(&self, gradient: &Matrix, runtime: &mut CudaRuntime) -> Matrix {
        let weights_transpose = runtime.matrix_transpose(&self.weights);
        runtime.matrix_multiply(gradient, &weights_transpose)
    }

    fn input_gradient_on(
        &self,
        gradient: &Matrix,
        runtime: &mut CudaRuntime,
        stream: &CudaStream,
    ) -> Matrix {
        let weights_transpose = runtime.matrix_transpose_on(&self.weights, stream);
        runtime.matrix_multiply_on(stream, gradient, &weights_transpose)
    }

    pub fn backward_with_res(
        &self,
        pre_activation: &Matrix,
        output_gradient: &Matrix,
        residual_gradient: &Matrix,
        runtime: &mut CudaRuntime,
        stream: Option<&CudaStream>,
    ) -> (Matrix, Option<Vector>) {
        if let Some(stream) = stream {
            return self.backward_with_res_on(
                pre_activation,
                output_gradient,
                residual_gradient,
                runtime,
                stream,
            );
        }

        self.backward_with_res_default(pre_activation, output_gradient, residual_gradient, runtime)
    }

    fn backward_with_res_default(
        &self,
        pre_activation: &Matrix,
        output_gradient: &Matrix,
        residual_gradient: &Matrix,
        runtime: &mut CudaRuntime,
    ) -> (Matrix, Option<Vector>) {
        assert_eq!(pre_activation.rows(), output_gradient.rows());
        assert_eq!(pre_activation.cols(), output_gradient.cols());
        assert_eq!(output_gradient.rows(), residual_gradient.rows());
        assert_eq!(output_gradient.cols(), residual_gradient.cols());

        let mut gradient = runtime.clone_matrix(output_gradient);
        gradient.binary_assign(residual_gradient, Add, runtime);

        if !matches!(self.activation, Activation::Identity) {
            let activation = self.activation;
            let mut derivative = runtime.clone_matrix(pre_activation);

            derivative.for_each(runtime, move |x| activation.derivative(x));

            gradient.binary_assign(&derivative, crate::cuda::BinaryOp::Mul, runtime);
        }

        let bias_gradient = if self.bias.is_some() {
            let transposed = runtime.matrix_transpose(&gradient);
            Some(runtime.matrix_sum_rows(&transposed))
        } else {
            None
        };

        (gradient, bias_gradient)
    }

    fn backward_with_res_on(
        &self,
        pre_activation: &Matrix,
        output_gradient: &Matrix,
        residual_gradient: &Matrix,
        runtime: &mut CudaRuntime,
        stream: &CudaStream,
    ) -> (Matrix, Option<Vector>) {
        assert_eq!(pre_activation.rows(), output_gradient.rows());
        assert_eq!(pre_activation.cols(), output_gradient.cols());
        assert_eq!(output_gradient.rows(), residual_gradient.rows());
        assert_eq!(output_gradient.cols(), residual_gradient.cols());

        let mut gradient = runtime.clone_matrix_on(output_gradient, stream);
        gradient.binary_assign_on(residual_gradient, Add, runtime, stream);

        if !matches!(self.activation, Activation::Identity) {
            let activation = self.activation;
            let mut derivative = runtime.clone_matrix_on(pre_activation, stream);
            derivative.for_each_on(runtime, stream, move |x| activation.derivative(x));
            gradient.binary_assign_on(&derivative, crate::cuda::BinaryOp::Mul, runtime, stream);
        }

        let bias_gradient = if self.bias.is_some() {
            let transposed = runtime.matrix_transpose_on(&gradient, stream);
            Some(runtime.matrix_sum_rows_on(&transposed, stream))
        } else {
            None
        };

        (gradient, bias_gradient)
    }
}
