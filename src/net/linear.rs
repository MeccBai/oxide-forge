use crate::cuda::{
    self,
    BinaryOp::{Add, Sub},
    runtime::CudaRuntime,
};
use cuda::container::{Matrix, Vector};
use cuda_core::CudaStream;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::path::Path;

use crate::net::checkpoint;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    Identity,
    Gelu,
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
        }
    }
}

pub struct Linear {
    pub(super) weights: Matrix,
    pub(super) bias: Option<Vector>,
    pub(super) activation: Activation,
}

impl Linear {
    pub fn new(weights: Matrix, bias: Option<Vector>, activation: Activation) -> Self {
        Self {
            weights,
            bias,
            activation,
        }
    }

    pub fn dump_to_file<P: AsRef<Path>>(
        &self,
        path: P,
        runtime: &CudaRuntime,
    ) -> Result<(), Box<dyn Error>> {
        checkpoint::dump_linear_file(self, path.as_ref(), runtime)
    }

    pub fn load_from_file<P: AsRef<Path>>(
        path: P,
        runtime: &mut CudaRuntime,
    ) -> Result<Self, Box<dyn Error>> {
        checkpoint::load_linear_file(path.as_ref(), runtime)
    }

    pub fn forward(
        &self,
        input: &Matrix,
        residual: Option<&Matrix>,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        assert_eq!(input.cols(), self.weights.rows());
        let mut output = runtime.new_uninit_matrix(input.rows(), self.weights.cols());
        self.forward_into_on(input, residual, &mut output, runtime, runtime.stream());
        output
    }

    pub(crate) fn forward_into_on(
        &self,
        input: &Matrix,
        residual: Option<&Matrix>,
        output: &mut Matrix,
        runtime: &CudaRuntime,
        stream: &CudaStream,
    ) {
        self.affine_into_on(input, residual, output, runtime, stream);
        self.activate_on(output, runtime, stream);
    }

    pub fn affine(
        &self,
        input: &Matrix,
        residual: Option<&Matrix>,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        assert_eq!(input.cols(), self.weights.rows());
        let mut output = runtime.new_uninit_matrix(input.rows(), self.weights.cols());
        self.affine_into_on(input, residual, &mut output, runtime, runtime.stream());
        output
    }

    pub(crate) fn affine_into_on(
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

    pub fn activate(&self, output: &mut Matrix, runtime: &CudaRuntime) {
        self.activate_on(output, runtime, runtime.stream());
    }

    pub(crate) fn activate_on(
        &self,
        output: &mut Matrix,
        runtime: &CudaRuntime,
        stream: &CudaStream,
    ) {
        if let Activation::Gelu = self.activation {
            let activation = self.activation;
            output.for_each_on(runtime, stream, move |x| activation.forward(x));
        }
    }

    pub fn backward(
        &self,
        pre_activation: Option<&Matrix>,
        output_gradient: &Matrix,
        runtime: &mut CudaRuntime,
    ) -> (Matrix, Option<Vector>) {
        let mut gradient = runtime.matrix_copy(output_gradient);

        if let Activation::Gelu = self.activation {
            let pre_activation = pre_activation.expect("GELU backward requires pre-activation");
            assert_eq!(pre_activation.rows(), output_gradient.rows());
            assert_eq!(pre_activation.cols(), output_gradient.cols());
            let activation = self.activation;
            let mut derivative = runtime.matrix_copy(pre_activation);

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

    pub fn needs_pre_activation(&self) -> bool {
        matches!(self.activation, Activation::Gelu)
    }

    pub fn loss_rows(output: &Matrix, target: &Matrix, runtime: &mut CudaRuntime) -> Vector {
        let mut loss = runtime.matrix_binary(output, target, Sub);
        loss.for_each(runtime, |x| 0.5 * x * x);

        let cols = loss.cols() as f32;
        let mut row_loss = runtime.matrix_sum_rows(&loss);
        row_loss.scale(1.0 / cols, runtime);
        row_loss
    }

    pub fn learn(
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

    pub fn input_gradient(&self, gradient: &Matrix, runtime: &mut CudaRuntime) -> Matrix {
        assert_eq!(gradient.cols(), self.weights.cols());
        let weights_transpose = runtime.matrix_transpose(&self.weights);
        runtime.matrix_multiply(gradient, &weights_transpose)
    }

    pub fn backward_with_res(
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

        let mut gradient = runtime.matrix_copy(output_gradient);
        gradient.binary_assign(residual_gradient, Add, runtime);

        if let Activation::Gelu = self.activation {
            let activation = self.activation;
            let mut derivative = runtime.matrix_copy(pre_activation);

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
}
