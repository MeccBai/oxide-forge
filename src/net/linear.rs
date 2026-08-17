use crate::cuda::{
    self,
    BinaryOp::{Add, Sub},
    runtime::CudaRuntime,
};
use cuda::container::{Matrix, Vector};

#[derive(Clone, Copy)]
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
    weights: Matrix,
    bias: Option<Vector>,
    activation: Activation,
}

impl Linear {
    pub fn new(weights: Matrix, bias: Option<Vector>, activation: Activation) -> Self {
        Self {
            weights,
            bias,
            activation,
        }
    }

    pub fn forward(
        &self,
        input: &Matrix,
        residual: Option<&Matrix>,
        runtime: &CudaRuntime,
    ) -> Matrix {
        let mut output = self.affine(input, residual, runtime);
        self.activate(&mut output, runtime);
        output
    }

    pub fn affine(
        &self,
        input: &Matrix,
        residual: Option<&Matrix>,
        runtime: &CudaRuntime,
    ) -> Matrix {
        assert_eq!(input.cols(), self.weights.rows());
        let mut output = runtime.matrix_multiply(input, &self.weights);
        if let Some(ref bias) = self.bias {
            assert_eq!(output.cols(), bias.len());
            output.binary_assign_by_rows(bias, Add, runtime);
        }
        if let Some(residual_matrix) = residual {
            assert_eq!(output.rows(), residual_matrix.rows());
            assert_eq!(output.cols(), residual_matrix.cols());
            output.binary_assign(residual_matrix, Add, runtime);
        }
        output
    }

    pub fn activate(&self, output: &mut Matrix, runtime: &CudaRuntime) {
        if let Activation::Gelu = self.activation {
            let activation = self.activation;
            output.for_each(runtime, move |x| activation.forward(x));
        }
    }

    pub fn backward(
        &self,
        pre_activation: Option<&Matrix>,
        output_gradient: &Matrix,
        runtime: &CudaRuntime,
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
            let mut transposed = runtime.matrix_transpose(&gradient);
            Some(transposed.sum_rows(runtime))
        } else {
            None
        };

        (gradient, bias_gradient)
    }

    pub fn needs_pre_activation(&self) -> bool {
        matches!(self.activation, Activation::Gelu)
    }

    pub fn loss_rows(output: &Matrix, target: &Matrix, runtime: &CudaRuntime) -> Vector {
        let mut loss = runtime.matrix_binary(output, target, Sub);
        loss.for_each(runtime, |x| 0.5 * x * x);

        let cols = loss.cols() as f32;
        let mut row_loss = loss.sum_rows(runtime);
        row_loss.scale(1.0 / cols, runtime);
        row_loss
    }

    pub fn learn(
        &mut self,
        input: &Matrix,
        gradient: &Matrix,
        bias_gradient: Option<&Vector>,
        learning_rate: f32,
        runtime: &CudaRuntime,
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

    pub fn input_gradient(&self, gradient: &Matrix, runtime: &CudaRuntime) -> Matrix {
        assert_eq!(gradient.cols(), self.weights.cols());
        let weights_transpose = runtime.matrix_transpose(&self.weights);
        runtime.matrix_multiply(gradient, &weights_transpose)
    }

    pub fn backward_with_res(
        &self,
        pre_activation: &Matrix,
        output_gradient: &Matrix,
        residual_gradient: &Matrix,
        runtime: &CudaRuntime,
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
            let mut transposed = runtime.matrix_transpose(&gradient);
            Some(transposed.sum_rows(runtime))
        } else {
            None
        };

        (gradient, bias_gradient)
    }
}
