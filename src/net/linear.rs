use crate::cuda;
use cuda::container::{Matrix, Vector};

#[derive(Clone, Copy)]
pub enum Activation {
    Identity,
    Gelu,
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

    pub fn forward(&self, input: &Matrix, runtime: &cuda::runtime::CudaRuntime) -> Matrix {
        assert_eq!(input.cols(), self.weights.rows());
        let mut output = runtime.matrix_multiply(input, &self.weights);
        if let Some(ref bias) = self.bias {
            assert_eq!(output.cols(), bias.len());
            output.add_by_rows(bias, runtime);
        }
        match self.activation {
            Activation::Identity => {}
            Activation::Gelu => output.for_each(runtime, move |x| {
                0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh())
            }),
        }
        output
    }
}
