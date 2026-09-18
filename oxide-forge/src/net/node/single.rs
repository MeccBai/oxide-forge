use crate::cuda::{CudaRuntime, container::Matrix};

use super::{SingleCache, SingleNode, SingleType, TrainingSingleNode};

impl SingleNode {
    pub fn new(op: SingleType) -> Self {
        Self { op }
    }

    pub fn op(&self) -> SingleType {
        self.op
    }

    /// Inference path: consumes its input and recycles it when the operation
    /// necessarily creates a differently laid out result.
    pub fn forward(&self, mut input: Matrix, runtime: &mut CudaRuntime) -> Matrix {
        if let SingleType::Transpose = self.op {
            let output = runtime.matrix_transpose(&input, None);
            runtime.recycle_matrix(input);
            return output;
        }
        apply_in_place(self.op, &mut input, runtime);
        input
    }
}

impl TrainingSingleNode {
    pub fn new(op: SingleType) -> Self {
        Self {
            node: SingleNode::new(op),
            forward_pending: false,
            cache: None,
        }
    }

    pub fn op(&self) -> SingleType {
        self.node.op()
    }

    pub fn forward(&mut self, input: Matrix, runtime: &mut CudaRuntime) -> Matrix {
        self.clear_cache(runtime);
        self.forward_pending = true;

        match self.node.op {
            SingleType::Scale(_)
            | SingleType::Transpose
            | SingleType::Activation(crate::net::linear::Activation::Identity) => {
                self.node.forward(input, runtime)
            }
            SingleType::Activation(_) | SingleType::LayerNorm | SingleType::RmsNorm => {
                let mut output = runtime.clone_matrix(&input, None);
                apply_in_place(self.node.op, &mut output, runtime);
                self.cache = Some(SingleCache::Input(input));
                output
            }
            SingleType::Softmax => {
                let output = self.node.forward(input, runtime);
                self.cache = Some(SingleCache::Output(runtime.clone_matrix(&output, None)));
                output
            }
        }
    }

    pub fn backward(&mut self, mut output_gradient: Matrix, runtime: &mut CudaRuntime) -> Matrix {
        assert!(
            self.forward_pending,
            "training node backward requires a preceding forward"
        );
        self.forward_pending = false;

        match self.node.op {
            SingleType::Scale(scale) => {
                output_gradient.scale(scale, runtime);
                output_gradient
            }
            SingleType::Transpose => self.node.forward(output_gradient, runtime),
            SingleType::Activation(crate::net::linear::Activation::Identity) => output_gradient,
            SingleType::Activation(activation) => {
                let SingleCache::Input(input) = self
                    .cache
                    .take()
                    .expect("Activation forward input cache missing")
                else {
                    unreachable!()
                };
                let mut derivative = input;
                derivative.for_each(runtime, move |value| activation.derivative(value));
                output_gradient.binary_assign(
                    &derivative,
                    move |gradient, derivative| gradient * derivative,
                    runtime,
                );
                runtime.recycle_matrix(derivative);
                output_gradient
            }
            SingleType::Softmax => {
                let SingleCache::Output(probabilities) = self
                    .cache
                    .take()
                    .expect("Softmax forward output cache missing")
                else {
                    unreachable!()
                };
                let gradient =
                    runtime.softmax_rows_backward(&probabilities, &output_gradient, None);
                runtime.recycle_matrix(probabilities);
                runtime.recycle_matrix(output_gradient);
                gradient
            }
            SingleType::LayerNorm => {
                let SingleCache::Input(input) = self
                    .cache
                    .take()
                    .expect("LayerNorm forward input cache missing")
                else {
                    unreachable!()
                };
                let gradient = runtime.layer_norm_backward(&input, &output_gradient, None);
                runtime.recycle_matrix(input);
                runtime.recycle_matrix(output_gradient);
                gradient
            }
            SingleType::RmsNorm => {
                let SingleCache::Input(input) = self
                    .cache
                    .take()
                    .expect("RmsNorm forward input cache missing")
                else {
                    unreachable!()
                };
                let gradient = runtime.rms_norm_backward(&input, &output_gradient, None);
                runtime.recycle_matrix(input);
                runtime.recycle_matrix(output_gradient);
                gradient
            }
        }
    }

    pub fn clear_cache(&mut self, runtime: &mut CudaRuntime) {
        if let Some(cache) = self.cache.take() {
            let matrix = match cache {
                SingleCache::Input(matrix) | SingleCache::Output(matrix) => matrix,
            };
            runtime.recycle_matrix(matrix);
        }
        self.forward_pending = false;
    }
}

fn apply_in_place(op: SingleType, matrix: &mut Matrix, runtime: &CudaRuntime) {
    match op {
        SingleType::Activation(activation) => {
            matrix.for_each(runtime, move |value| activation.forward(value))
        }
        SingleType::Softmax => matrix.softmax_rows(runtime),
        SingleType::RmsNorm => matrix.rms_norm(runtime),
        SingleType::LayerNorm => matrix.layer_norm(runtime),
        SingleType::Scale(scale) => matrix.scale(scale, runtime),
        SingleType::Transpose => unreachable!("transpose is not an in-place operation"),
    }
}
