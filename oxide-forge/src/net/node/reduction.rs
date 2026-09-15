use crate::cuda::{CudaRuntime, container::Matrix};

use super::{RowReduceNode, RowReduction, TrainingRowReduceNode};

impl RowReduceNode {
    pub fn new(reduction: RowReduction) -> Self {
        Self { reduction }
    }

    pub fn reduction(&self) -> RowReduction {
        self.reduction
    }

    /// Consumes `[rows, cols]` and returns a `[rows, 1]` Matrix.
    pub fn forward(&self, input: Matrix, runtime: &mut CudaRuntime) -> Matrix {
        assert!(input.cols() > 0, "row reduction requires non-empty rows");
        let cols = input.cols();
        let output = runtime.matrix_sum_rows(&input);
        runtime.recycle_matrix(input);
        let mut output = runtime.vector_into_matrix(output);
        if let RowReduction::Mean = self.reduction {
            output.scale(1.0 / cols as f32, runtime);
        }
        output
    }
}

impl TrainingRowReduceNode {
    pub fn new(reduction: RowReduction) -> Self {
        Self {
            node: RowReduceNode::new(reduction),
            input_shape: None,
        }
    }

    pub fn reduction(&self) -> RowReduction {
        self.node.reduction()
    }

    pub fn forward(&mut self, input: Matrix, runtime: &mut CudaRuntime) -> Matrix {
        self.input_shape = Some((input.rows(), input.cols()));
        self.node.forward(input, runtime)
    }

    pub fn backward(&mut self, output_gradient: Matrix, runtime: &mut CudaRuntime) -> Matrix {
        let (rows, cols) = self
            .input_shape
            .take()
            .expect("row reduction backward requires a preceding forward");
        assert_eq!(output_gradient.rows(), rows, "row gradient count mismatch");
        assert_eq!(output_gradient.cols(), 1, "row gradient must be [rows, 1]");

        let gradient = runtime.matrix_into_vector(output_gradient);
        let expanded_t = runtime.broadcast(&gradient, cols);
        runtime.recycle_vector(gradient);
        let mut expanded = runtime.matrix_transpose(&expanded_t);
        runtime.recycle_matrix(expanded_t);
        if let RowReduction::Mean = self.node.reduction {
            expanded.scale(1.0 / cols as f32, runtime);
        }
        expanded
    }

    pub fn clear_cache(&mut self) {
        self.input_shape = None;
    }
}
