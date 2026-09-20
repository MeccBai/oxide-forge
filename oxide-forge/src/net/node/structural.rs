use crate::cuda::{CudaRuntime, container::Matrix};
use crate::graph::{GraphNode, LearnConfig};

use super::{
    BinaryNode, BinaryOp, ConcatNode, FanOutMode, FanOutNode, MatrixAxis, SplitNode,
    TrainingConcatNode, TrainingFanOutNode, TrainingSplitNode,
};

impl ConcatNode {
    pub fn new(axis: MatrixAxis) -> Self {
        Self { axis }
    }

    pub fn axis(&self) -> MatrixAxis {
        self.axis
    }

    pub fn forward(&self, inputs: &[&Matrix], runtime: &mut CudaRuntime) -> Matrix {
        assert!(inputs.len() >= 2, "Concat requires at least two inputs");
        match self.axis {
            MatrixAxis::Rows => runtime.matrix_concat_rows(inputs, None),
            MatrixAxis::Columns => runtime.matrix_concat_cols(inputs, None),
        }
    }

    pub fn forward_owned(&self, inputs: Vec<Matrix>, runtime: &mut CudaRuntime) -> Matrix {
        let output = {
            let borrowed = inputs.iter().collect::<Vec<_>>();
            self.forward(&borrowed, runtime)
        };
        for input in inputs {
            runtime.recycle_matrix(input);
        }
        output
    }
}

impl TrainingConcatNode {
    pub fn new(axis: MatrixAxis) -> Self {
        Self {
            node: ConcatNode::new(axis),
            input_shapes: None,
        }
    }

    pub fn axis(&self) -> MatrixAxis {
        self.node.axis()
    }

    pub fn forward(&mut self, inputs: Vec<Matrix>, runtime: &mut CudaRuntime) -> Matrix {
        self.input_shapes = Some(
            inputs
                .iter()
                .map(|input| (input.rows(), input.cols()))
                .collect(),
        );
        self.node.forward_owned(inputs, runtime)
    }

    pub fn backward(&mut self, output_gradient: Matrix, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        let shapes = self
            .input_shapes
            .take()
            .expect("Concat backward requires a preceding forward");
        let expected_shape = concatenated_shape(self.node.axis, &shapes);
        assert_eq!(
            (output_gradient.rows(), output_gradient.cols()),
            expected_shape,
            "Concat output gradient shape mismatch"
        );
        let sizes = shapes
            .iter()
            .map(|&(rows, cols)| match self.node.axis {
                MatrixAxis::Rows => rows,
                MatrixAxis::Columns => cols,
            })
            .collect::<Vec<_>>();
        let gradients = split(&output_gradient, self.node.axis, &sizes, runtime);
        runtime.recycle_matrix(output_gradient);
        gradients
    }

    pub fn clear_cache(&mut self) {
        self.input_shapes = None;
    }
}

impl SplitNode {
    pub fn new(axis: MatrixAxis, sizes: Vec<usize>) -> Self {
        assert!(!sizes.is_empty(), "Split requires at least one output");
        assert!(
            sizes.iter().all(|&size| size > 0),
            "Split sizes must be non-zero"
        );
        Self { axis, sizes }
    }

    pub fn axis(&self) -> MatrixAxis {
        self.axis
    }

    pub fn sizes(&self) -> &[usize] {
        &self.sizes
    }

    pub fn forward(&self, input: Matrix, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        let outputs = split(&input, self.axis, &self.sizes, runtime);
        runtime.recycle_matrix(input);
        outputs
    }
}

impl TrainingSplitNode {
    pub fn new(axis: MatrixAxis, sizes: Vec<usize>) -> Self {
        Self {
            node: SplitNode::new(axis, sizes),
            forward_pending: false,
        }
    }

    pub fn axis(&self) -> MatrixAxis {
        self.node.axis()
    }

    pub fn sizes(&self) -> &[usize] {
        self.node.sizes()
    }

    pub fn forward(&mut self, input: Matrix, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        assert!(
            !self.forward_pending,
            "Split forward called twice without backward"
        );
        let outputs = self.node.forward(input, runtime);
        self.forward_pending = true;
        outputs
    }

    pub fn backward(&mut self, output_gradients: Vec<Matrix>, runtime: &mut CudaRuntime) -> Matrix {
        assert!(
            self.forward_pending,
            "Split backward requires a preceding forward"
        );
        assert_eq!(
            output_gradients.len(),
            self.node.sizes.len(),
            "Split backward gradient count mismatch"
        );
        for (gradient, &size) in output_gradients.iter().zip(&self.node.sizes) {
            let actual = match self.node.axis {
                MatrixAxis::Rows => gradient.rows(),
                MatrixAxis::Columns => gradient.cols(),
            };
            assert_eq!(actual, size, "Split backward gradient shape mismatch");
        }
        self.forward_pending = false;
        ConcatNode::new(self.node.axis).forward_owned(output_gradients, runtime)
    }

    pub fn clear_cache(&mut self) {
        self.forward_pending = false;
    }
}

impl FanOutNode {
    pub fn new(mode: FanOutMode, output_count: usize) -> Self {
        assert!(output_count >= 2, "fan-out requires at least two outputs");
        Self { mode, output_count }
    }

    pub fn mode(&self) -> FanOutMode {
        self.mode
    }

    pub fn output_count(&self) -> usize {
        self.output_count
    }

    /// Consumes one Matrix and returns independently owned, equal-shaped
    /// outputs. Average distributes `input / output_count` to every branch.
    pub fn forward(&self, mut input: Matrix, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        if let FanOutMode::Average = self.mode {
            input.scale(1.0 / self.output_count as f32, runtime, None);
        }

        let mut outputs = Vec::with_capacity(self.output_count);
        for _ in 1..self.output_count {
            outputs.push(runtime.clone_matrix(&input, None));
        }
        outputs.push(input);
        outputs
    }

    /// Rejoins branch gradients according to the fan-out derivative.
    pub fn backward(&self, output_gradients: Vec<Matrix>, runtime: &mut CudaRuntime) -> Matrix {
        assert_eq!(
            output_gradients.len(),
            self.output_count,
            "fan-out gradient count mismatch"
        );
        let mut gradient = BinaryNode::new(BinaryOp::Add).forward_owned(output_gradients, runtime);
        if let FanOutMode::Average = self.mode {
            gradient.scale(1.0 / self.output_count as f32, runtime, None);
        }
        gradient
    }
}

impl TrainingFanOutNode {
    pub fn new(mode: FanOutMode, output_count: usize) -> Self {
        Self {
            node: FanOutNode::new(mode, output_count),
            forward_pending: false,
        }
    }

    pub fn mode(&self) -> FanOutMode {
        self.node.mode()
    }

    pub fn output_count(&self) -> usize {
        self.node.output_count()
    }

    pub fn forward(&mut self, input: Matrix, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        assert!(
            !self.forward_pending,
            "fan-out forward called twice without backward"
        );
        let outputs = self.node.forward(input, runtime);
        self.forward_pending = true;
        outputs
    }

    pub fn backward(&mut self, output_gradients: Vec<Matrix>, runtime: &mut CudaRuntime) -> Matrix {
        assert!(
            self.forward_pending,
            "fan-out backward requires a preceding forward"
        );
        let gradient = self.node.backward(output_gradients, runtime);
        self.forward_pending = false;
        gradient
    }

    pub fn clear_cache(&mut self) {
        self.forward_pending = false;
    }
}

fn concatenated_shape(axis: MatrixAxis, shapes: &[(usize, usize)]) -> (usize, usize) {
    let (first_rows, first_cols) = shapes[0];
    match axis {
        MatrixAxis::Rows => {
            let rows = shapes
                .iter()
                .map(|&(rows, cols)| {
                    assert_eq!(cols, first_cols, "Concat cached column mismatch");
                    rows
                })
                .try_fold(0usize, usize::checked_add)
                .expect("Concat cached row size overflow");
            (rows, first_cols)
        }
        MatrixAxis::Columns => {
            let cols = shapes
                .iter()
                .map(|&(rows, cols)| {
                    assert_eq!(rows, first_rows, "Concat cached row mismatch");
                    cols
                })
                .try_fold(0usize, usize::checked_add)
                .expect("Concat cached column size overflow");
            (first_rows, cols)
        }
    }
}

fn split(
    input: &Matrix,
    axis: MatrixAxis,
    sizes: &[usize],
    runtime: &mut CudaRuntime,
) -> Vec<Matrix> {
    match axis {
        MatrixAxis::Rows => runtime.matrix_split_rows(input, sizes, None),
        MatrixAxis::Columns => runtime.matrix_split_cols(input, sizes, None),
    }
}

impl GraphNode for ConcatNode {
    fn forward(&mut self, inputs: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        vec![self.forward_owned(inputs, runtime)]
    }

    fn backward(&mut self, _gradients: Vec<Matrix>, _runtime: &mut CudaRuntime) -> Vec<Matrix> {
        panic!("inference ConcatNode does not support backward; use TrainingConcatNode")
    }

    fn learn(&mut self, _config: LearnConfig, _runtime: &mut CudaRuntime) {}

    fn clear_cache(&mut self, _runtime: &mut CudaRuntime) {}
}

impl GraphNode for TrainingConcatNode {
    fn forward(&mut self, inputs: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        vec![TrainingConcatNode::forward(self, inputs, runtime)]
    }

    fn backward(&mut self, mut gradients: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        assert_eq!(gradients.len(), 1, "Concat backward expects one gradient");
        TrainingConcatNode::backward(self, gradients.pop().unwrap(), runtime)
    }

    fn learn(&mut self, _config: LearnConfig, _runtime: &mut CudaRuntime) {}

    fn clear_cache(&mut self, _runtime: &mut CudaRuntime) {
        TrainingConcatNode::clear_cache(self);
    }
}

impl GraphNode for SplitNode {
    fn forward(&mut self, mut inputs: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        assert_eq!(inputs.len(), 1, "Split forward expects one matrix");
        SplitNode::forward(self, inputs.pop().unwrap(), runtime)
    }

    fn backward(&mut self, _gradients: Vec<Matrix>, _runtime: &mut CudaRuntime) -> Vec<Matrix> {
        panic!("inference SplitNode does not support backward; use TrainingSplitNode")
    }

    fn learn(&mut self, _config: LearnConfig, _runtime: &mut CudaRuntime) {}

    fn clear_cache(&mut self, _runtime: &mut CudaRuntime) {}
}

impl GraphNode for TrainingSplitNode {
    fn forward(&mut self, mut inputs: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        assert_eq!(inputs.len(), 1, "Split forward expects one matrix");
        TrainingSplitNode::forward(self, inputs.pop().unwrap(), runtime)
    }

    fn backward(&mut self, gradients: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        vec![TrainingSplitNode::backward(self, gradients, runtime)]
    }

    fn learn(&mut self, _config: LearnConfig, _runtime: &mut CudaRuntime) {}

    fn clear_cache(&mut self, _runtime: &mut CudaRuntime) {
        TrainingSplitNode::clear_cache(self);
    }
}

impl GraphNode for FanOutNode {
    fn forward(&mut self, mut inputs: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        assert_eq!(inputs.len(), 1, "FanOut forward expects one matrix");
        FanOutNode::forward(self, inputs.pop().unwrap(), runtime)
    }

    fn backward(&mut self, _gradients: Vec<Matrix>, _runtime: &mut CudaRuntime) -> Vec<Matrix> {
        panic!("inference FanOutNode does not support backward; use TrainingFanOutNode")
    }

    fn learn(&mut self, _config: LearnConfig, _runtime: &mut CudaRuntime) {}

    fn clear_cache(&mut self, _runtime: &mut CudaRuntime) {}
}

impl GraphNode for TrainingFanOutNode {
    fn forward(&mut self, mut inputs: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        assert_eq!(inputs.len(), 1, "FanOut forward expects one matrix");
        TrainingFanOutNode::forward(self, inputs.pop().unwrap(), runtime)
    }

    fn backward(&mut self, gradients: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        vec![TrainingFanOutNode::backward(self, gradients, runtime)]
    }

    fn learn(&mut self, _config: LearnConfig, _runtime: &mut CudaRuntime) {}

    fn clear_cache(&mut self, _runtime: &mut CudaRuntime) {
        TrainingFanOutNode::clear_cache(self);
    }
}
