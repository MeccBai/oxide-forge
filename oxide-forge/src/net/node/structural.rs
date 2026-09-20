use crate::cuda::{CudaRuntime, container::Matrix};
use crate::graph::{GraphNode, LearnConfig, MatrixConfig};

use super::{
    BinaryNode, BinaryOp, ConcatNode, CopyNode, MatrixAxis, ReduceNode, ReduceOp, SplitNode,
    TrainingConcatNode, TrainingCopyNode, TrainingReduceNode, TrainingSplitNode,
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

impl CopyNode {
    pub fn new(output_count: usize) -> Self {
        assert!(output_count >= 2, "Copy requires at least two outputs");
        Self { output_count }
    }

    pub fn output_count(&self) -> usize {
        self.output_count
    }

    /// Consumes one Matrix and returns independently owned, equal-shaped outputs.
    pub fn forward(&self, input: Matrix, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        let mut outputs = Vec::with_capacity(self.output_count);
        for _ in 1..self.output_count {
            outputs.push(runtime.clone_matrix(&input, None));
        }
        outputs.push(input);
        outputs
    }

    /// Copy's backward is an element-wise Sum reduction.
    pub fn backward(&self, output_gradients: Vec<Matrix>, runtime: &mut CudaRuntime) -> Matrix {
        assert_eq!(
            output_gradients.len(),
            self.output_count,
            "Copy gradient count mismatch"
        );
        ReduceNode::new(ReduceOp::Sum, self.output_count).forward(output_gradients, runtime)
    }
}

impl TrainingCopyNode {
    pub fn new(output_count: usize) -> Self {
        Self {
            node: CopyNode::new(output_count),
            forward_pending: false,
        }
    }

    pub fn output_count(&self) -> usize {
        self.node.output_count()
    }

    pub fn forward(&mut self, input: Matrix, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        assert!(
            !self.forward_pending,
            "Copy forward called twice without backward"
        );
        let outputs = self.node.forward(input, runtime);
        self.forward_pending = true;
        outputs
    }

    pub fn backward(&mut self, output_gradients: Vec<Matrix>, runtime: &mut CudaRuntime) -> Matrix {
        assert!(
            self.forward_pending,
            "Copy backward requires a preceding forward"
        );
        let gradient = self.node.backward(output_gradients, runtime);
        self.forward_pending = false;
        gradient
    }

    pub fn clear_cache(&mut self) {
        self.forward_pending = false;
    }
}

impl ReduceNode {
    pub fn new(op: ReduceOp, input_count: usize) -> Self {
        assert!(input_count >= 2, "Reduce requires at least two inputs");
        Self { op, input_count }
    }

    pub fn op(&self) -> ReduceOp {
        self.op
    }

    pub fn input_count(&self) -> usize {
        self.input_count
    }

    pub fn forward(&self, inputs: Vec<Matrix>, runtime: &mut CudaRuntime) -> Matrix {
        assert_eq!(
            inputs.len(),
            self.input_count,
            "Reduce input count mismatch"
        );
        let mut output = BinaryNode::new(BinaryOp::Add).forward_owned(inputs, runtime);
        if let ReduceOp::Mean = self.op {
            output.scale(1.0 / self.input_count as f32, runtime, None);
        }
        output
    }

    pub fn backward(&self, mut gradient: Matrix, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        if let ReduceOp::Mean = self.op {
            gradient.scale(1.0 / self.input_count as f32, runtime, None);
        }
        CopyNode::new(self.input_count).forward(gradient, runtime)
    }
}

impl TrainingReduceNode {
    pub fn new(op: ReduceOp, input_count: usize) -> Self {
        Self {
            node: ReduceNode::new(op, input_count),
            forward_pending: false,
        }
    }

    pub fn op(&self) -> ReduceOp {
        self.node.op()
    }

    pub fn input_count(&self) -> usize {
        self.node.input_count()
    }

    pub fn forward(&mut self, inputs: Vec<Matrix>, runtime: &mut CudaRuntime) -> Matrix {
        assert!(
            !self.forward_pending,
            "Reduce forward called twice without backward"
        );
        self.forward_pending = true;
        self.node.forward(inputs, runtime)
    }

    pub fn backward(&mut self, gradient: Matrix, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        assert!(
            self.forward_pending,
            "Reduce backward requires a preceding forward"
        );
        self.forward_pending = false;
        self.node.backward(gradient, runtime)
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

fn concat_config(axis: MatrixAxis, inputs: &[MatrixConfig]) -> MatrixConfig {
    assert!(inputs.len() >= 2, "Concat requires at least two inputs");
    let shapes = inputs
        .iter()
        .map(|config| (config.rows, config.cols))
        .collect::<Vec<_>>();
    let (rows, cols) = concatenated_shape(axis, &shapes);
    MatrixConfig::new(rows, cols)
}

fn split_configs(axis: MatrixAxis, sizes: &[usize], inputs: &[MatrixConfig]) -> Vec<MatrixConfig> {
    assert_eq!(inputs.len(), 1, "Split expects one input");
    let input = inputs[0];
    let expected = match axis {
        MatrixAxis::Rows => input.rows,
        MatrixAxis::Columns => input.cols,
    };
    assert_eq!(
        sizes.iter().sum::<usize>(),
        expected,
        "Split sizes do not cover the input"
    );
    sizes
        .iter()
        .map(|&size| match axis {
            MatrixAxis::Rows => MatrixConfig::new(size, input.cols),
            MatrixAxis::Columns => MatrixConfig::new(input.rows, size),
        })
        .collect()
}

fn reduce_config(input_count: usize, inputs: &[MatrixConfig]) -> Vec<MatrixConfig> {
    assert_eq!(inputs.len(), input_count, "Reduce input count mismatch");
    assert!(
        inputs.iter().all(|config| *config == inputs[0]),
        "Reduce shapes must match"
    );
    vec![inputs[0]]
}

impl GraphNode for ConcatNode {
    fn output_configs(&self, inputs: &[MatrixConfig]) -> Vec<MatrixConfig> {
        vec![concat_config(self.axis(), inputs)]
    }
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
    fn output_configs(&self, inputs: &[MatrixConfig]) -> Vec<MatrixConfig> {
        vec![concat_config(self.axis(), inputs)]
    }
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
    fn output_configs(&self, inputs: &[MatrixConfig]) -> Vec<MatrixConfig> {
        split_configs(self.axis(), self.sizes(), inputs)
    }
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
    fn output_configs(&self, inputs: &[MatrixConfig]) -> Vec<MatrixConfig> {
        split_configs(self.axis(), self.sizes(), inputs)
    }
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

impl GraphNode for CopyNode {
    fn output_configs(&self, inputs: &[MatrixConfig]) -> Vec<MatrixConfig> {
        assert_eq!(inputs.len(), 1, "Copy expects one input");
        vec![inputs[0]; self.output_count()]
    }
    fn forward(&mut self, mut inputs: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        assert_eq!(inputs.len(), 1, "Copy forward expects one matrix");
        CopyNode::forward(self, inputs.pop().unwrap(), runtime)
    }

    fn backward(&mut self, _gradients: Vec<Matrix>, _runtime: &mut CudaRuntime) -> Vec<Matrix> {
        panic!("inference CopyNode does not support backward; use TrainingCopyNode")
    }

    fn learn(&mut self, _config: LearnConfig, _runtime: &mut CudaRuntime) {}

    fn clear_cache(&mut self, _runtime: &mut CudaRuntime) {}
}

impl GraphNode for TrainingCopyNode {
    fn output_configs(&self, inputs: &[MatrixConfig]) -> Vec<MatrixConfig> {
        assert_eq!(inputs.len(), 1, "Copy expects one input");
        vec![inputs[0]; self.output_count()]
    }
    fn forward(&mut self, mut inputs: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        assert_eq!(inputs.len(), 1, "Copy forward expects one matrix");
        TrainingCopyNode::forward(self, inputs.pop().unwrap(), runtime)
    }

    fn backward(&mut self, gradients: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        vec![TrainingCopyNode::backward(self, gradients, runtime)]
    }

    fn learn(&mut self, _config: LearnConfig, _runtime: &mut CudaRuntime) {}

    fn clear_cache(&mut self, _runtime: &mut CudaRuntime) {
        TrainingCopyNode::clear_cache(self);
    }
}

impl GraphNode for ReduceNode {
    fn output_configs(&self, inputs: &[MatrixConfig]) -> Vec<MatrixConfig> {
        reduce_config(self.input_count(), inputs)
    }
    fn forward(&mut self, inputs: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        vec![ReduceNode::forward(self, inputs, runtime)]
    }

    fn backward(&mut self, _gradients: Vec<Matrix>, _runtime: &mut CudaRuntime) -> Vec<Matrix> {
        panic!("inference ReduceNode does not support backward; use TrainingReduceNode")
    }

    fn learn(&mut self, _config: LearnConfig, _runtime: &mut CudaRuntime) {}

    fn clear_cache(&mut self, _runtime: &mut CudaRuntime) {}
}

impl GraphNode for TrainingReduceNode {
    fn output_configs(&self, inputs: &[MatrixConfig]) -> Vec<MatrixConfig> {
        reduce_config(self.input_count(), inputs)
    }
    fn forward(&mut self, inputs: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        vec![TrainingReduceNode::forward(self, inputs, runtime)]
    }

    fn backward(&mut self, mut gradients: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        assert_eq!(gradients.len(), 1, "Reduce backward expects one gradient");
        TrainingReduceNode::backward(self, gradients.pop().unwrap(), runtime)
    }

    fn learn(&mut self, _config: LearnConfig, _runtime: &mut CudaRuntime) {}

    fn clear_cache(&mut self, _runtime: &mut CudaRuntime) {
        TrainingReduceNode::clear_cache(self);
    }
}
