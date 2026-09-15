use crate::cuda::{CudaRuntime, container::Matrix};

use super::{BinaryCache, BinaryNode, BinaryOp, TrainingBinaryNode};

impl BinaryNode {
    pub fn new(op: BinaryOp) -> Self {
        Self { op }
    }

    pub fn op(&self) -> BinaryOp {
        self.op
    }

    /// Executes without retaining any state. Add accepts two or more inputs;
    /// element-wise multiplication currently has exactly two inputs.
    pub fn forward(&self, inputs: &[&Matrix], runtime: &mut CudaRuntime) -> Matrix {
        validate_inputs(self.op, inputs);

        let mut output = match self.op {
            BinaryOp::Add => runtime.matrix_add(inputs[0], inputs[1]),
            BinaryOp::Mul => runtime.matrix_mul(inputs[0], inputs[1]),
        };

        if let BinaryOp::Add = self.op {
            for input in &inputs[2..] {
                output.binary_assign(input, move |lhs, rhs| lhs + rhs, runtime);
            }
        }

        output
    }

    /// Consuming variant used between owned network outputs. Add reuses the
    /// first input as its output and returns the remaining inputs to the pool.
    pub fn forward_owned(&self, mut inputs: Vec<Matrix>, runtime: &mut CudaRuntime) -> Matrix {
        validate_owned_inputs(self.op, &inputs);
        match self.op {
            BinaryOp::Add => {
                let mut output = inputs.remove(0);
                for input in inputs {
                    output.binary_assign(&input, move |lhs, rhs| lhs + rhs, runtime);
                    runtime.recycle_matrix(input);
                }
                output
            }
            BinaryOp::Mul => {
                let output = runtime.matrix_mul(&inputs[0], &inputs[1]);
                for input in inputs {
                    runtime.recycle_matrix(input);
                }
                output
            }
        }
    }
}

impl TrainingBinaryNode {
    pub fn new(op: BinaryOp) -> Self {
        Self {
            node: BinaryNode::new(op),
            input_count: 0,
            forward_pending: false,
            cache: None,
        }
    }

    pub fn op(&self) -> BinaryOp {
        self.node.op()
    }

    /// Consumes branch outputs so values needed by backward can be retained
    /// without device copies. Values not needed by backward return to the pool.
    pub fn forward(&mut self, inputs: Vec<Matrix>, runtime: &mut CudaRuntime) -> Matrix {
        self.clear_cache(runtime);
        self.input_count = inputs.len();
        self.forward_pending = true;

        match self.node.op {
            BinaryOp::Add => self.node.forward_owned(inputs, runtime),
            BinaryOp::Mul => {
                validate_owned_inputs(self.node.op, &inputs);
                let output = runtime.matrix_mul(&inputs[0], &inputs[1]);
                self.cache = Some(BinaryCache::Inputs(inputs));
                output
            }
        }
    }

    /// Borrowed training path for residuals whose source is already retained
    /// by an upstream trainable module. It is valid for Add, which needs no
    /// forward values during backward.
    pub fn forward_borrowed(&mut self, inputs: &[&Matrix], runtime: &mut CudaRuntime) -> Matrix {
        assert_eq!(
            self.node.op,
            BinaryOp::Add,
            "borrowed training forward is only available for Add"
        );
        self.clear_cache(runtime);
        let output = self.node.forward(inputs, runtime);
        self.input_count = inputs.len();
        self.forward_pending = true;
        output
    }

    /// Consumes the upstream gradient. Add reuses it for one branch and makes
    /// only N-1 copies; Mul uses its cached forward inputs.
    pub fn backward(&mut self, output_gradient: Matrix, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        assert!(
            self.forward_pending,
            "training node backward requires a preceding forward"
        );
        self.forward_pending = false;

        match self.node.op {
            BinaryOp::Add => {
                let mut gradients = Vec::with_capacity(self.input_count);
                for _ in 1..self.input_count {
                    gradients.push(runtime.clone_matrix(&output_gradient));
                }
                gradients.push(output_gradient);
                gradients
            }
            BinaryOp::Mul => {
                let BinaryCache::Inputs(inputs) =
                    self.cache.take().expect("Mul forward input cache missing");
                debug_assert_eq!(inputs.len(), 2);
                let lhs_gradient = runtime.matrix_mul(&output_gradient, &inputs[1]);
                let rhs_gradient = runtime.matrix_mul(&output_gradient, &inputs[0]);
                runtime.recycle_matrix(output_gradient);
                for input in inputs {
                    runtime.recycle_matrix(input);
                }
                vec![lhs_gradient, rhs_gradient]
            }
        }
    }

    pub fn clear_cache(&mut self, runtime: &mut CudaRuntime) {
        if let Some(BinaryCache::Inputs(inputs)) = self.cache.take() {
            for input in inputs {
                runtime.recycle_matrix(input);
            }
        }
        self.input_count = 0;
        self.forward_pending = false;
    }
}

fn validate_inputs(op: BinaryOp, inputs: &[&Matrix]) {
    assert!(
        inputs.len() >= 2,
        "a multi-input node needs at least two inputs"
    );
    if let BinaryOp::Mul = op {
        assert_eq!(inputs.len(), 2, "Mul currently accepts exactly two inputs");
    }

    let rows = inputs[0].rows();
    let cols = inputs[0].cols();
    for input in &inputs[1..] {
        assert_eq!(input.rows(), rows, "node input row mismatch");
        assert_eq!(input.cols(), cols, "node input column mismatch");
    }
}

fn validate_owned_inputs(op: BinaryOp, inputs: &[Matrix]) {
    assert!(
        inputs.len() >= 2,
        "a multi-input node needs at least two inputs"
    );
    if let BinaryOp::Mul = op {
        assert_eq!(inputs.len(), 2, "Mul currently accepts exactly two inputs");
    }

    let rows = inputs[0].rows();
    let cols = inputs[0].cols();
    for input in &inputs[1..] {
        assert_eq!(input.rows(), rows, "node input row mismatch");
        assert_eq!(input.cols(), cols, "node input column mismatch");
    }
}
