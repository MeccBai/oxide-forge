use crate::cuda::{CudaRuntime, container::Matrix};

use super::{BinaryCache, BinaryNode, BinaryOp, TrainingBinaryNode};

impl BinaryNode {
    pub fn new(op: BinaryOp) -> Self {
        Self { op }
    }

    pub fn op(&self) -> BinaryOp {
        self.op
    }

    /// Executes without retaining state. Add accepts two or more inputs;
    /// every other operation accepts exactly two.
    pub fn forward(&self, inputs: &[&Matrix], runtime: &mut CudaRuntime) -> Matrix {
        validate_inputs(self.op, inputs);

        let mut output = match self.op {
            BinaryOp::Add => runtime.matrix_add(inputs[0], inputs[1]),
            BinaryOp::Sub => runtime.matrix_sub(inputs[0], inputs[1]),
            BinaryOp::Mul => runtime.matrix_mul(inputs[0], inputs[1]),
            BinaryOp::Div => runtime.matrix_div(inputs[0], inputs[1]),
            BinaryOp::MatMul => runtime.matrix_multiply(inputs[0], inputs[1]),
        };

        if let BinaryOp::Add = self.op {
            for input in &inputs[2..] {
                output.binary_assign(input, move |lhs, rhs| lhs + rhs, runtime);
            }
        }
        output
    }

    /// Consuming variant used between owned network outputs. Element-wise
    /// operations reuse the first input; MatMul allocates its differently
    /// shaped output. Consumed storage is returned to the runtime pool.
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
            BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
                let rhs = inputs.pop().unwrap();
                let mut output = inputs.pop().unwrap();
                match self.op {
                    BinaryOp::Sub => output.binary_assign(&rhs, move |lhs, rhs| lhs - rhs, runtime),
                    BinaryOp::Mul => output.binary_assign(&rhs, move |lhs, rhs| lhs * rhs, runtime),
                    BinaryOp::Div => output.binary_assign(&rhs, move |lhs, rhs| lhs / rhs, runtime),
                    _ => unreachable!(),
                }
                runtime.recycle_matrix(rhs);
                output
            }
            BinaryOp::MatMul => {
                let output = runtime.matrix_multiply(&inputs[0], &inputs[1]);
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

    /// Consumes branch outputs. Operations whose derivatives depend on their
    /// operands retain ownership; Add and Sub retain no forward values.
    pub fn forward(&mut self, inputs: Vec<Matrix>, runtime: &mut CudaRuntime) -> Matrix {
        self.clear_cache(runtime);
        self.input_count = inputs.len();
        self.forward_pending = true;

        match self.node.op {
            BinaryOp::Add | BinaryOp::Sub => self.node.forward_owned(inputs, runtime),
            BinaryOp::Mul | BinaryOp::Div | BinaryOp::MatMul => {
                validate_owned_inputs(self.node.op, &inputs);
                let output = match self.node.op {
                    BinaryOp::Mul => runtime.matrix_mul(&inputs[0], &inputs[1]),
                    BinaryOp::Div => runtime.matrix_div(&inputs[0], &inputs[1]),
                    BinaryOp::MatMul => runtime.matrix_multiply(&inputs[0], &inputs[1]),
                    _ => unreachable!(),
                };
                self.cache = Some(BinaryCache::Inputs(inputs));
                output
            }
        }
    }

    /// Borrowed training path for operations whose derivatives do not need
    /// their forward operands.
    pub fn forward_borrowed(&mut self, inputs: &[&Matrix], runtime: &mut CudaRuntime) -> Matrix {
        assert!(
            matches!(self.node.op, BinaryOp::Add | BinaryOp::Sub),
            "borrowed training forward is only available for Add and Sub"
        );
        self.clear_cache(runtime);
        let output = self.node.forward(inputs, runtime);
        self.input_count = inputs.len();
        self.forward_pending = true;
        output
    }

    /// Consumes the upstream gradient and returns one gradient per input in
    /// the same order as forward.
    pub fn backward(
        &mut self,
        mut output_gradient: Matrix,
        runtime: &mut CudaRuntime,
    ) -> Vec<Matrix> {
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
            BinaryOp::Sub => {
                let lhs_gradient = runtime.clone_matrix(&output_gradient);
                output_gradient.scale(-1.0, runtime);
                vec![lhs_gradient, output_gradient]
            }
            BinaryOp::Mul => {
                let inputs = take_inputs(&mut self.cache, "Mul");
                let lhs_gradient = runtime.matrix_mul(&output_gradient, &inputs[1]);
                let rhs_gradient = runtime.matrix_mul(&output_gradient, &inputs[0]);
                recycle_backward_inputs(output_gradient, inputs, runtime);
                vec![lhs_gradient, rhs_gradient]
            }
            BinaryOp::Div => {
                let inputs = take_inputs(&mut self.cache, "Div");
                let lhs_gradient = runtime.matrix_div(&output_gradient, &inputs[1]);
                let denominator = runtime.matrix_mul(&inputs[1], &inputs[1]);
                let numerator = runtime.matrix_mul(&output_gradient, &inputs[0]);
                let mut rhs_gradient = runtime.matrix_div(&numerator, &denominator);
                rhs_gradient.scale(-1.0, runtime);

                runtime.recycle_matrix(denominator);
                runtime.recycle_matrix(numerator);
                recycle_backward_inputs(output_gradient, inputs, runtime);
                vec![lhs_gradient, rhs_gradient]
            }
            BinaryOp::MatMul => {
                let inputs = take_inputs(&mut self.cache, "MatMul");
                let rhs_t = runtime.matrix_transpose(&inputs[1]);
                let lhs_gradient = runtime.matrix_multiply(&output_gradient, &rhs_t);
                let lhs_t = runtime.matrix_transpose(&inputs[0]);
                let rhs_gradient = runtime.matrix_multiply(&lhs_t, &output_gradient);

                runtime.recycle_matrix(rhs_t);
                runtime.recycle_matrix(lhs_t);
                recycle_backward_inputs(output_gradient, inputs, runtime);
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

fn take_inputs(cache: &mut Option<BinaryCache>, operation: &str) -> Vec<Matrix> {
    let BinaryCache::Inputs(inputs) = cache
        .take()
        .unwrap_or_else(|| panic!("{operation} forward input cache missing"));
    debug_assert_eq!(inputs.len(), 2);
    inputs
}

fn recycle_backward_inputs(
    output_gradient: Matrix,
    inputs: Vec<Matrix>,
    runtime: &mut CudaRuntime,
) {
    runtime.recycle_matrix(output_gradient);
    for input in inputs {
        runtime.recycle_matrix(input);
    }
}

fn validate_inputs(op: BinaryOp, inputs: &[&Matrix]) {
    assert!(
        inputs.len() >= 2,
        "a multi-input node needs at least two inputs"
    );
    if !matches!(op, BinaryOp::Add) {
        assert_eq!(inputs.len(), 2, "binary operation requires two inputs");
    }
    validate_shapes(op, inputs.iter().copied());
}

fn validate_owned_inputs(op: BinaryOp, inputs: &[Matrix]) {
    assert!(
        inputs.len() >= 2,
        "a multi-input node needs at least two inputs"
    );
    if !matches!(op, BinaryOp::Add) {
        assert_eq!(inputs.len(), 2, "binary operation requires two inputs");
    }
    validate_shapes(op, inputs.iter());
}

fn validate_shapes<'a>(op: BinaryOp, mut inputs: impl Iterator<Item = &'a Matrix>) {
    let first = inputs.next().unwrap();
    if matches!(op, BinaryOp::MatMul) {
        let second = inputs.next().unwrap();
        assert_eq!(
            first.cols(),
            second.rows(),
            "MatMul inner dimension mismatch"
        );
        return;
    }

    for input in inputs {
        assert_eq!(input.rows(), first.rows(), "node input row mismatch");
        assert_eq!(input.cols(), first.cols(), "node input column mismatch");
    }
}
