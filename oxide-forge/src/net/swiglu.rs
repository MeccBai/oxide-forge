use crate::cuda::{CudaRuntime, container::Matrix};

use super::{
    linear::{Activation, Linear, LinearMomentum},
    metadata::HostData,
    node::{BinaryNode, BinaryOp, SingleNode, SingleType, TrainingBinaryNode, TrainingSingleNode},
};

pub struct InferenceSwiglu {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl InferenceSwiglu {
    /// Creates `down(SiLU(gate(input)) * up(input))` from three weight
    /// matrices. Gate and up are `[input, hidden]`; down is `[hidden, output]`.
    pub fn new(gate: Matrix, up: Matrix, down: Matrix) -> Self {
        Self {
            gate: identity_linear(gate),
            up: identity_linear(up),
            down: identity_linear(down),
        }
    }

    pub fn forward(&self, input: &Matrix, runtime: &mut CudaRuntime) -> Matrix {
        let gate = self.gate.forward(input, None, runtime, None);
        let gate = SingleNode::new(SingleType::Activation(Activation::Silu)).forward(gate, runtime);
        let up = self.up.forward(input, None, runtime, None);
        let hidden = BinaryNode::new(BinaryOp::Mul).forward_owned(vec![gate, up], runtime);
        let output = self.down.forward(&hidden, None, runtime, None);
        runtime.recycle_matrix(hidden);
        output
    }

    pub fn get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
        let mut data = self.gate.get_data(runtime);
        data.extend(self.up.get_data(runtime));
        data.extend(self.down.get_data(runtime));
        data
    }
}

pub struct TrainingSwiglu {
    gate: Linear,
    up: Linear,
    down: Linear,
    gate_optimizer: LinearMomentum,
    up_optimizer: LinearMomentum,
    down_optimizer: LinearMomentum,
    activation: TrainingSingleNode,
    product: TrainingBinaryNode,
    cache: Option<SwigluCache>,
}

struct SwigluCache {
    input: Matrix,
    hidden: Matrix,
}

impl TrainingSwiglu {
    pub fn new(gate: Matrix, up: Matrix, down: Matrix) -> Self {
        Self {
            gate: identity_linear(gate),
            up: identity_linear(up),
            down: identity_linear(down),
            gate_optimizer: LinearMomentum::default(),
            up_optimizer: LinearMomentum::default(),
            down_optimizer: LinearMomentum::default(),
            activation: TrainingSingleNode::new(SingleType::Activation(Activation::Silu)),
            product: TrainingBinaryNode::new(BinaryOp::Mul),
            cache: None,
        }
    }

    /// Consumes the input so it can become the parameter-gradient cache without
    /// an additional device copy.
    pub fn forward(&mut self, input: Matrix, runtime: &mut CudaRuntime) -> Matrix {
        self.clear_cache(runtime);

        let gate = self.gate.forward(&input, None, runtime, None);
        let gate = self.activation.forward(gate, runtime);
        let up = self.up.forward(&input, None, runtime, None);
        let hidden = self.product.forward(vec![gate, up], runtime);
        let output = self.down.forward(&hidden, None, runtime, None);

        self.cache = Some(SwigluCache { input, hidden });
        output
    }

    pub fn backward(
        &mut self,
        output_gradient: &Matrix,
        learning_rate: f32,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        let input_gradient = self.backward_accumulate(output_gradient, runtime);
        self.step(learning_rate, 0.0, 1, runtime);
        input_gradient
    }

    pub fn backward_accumulate(
        &mut self,
        output_gradient: &Matrix,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        let cache = self
            .cache
            .take()
            .expect("SwiGLU forward must run before backward");

        let hidden_gradient = self.down.input_gradient(output_gradient, runtime, None);
        self.down_optimizer
            .accumulate(&cache.hidden, output_gradient, None, runtime);

        let [gate_activation_gradient, up_gradient]: [Matrix; 2] = self
            .product
            .backward(hidden_gradient, runtime)
            .try_into()
            .unwrap_or_else(|_| unreachable!("SwiGLU product has two inputs"));
        let gate_gradient = self.activation.backward(gate_activation_gradient, runtime);

        let gate_input_gradient = self.gate.input_gradient(&gate_gradient, runtime, None);
        let up_input_gradient = self.up.input_gradient(&up_gradient, runtime, None);
        self.gate_optimizer
            .accumulate(&cache.input, &gate_gradient, None, runtime);
        self.up_optimizer
            .accumulate(&cache.input, &up_gradient, None, runtime);

        runtime.recycle_matrix(gate_gradient);
        runtime.recycle_matrix(up_gradient);
        runtime.recycle_matrix(cache.hidden);
        runtime.recycle_matrix(cache.input);

        BinaryNode::new(BinaryOp::Add)
            .forward_owned(vec![gate_input_gradient, up_input_gradient], runtime)
    }

    pub fn step(
        &mut self,
        learning_rate: f32,
        momentum: f32,
        batch_len: usize,
        runtime: &mut CudaRuntime,
    ) {
        self.gate_optimizer
            .step(&mut self.gate, learning_rate, momentum, batch_len, runtime);
        self.up_optimizer
            .step(&mut self.up, learning_rate, momentum, batch_len, runtime);
        self.down_optimizer
            .step(&mut self.down, learning_rate, momentum, batch_len, runtime);
    }

    pub fn get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
        let mut data = self.gate.get_data(runtime);
        data.extend(self.up.get_data(runtime));
        data.extend(self.down.get_data(runtime));
        data
    }

    pub fn clear_cache(&mut self, runtime: &mut CudaRuntime) {
        if let Some(cache) = self.cache.take() {
            runtime.recycle_matrix(cache.input);
            runtime.recycle_matrix(cache.hidden);
        }
        self.activation.clear_cache(runtime);
        self.product.clear_cache(runtime);
    }
}

fn identity_linear(weights: Matrix) -> Linear {
    Linear::new(weights, None, Activation::Identity)
}
