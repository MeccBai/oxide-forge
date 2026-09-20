use crate::cuda::{CudaRuntime, container::Matrix};
use crate::graph::{GraphNode, LearnConfig, NodeTrainState};

use super::{
    linear::{Activation, Linear, LinearTrainingState},
    metadata::{HostData, HostDataCursor},
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

    pub fn set_data(&mut self, data: &mut HostDataCursor, runtime: &CudaRuntime) {
        self.gate.set_data(data, runtime);
        self.up.set_data(data, runtime);
        self.down.set_data(data, runtime);
    }
}

pub struct TrainingSwiglu {
    gate: Linear,
    up: Linear,
    down: Linear,
    gate_training: LinearTrainingState,
    up_training: LinearTrainingState,
    down_training: LinearTrainingState,
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
            gate_training: LinearTrainingState::with_parameter_count(2),
            up_training: LinearTrainingState::with_parameter_count(2),
            down_training: LinearTrainingState::with_parameter_count(2),
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

    pub fn backward(&mut self, output_gradient: &Matrix, runtime: &mut CudaRuntime) -> Matrix {
        self.backward_accumulate(output_gradient, runtime)
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
        self.down.accumulate_training(
            &mut self.down_training,
            &cache.hidden,
            output_gradient,
            None,
            runtime,
        );

        let [gate_activation_gradient, up_gradient]: [Matrix; 2] = self
            .product
            .backward(hidden_gradient, runtime)
            .try_into()
            .unwrap_or_else(|_| unreachable!("SwiGLU product has two inputs"));
        let gate_gradient = self.activation.backward(gate_activation_gradient, runtime);

        let gate_input_gradient = self.gate.input_gradient(&gate_gradient, runtime, None);
        let up_input_gradient = self.up.input_gradient(&up_gradient, runtime, None);
        self.gate.accumulate_training(
            &mut self.gate_training,
            &cache.input,
            &gate_gradient,
            None,
            runtime,
        );
        self.up.accumulate_training(
            &mut self.up_training,
            &cache.input,
            &up_gradient,
            None,
            runtime,
        );

        runtime.recycle_matrix(gate_gradient);
        runtime.recycle_matrix(up_gradient);
        runtime.recycle_matrix(cache.hidden);
        runtime.recycle_matrix(cache.input);

        BinaryNode::new(BinaryOp::Add)
            .forward_owned(vec![gate_input_gradient, up_input_gradient], runtime)
    }

    pub fn learn(
        &mut self,
        learning_rate: f32,
        momentum: f32,
        batch_len: usize,
        runtime: &mut CudaRuntime,
    ) {
        self.gate.learn_from_state(
            &mut self.gate_training,
            learning_rate,
            momentum,
            batch_len,
            runtime,
        );
        self.up.learn_from_state(
            &mut self.up_training,
            learning_rate,
            momentum,
            batch_len,
            runtime,
        );
        self.down.learn_from_state(
            &mut self.down_training,
            learning_rate,
            momentum,
            batch_len,
            runtime,
        );
    }

    pub fn get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
        let mut data = self.gate.get_data(runtime);
        data.extend(self.up.get_data(runtime));
        data.extend(self.down.get_data(runtime));
        data
    }

    pub fn set_data(&mut self, data: &mut HostDataCursor, runtime: &CudaRuntime) {
        self.gate.set_data(data, runtime);
        self.up.set_data(data, runtime);
        self.down.set_data(data, runtime);
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

impl GraphNode for InferenceSwiglu {
    fn create_train_state(&self) -> NodeTrainState {
        NodeTrainState::with_children(vec![
            self.gate.create_train_state(),
            self.up.create_train_state(),
            self.down.create_train_state(),
        ])
    }

    fn forward(&mut self, mut inputs: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        assert_eq!(inputs.len(), 1, "SwiGLU forward expects one matrix");
        let input = inputs.pop().unwrap();
        let output = InferenceSwiglu::forward(self, &input, runtime);
        runtime.recycle_matrix(input);
        vec![output]
    }

    fn backward(&mut self, _gradients: Vec<Matrix>, _runtime: &mut CudaRuntime) -> Vec<Matrix> {
        panic!("inference SwiGLU does not support backward; use TrainingSwiglu")
    }

    fn learn(&mut self, _config: LearnConfig, _runtime: &mut CudaRuntime) {
        panic!("inference SwiGLU does not support learning; use TrainingSwiglu")
    }

    fn clear_cache(&mut self, _runtime: &mut CudaRuntime) {}

    fn get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
        InferenceSwiglu::get_data(self, runtime)
    }

    fn set_data(&mut self, data: &mut HostDataCursor, runtime: &CudaRuntime) {
        InferenceSwiglu::set_data(self, data, runtime);
    }
}

impl GraphNode for TrainingSwiglu {
    fn forward(&mut self, mut inputs: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        assert_eq!(inputs.len(), 1, "SwiGLU forward expects one matrix");
        vec![TrainingSwiglu::forward(
            self,
            inputs.pop().unwrap(),
            runtime,
        )]
    }

    fn backward(&mut self, mut gradients: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        assert_eq!(gradients.len(), 1, "SwiGLU backward expects one gradient");
        let output_gradient = gradients.pop().unwrap();
        let input_gradient = self.backward_accumulate(&output_gradient, runtime);
        runtime.recycle_matrix(output_gradient);
        vec![input_gradient]
    }

    fn learn(&mut self, config: LearnConfig, runtime: &mut CudaRuntime) {
        TrainingSwiglu::learn(
            self,
            config.learning_rate,
            config.momentum,
            config.batch_len,
            runtime,
        );
    }

    fn clear_cache(&mut self, runtime: &mut CudaRuntime) {
        TrainingSwiglu::clear_cache(self, runtime);
    }

    fn get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
        TrainingSwiglu::get_data(self, runtime)
    }

    fn set_data(&mut self, data: &mut HostDataCursor, runtime: &CudaRuntime) {
        TrainingSwiglu::set_data(self, data, runtime);
    }
}
