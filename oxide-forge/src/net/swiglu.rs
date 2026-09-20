use crate::cuda::{CudaRuntime, container::Matrix};
use crate::graph::{GraphNode, LearnConfig, NodeTrainState};

use super::{
    linear::{Activation, Linear},
    metadata::{HostData, HostDataCursor},
    node::{BinaryNode, BinaryOp, SingleNode, SingleType},
};

/// Parameter block for `down(SiLU(gate(input)) * up(input))`.
/// Training topology and caches belong to `Graph<true>`.
pub struct Swiglu {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Swiglu {
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

fn identity_linear(weights: Matrix) -> Linear {
    Linear::new(weights, None, Activation::Identity)
}

impl GraphNode for Swiglu {
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
        let output = Swiglu::forward(self, &input, runtime);
        runtime.recycle_matrix(input);
        vec![output]
    }

    fn backward(&mut self, _gradients: Vec<Matrix>, _runtime: &mut CudaRuntime) -> Vec<Matrix> {
        panic!("SwiGLU training is assembled and owned by Graph<true>")
    }

    fn learn(&mut self, _config: LearnConfig, _runtime: &mut CudaRuntime) {
        panic!("SwiGLU training is assembled and owned by Graph<true>")
    }

    fn clear_cache(&mut self, _runtime: &mut CudaRuntime) {}

    fn get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
        Swiglu::get_data(self, runtime)
    }

    fn set_data(&mut self, data: &mut HostDataCursor, runtime: &CudaRuntime) {
        Swiglu::set_data(self, data, runtime);
    }
}
