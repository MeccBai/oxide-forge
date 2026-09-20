use crate::cuda::{CudaRuntime, container::Matrix};
use crate::net::metadata::{HostData, HostDataCursor};

pub mod builder;
pub mod node;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LearnConfig {
    pub learning_rate: f32,
    pub momentum: f32,
    pub batch_len: usize,
}

impl LearnConfig {
    pub fn single(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            momentum: 0.0,
            batch_len: 1,
        }
    }
}

pub trait GraphNode {
    fn forward(&mut self, inputs: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix>;

    fn backward(&mut self, gradients: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix>;

    /// Applies gradients produced by the preceding backward pass. Graph owns
    /// the scheduling of this phase and calls it only after every node has
    /// completed backward.
    fn learn(&mut self, config: LearnConfig, runtime: &mut CudaRuntime);

    fn clear_cache(&mut self, runtime: &mut CudaRuntime);

    /// Exports this node's parameters in metadata order. Stateless nodes use
    /// the empty default implementation.
    fn get_data(&self, _runtime: &CudaRuntime) -> Vec<HostData> {
        Vec::new()
    }

    /// Replaces parameters in-place from the graph's sequential host-data
    /// stream. Stateless nodes consume nothing.
    fn set_data(&mut self, _data: &mut HostDataCursor, _runtime: &CudaRuntime) {}
}
