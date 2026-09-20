use crate::cuda::{CudaRuntime, container::Matrix};
use crate::net::metadata::{HostData, HostDataCursor};
use serde::{Deserialize, Serialize};

pub mod builder;
pub mod draft;

pub use builder::{BranchBuilder, GraphBuilder};
pub use draft::{BranchDraft, GraphDraft};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatrixConfig {
    pub rows: usize,
    pub cols: usize,
}

impl MatrixConfig {
    pub const fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InitConfig {
    Random,
    Load(std::path::PathBuf),
}

/// Initialized executable graph. `TRAINING` controls whether backward and
/// optimizer phases are available.
pub struct Graph<const TRAINING: bool> {
    input_config: MatrixConfig,
    output_config: MatrixConfig,
}

/// Initialized executable branch, primarily used as a reusable graph segment.
pub struct Branch<const TRAINING: bool> {
    input_config: MatrixConfig,
    output_config: MatrixConfig,
}

impl<const TRAINING: bool> Graph<TRAINING> {
    pub fn get_input_config(&self) -> MatrixConfig {
        self.input_config
    }

    pub fn get_output_config(&self) -> MatrixConfig {
        self.output_config
    }
}

impl<const TRAINING: bool> Branch<TRAINING> {
    pub fn get_input_config(&self) -> MatrixConfig {
        self.input_config
    }

    pub fn get_output_config(&self) -> MatrixConfig {
        self.output_config
    }
}

impl<const TRAINING: bool> GraphNode for Graph<TRAINING> {
    fn forward(&mut self, _inputs: Vec<Matrix>, _runtime: &mut CudaRuntime) -> Vec<Matrix> {
        todo!("Graph forward scheduling")
    }

    fn backward(&mut self, _gradients: Vec<Matrix>, _runtime: &mut CudaRuntime) -> Vec<Matrix> {
        todo!("Graph backward scheduling")
    }

    fn learn(&mut self, _config: LearnConfig, _runtime: &mut CudaRuntime) {
        todo!("Graph optimizer scheduling")
    }

    fn clear_cache(&mut self, _runtime: &mut CudaRuntime) {
        todo!("clearing Graph node caches")
    }

    fn get_data(&self, _runtime: &CudaRuntime) -> Vec<HostData> {
        todo!("collecting Graph parameters")
    }

    fn set_data(&mut self, _data: &mut HostDataCursor, _runtime: &CudaRuntime) {
        todo!("restoring Graph parameters")
    }
}

impl<const TRAINING: bool> GraphNode for Branch<TRAINING> {
    fn forward(&mut self, _inputs: Vec<Matrix>, _runtime: &mut CudaRuntime) -> Vec<Matrix> {
        todo!("Branch forward scheduling")
    }

    fn backward(&mut self, _gradients: Vec<Matrix>, _runtime: &mut CudaRuntime) -> Vec<Matrix> {
        todo!("Branch backward scheduling")
    }

    fn learn(&mut self, _config: LearnConfig, _runtime: &mut CudaRuntime) {
        todo!("Branch optimizer scheduling")
    }

    fn clear_cache(&mut self, _runtime: &mut CudaRuntime) {
        todo!("clearing Branch node caches")
    }

    fn get_data(&self, _runtime: &CudaRuntime) -> Vec<HostData> {
        todo!("collecting Branch parameters")
    }

    fn set_data(&mut self, _data: &mut HostDataCursor, _runtime: &CudaRuntime) {
        todo!("restoring Branch parameters")
    }
}

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
