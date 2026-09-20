use crate::cuda::CudaRuntime;
use crate::graph::{Branch, GraphDraft, InitConfig, MatrixConfig};
use crate::net::checkpoint::CheckpointResult;

/// Device-independent description of a reusable graph branch.
pub struct BranchDraft {
    pub(crate) input_config: MatrixConfig,
    pub(crate) output_config: MatrixConfig,
}

impl BranchDraft {
    pub fn get_input_config(&self) -> MatrixConfig {
        self.input_config
    }

    pub fn get_output_config(&self) -> MatrixConfig {
        self.output_config
    }

    pub fn then(self, _next: BranchDraft) -> BranchDraft {
        todo!("composing BranchDraft values")
    }

    pub fn into_graph(self) -> GraphDraft {
        todo!("promoting a BranchDraft to GraphDraft")
    }

    pub fn init<const TRAINING: bool>(
        self,
        _runtime: &mut CudaRuntime,
        _config: InitConfig,
    ) -> CheckpointResult<Branch<TRAINING>> {
        todo!("initializing a Branch from its draft")
    }
}
