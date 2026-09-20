use crate::cuda::CudaRuntime;
use crate::graph::{Graph, InitConfig, MatrixConfig};
use crate::net::checkpoint::CheckpointResult;

/// Device-independent description of a complete graph.
pub struct GraphDraft {
    pub(crate) input_config: MatrixConfig,
    pub(crate) output_config: MatrixConfig,
}

impl GraphDraft {
    pub fn get_input_config(&self) -> MatrixConfig {
        self.input_config
    }

    pub fn get_output_config(&self) -> MatrixConfig {
        self.output_config
    }

    pub fn then(self, _next: GraphDraft) -> GraphDraft {
        todo!("composing GraphDraft values")
    }

    pub fn init<const TRAINING: bool>(
        self,
        _runtime: &mut CudaRuntime,
        _config: InitConfig,
    ) -> CheckpointResult<Graph<TRAINING>> {
        todo!("initializing a Graph from its draft")
    }
}
