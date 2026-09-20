use crate::graph::MatrixConfig;
use crate::graph::draft::{BranchDraft, GraphDraft};
use crate::net::node::MatrixAxis;

/// Consuming builder for a complete graph draft.
pub struct GraphBuilder {
    _private: (),
}

impl GraphBuilder {
    pub fn start(_input: MatrixConfig) -> Self {
        todo!("GraphBuilder topology recording")
    }

    pub fn continue_from(_draft: GraphDraft) -> Self {
        todo!("continuing a GraphDraft")
    }

    pub fn then<N>(self, _node: N) -> Self {
        todo!("appending a node to GraphBuilder")
    }

    pub fn copy(self, _count: usize) -> Self {
        todo!("adding graph fan-out")
    }

    pub fn map<const N: usize>(self, _branches: [BranchDraft; N]) -> Self {
        todo!("mapping graph outputs through branches")
    }

    pub fn concat(self, _axis: MatrixAxis) -> Self {
        todo!("joining graph branches")
    }

    pub fn end(self) -> GraphDraft {
        todo!("finalizing GraphBuilder")
    }
}
