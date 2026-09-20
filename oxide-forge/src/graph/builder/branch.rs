use crate::graph::draft::BranchDraft;

/// Consuming builder for a reusable single-input branch draft.
pub struct BranchBuilder {
    _private: (),
}

impl BranchBuilder {
    pub fn start() -> Self {
        todo!("BranchBuilder topology recording")
    }

    pub fn continue_from(_draft: BranchDraft) -> Self {
        todo!("continuing a BranchDraft")
    }

    pub fn then<N>(self, _node: N) -> Self {
        todo!("appending a node to BranchBuilder")
    }

    pub fn end(self) -> BranchDraft {
        todo!("finalizing BranchBuilder")
    }
}
