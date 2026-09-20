use crate::graph::GraphNode;
use crate::graph::draft::{BranchDraft, DraftStep};
use crate::net::linear::LinearConfig;

pub struct BranchBuilder {
    draft: BranchDraft,
}

impl BranchBuilder {
    pub fn start() -> Self {
        Self {
            draft: BranchDraft {
                input_config: None,
                output_config: None,
                steps: Vec::new(),
            },
        }
    }

    pub fn continue_from(draft: BranchDraft) -> Self {
        Self { draft }
    }

    pub fn then(mut self, config: LinearConfig) -> Self {
        self.draft.steps.push(DraftStep::Linear(config));
        if let Some(input) = self.draft.input_config {
            self.draft.attach(input);
        }
        self
    }

    pub fn then_node<N: GraphNode + 'static>(mut self, node: N) -> Self {
        self.draft.steps.push(DraftStep::Node(Box::new(node)));
        if let Some(input) = self.draft.input_config {
            self.draft.attach(input);
        }
        self
    }

    pub fn end(self) -> BranchDraft {
        self.draft
    }
}
