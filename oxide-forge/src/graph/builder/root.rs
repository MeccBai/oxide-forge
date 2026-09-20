use crate::graph::draft::{BranchDraft, GraphDraft};
use crate::graph::{GraphNode, MatrixConfig};
use crate::net::linear::LinearConfig;
use crate::net::node::MatrixAxis;

use super::super::draft::DraftStep;

pub struct GraphBuilder {
    input_config: MatrixConfig,
    current: Vec<MatrixConfig>,
    steps: Vec<DraftStep>,
}

impl GraphBuilder {
    pub fn start(input: MatrixConfig) -> Self {
        Self {
            input_config: input,
            current: vec![input],
            steps: Vec::new(),
        }
    }

    pub fn continue_from(draft: GraphDraft) -> Self {
        Self {
            input_config: draft.input_config,
            current: vec![draft.output_config],
            steps: draft.steps,
        }
    }

    pub fn then(mut self, config: LinearConfig) -> Self {
        let step = DraftStep::Linear(config);
        self.current = step.output_configs(&self.current);
        self.steps.push(step);
        self
    }

    pub fn then_node<N: GraphNode + 'static>(mut self, node: N) -> Self {
        self.current = node.output_configs(&self.current);
        self.steps.push(DraftStep::Node(Box::new(node)));
        self
    }

    pub fn copy(mut self, count: usize) -> Self {
        assert!(count >= 2, "Copy requires at least two outputs");
        assert_eq!(self.current.len(), 1, "Copy expects one graph value");
        self.current = vec![self.current[0]; count];
        self.steps.push(DraftStep::Copy(count));
        self
    }

    pub fn map<const N: usize>(mut self, mut branches: [BranchDraft; N]) -> Self {
        assert_eq!(self.current.len(), N, "Map input and branch count mismatch");
        self.current = self
            .current
            .iter()
            .zip(&mut branches)
            .map(|(&input, branch)| branch.attach(input))
            .collect();
        self.steps.push(DraftStep::Map(Vec::from(branches)));
        self
    }

    pub fn concat(mut self, axis: MatrixAxis) -> Self {
        let step = DraftStep::Concat(axis);
        self.current = step.output_configs(&self.current);
        self.steps.push(step);
        self
    }

    pub fn end(self) -> GraphDraft {
        assert_eq!(
            self.current.len(),
            1,
            "a complete GraphDraft must have one output"
        );
        GraphDraft {
            input_config: self.input_config,
            output_config: self.current[0],
            steps: self.steps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::GraphBuilder;
    use crate::graph::{BranchBuilder, MatrixConfig};
    use crate::net::node::MatrixAxis;

    #[test]
    fn copy_map_concat_propagates_shapes() {
        let draft = GraphBuilder::start(MatrixConfig::new(4, 8))
            .copy(2)
            .map([BranchBuilder::start().end(), BranchBuilder::start().end()])
            .concat(MatrixAxis::Rows)
            .end();

        assert_eq!(draft.get_input_config(), MatrixConfig::new(4, 8));
        assert_eq!(draft.get_output_config(), MatrixConfig::new(8, 8));
    }
}
