mod branch;
mod root;

pub use branch::BranchDraft;
pub use root::GraphDraft;

use super::GraphNode;
use crate::net::linear::LinearConfig;
use crate::net::node::MatrixAxis;

pub(crate) enum DraftStep {
    Node(Box<dyn GraphNode>),
    Map(Vec<BranchDraft>),
    Copy(usize),
    Concat(MatrixAxis),
    Linear(LinearConfig),
}

impl DraftStep {
    pub(crate) fn output_configs(
        &self,
        inputs: &[super::MatrixConfig],
    ) -> Vec<super::MatrixConfig> {
        match self {
            Self::Node(node) => node.output_configs(inputs),
            Self::Map(branches) => {
                assert_eq!(inputs.len(), branches.len(), "Map branch count mismatch");
                inputs
                    .iter()
                    .zip(branches)
                    .map(|(input, branch)| branch.resolve_output(*input))
                    .collect()
            }
            Self::Copy(count) => {
                assert_eq!(inputs.len(), 1, "Copy expects one input");
                vec![inputs[0]; *count]
            }
            Self::Concat(axis) => {
                assert!(inputs.len() >= 2, "Concat requires at least two inputs");
                let first = inputs[0];
                let output = match axis {
                    MatrixAxis::Rows => {
                        assert!(inputs.iter().all(|input| input.cols == first.cols));
                        super::MatrixConfig::new(
                            inputs.iter().map(|input| input.rows).sum(),
                            first.cols,
                        )
                    }
                    MatrixAxis::Columns => {
                        assert!(inputs.iter().all(|input| input.rows == first.rows));
                        super::MatrixConfig::new(
                            first.rows,
                            inputs.iter().map(|input| input.cols).sum(),
                        )
                    }
                };
                vec![output]
            }
            Self::Linear(config) => {
                assert_eq!(inputs.len(), 1, "Linear expects one input");
                assert_eq!(
                    inputs[0].cols, config.weights.rows,
                    "Linear input width mismatch"
                );
                vec![super::MatrixConfig::new(
                    inputs[0].rows,
                    config.weights.cols,
                )]
            }
        }
    }
}
