use super::DraftStep;
use crate::cuda::CudaRuntime;
use crate::graph::{Graph, InitConfig, MatrixConfig, Step, TrainState};
use crate::net::checkpoint::CheckpointResult;

pub struct GraphDraft {
    pub(crate) input_config: MatrixConfig,
    pub(crate) output_config: MatrixConfig,
    pub(crate) steps: Vec<DraftStep>,
}

impl GraphDraft {
    pub fn get_input_config(&self) -> MatrixConfig {
        self.input_config
    }
    pub fn get_output_config(&self) -> MatrixConfig {
        self.output_config
    }

    pub fn then(mut self, next: GraphDraft) -> GraphDraft {
        assert_eq!(
            self.output_config, next.input_config,
            "GraphDraft shape mismatch"
        );
        self.output_config = next.output_config;
        self.steps.extend(next.steps);
        self
    }

    pub fn init<const TRAINING: bool>(
        self,
        runtime: &mut CudaRuntime,
        config: InitConfig,
    ) -> CheckpointResult<Graph<TRAINING>> {
        let (loss, learning_rate) = match config {
            InitConfig::Random {
                loss,
                learning_rate,
            } => (loss, learning_rate),
            InitConfig::Load(path) => {
                return Err(format!(
                    "Graph checkpoint loading is not implemented for dynamic topology: {}",
                    path.display()
                )
                .into());
            }
        };
        let (steps, states) = compile_steps::<TRAINING>(self.steps, runtime, loss, &learning_rate);
        Ok(Graph {
            input_config: self.input_config,
            output_config: self.output_config,
            loss,
            learning_rate,
            steps,
            training: TRAINING.then(|| TrainState::new(states)),
        })
    }
}

pub(crate) fn compile_steps<const TRAINING: bool>(
    drafts: Vec<DraftStep>,
    runtime: &mut CudaRuntime,
    loss: crate::graph::Loss,
    learning_rate: &crate::graph::LearningRateScheduler,
) -> (Vec<Step<TRAINING>>, Vec<crate::graph::NodeTrainState>) {
    let mut steps = Vec::with_capacity(drafts.len());
    let mut states = Vec::with_capacity(drafts.len());
    for draft in drafts {
        match draft {
            DraftStep::Node(node) => {
                if TRAINING {
                    states.push(node.create_train_state());
                }
                steps.push(Step::Node(node));
            }
            DraftStep::Map(branches) => {
                let branches = branches
                    .into_iter()
                    .map(|branch| branch.compile::<TRAINING>(runtime, loss, learning_rate.clone()))
                    .collect();
                if TRAINING {
                    states.push(crate::graph::NodeTrainState::default());
                }
                steps.push(Step::Map(branches));
            }
            DraftStep::Copy(count) => {
                let node: Box<dyn crate::graph::GraphNode> = if TRAINING {
                    Box::new(crate::net::node::TrainingCopyNode::new(count))
                } else {
                    Box::new(crate::net::node::CopyNode::new(count))
                };
                if TRAINING {
                    states.push(node.create_train_state());
                }
                steps.push(Step::Node(node));
            }
            DraftStep::Concat(axis) => {
                let node: Box<dyn crate::graph::GraphNode> = if TRAINING {
                    Box::new(crate::net::node::TrainingConcatNode::new(axis))
                } else {
                    Box::new(crate::net::node::ConcatNode::new(axis))
                };
                if TRAINING {
                    states.push(node.create_train_state());
                }
                steps.push(Step::Node(node));
            }
            DraftStep::Linear(config) => {
                let weights = runtime.new_matrix(
                    crate::cuda::InitType::Random,
                    config.weights.rows,
                    config.weights.cols,
                    None,
                );
                let bias = config.bias.then(|| {
                    runtime.new_vector(crate::cuda::InitType::Zero, config.weights.cols, None)
                });
                let linear = crate::net::linear::Linear::new(weights, bias, config.activation);
                let node: Box<dyn crate::graph::GraphNode> = if TRAINING {
                    Box::new(crate::net::linear::LinearTrainingNode::new(linear))
                } else {
                    Box::new(linear)
                };
                if TRAINING {
                    states.push(node.create_train_state());
                }
                steps.push(Step::Node(node));
            }
        }
    }
    (steps, states)
}
