use crate::cuda::{
    CudaRuntime,
    container::{Matrix, Vector},
};
use crate::net::metadata::{HostData, HostDataCursor};
pub use crate::net::mlp::Loss;
use serde::{Deserialize, Serialize};

pub mod builder;
pub mod draft;
pub mod schedule;
pub mod state;

pub use builder::{BranchBuilder, GraphBuilder};
pub use draft::{BranchDraft, GraphDraft};
pub use schedule::{Decay, DecayStage, LearningRateScheduler};
pub use state::{NodeCache, NodeTrainState, ParameterTrainState, TrainState};

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

#[derive(Clone, Debug, PartialEq)]
pub enum InitConfig {
    Random {
        loss: Loss,
        learning_rate: LearningRateScheduler,
    },
    Load(std::path::PathBuf),
}

pub(crate) enum Step<const TRAINING: bool> {
    Node(Box<dyn GraphNode>),
    Map(Vec<Branch<TRAINING>>),
}

impl<const TRAINING: bool> Step<TRAINING> {
    fn forward(&mut self, inputs: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        match self {
            Self::Node(node) => node.forward(inputs, runtime),
            Self::Map(branches) => {
                assert_eq!(inputs.len(), branches.len(), "Map branch count mismatch");
                inputs
                    .into_iter()
                    .zip(branches)
                    .flat_map(|(input, branch)| branch.execute_forward(vec![input], runtime))
                    .collect()
            }
        }
    }

    fn backward(&mut self, gradients: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        match self {
            Self::Node(node) => node.backward(gradients, runtime),
            Self::Map(branches) => {
                assert_eq!(
                    gradients.len(),
                    branches.len(),
                    "Map gradient count mismatch"
                );
                gradients
                    .into_iter()
                    .zip(branches)
                    .flat_map(|(gradient, branch)| branch.execute_backward(vec![gradient], runtime))
                    .collect()
            }
        }
    }

    fn learn(&mut self, config: LearnConfig, runtime: &mut CudaRuntime) {
        match self {
            Self::Node(node) => node.learn(config, runtime),
            Self::Map(branches) => branches
                .iter_mut()
                .for_each(|branch| branch.execute_learn(config, runtime)),
        }
    }

    fn clear_cache(&mut self, runtime: &mut CudaRuntime) {
        match self {
            Self::Node(node) => node.clear_cache(runtime),
            Self::Map(branches) => branches
                .iter_mut()
                .for_each(|branch| branch.execute_clear_cache(runtime)),
        }
    }

    fn get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
        match self {
            Self::Node(node) => node.get_data(runtime),
            Self::Map(branches) => branches
                .iter()
                .flat_map(|branch| branch.execute_get_data(runtime))
                .collect(),
        }
    }

    fn set_data(&mut self, data: &mut HostDataCursor, runtime: &CudaRuntime) {
        match self {
            Self::Node(node) => node.set_data(data, runtime),
            Self::Map(branches) => branches
                .iter_mut()
                .for_each(|branch| branch.execute_set_data(data, runtime)),
        }
    }
}

pub struct Graph<const TRAINING: bool> {
    pub(crate) input_config: MatrixConfig,
    pub(crate) output_config: MatrixConfig,
    pub(crate) loss: Loss,
    pub(crate) learning_rate: LearningRateScheduler,
    pub(crate) steps: Vec<Step<TRAINING>>,
    pub(crate) training: Option<TrainState>,
}

pub struct Branch<const TRAINING: bool> {
    pub(crate) input_config: MatrixConfig,
    pub(crate) output_config: MatrixConfig,
    pub(crate) loss: Loss,
    pub(crate) learning_rate: LearningRateScheduler,
    pub(crate) steps: Vec<Step<TRAINING>>,
    pub(crate) training: Option<TrainState>,
}

macro_rules! impl_model {
    ($type:ident) => {
        impl<const TRAINING: bool> $type<TRAINING> {
            pub fn get_input_config(&self) -> MatrixConfig {
                self.input_config
            }
            pub fn get_output_config(&self) -> MatrixConfig {
                self.output_config
            }
            pub fn get_loss(&self) -> Loss {
                self.loss
            }
            pub fn learning_rate(&self) -> &LearningRateScheduler {
                &self.learning_rate
            }
            pub fn learning_rate_mut(&mut self) -> &mut LearningRateScheduler {
                &mut self.learning_rate
            }
            pub fn train_state(&self) -> Option<&TrainState> {
                self.training.as_ref()
            }
            pub fn train_state_mut(&mut self) -> Option<&mut TrainState> {
                self.training.as_mut()
            }

            fn execute_forward(
                &mut self,
                mut values: Vec<Matrix>,
                runtime: &mut CudaRuntime,
            ) -> Vec<Matrix> {
                for step in &mut self.steps {
                    values = step.forward(values, runtime);
                }
                values
            }

            fn execute_backward(
                &mut self,
                mut gradients: Vec<Matrix>,
                runtime: &mut CudaRuntime,
            ) -> Vec<Matrix> {
                assert!(TRAINING, "backward is unavailable for an inference graph");
                for step in self.steps.iter_mut().rev() {
                    gradients = step.backward(gradients, runtime);
                }
                gradients
            }

            fn execute_learn(&mut self, config: LearnConfig, runtime: &mut CudaRuntime) {
                assert!(
                    TRAINING,
                    "optimizer steps are unavailable for an inference graph"
                );
                for step in &mut self.steps {
                    step.learn(config, runtime);
                }
            }

            fn execute_clear_cache(&mut self, runtime: &mut CudaRuntime) {
                for step in &mut self.steps {
                    step.clear_cache(runtime);
                }
                if let Some(training) = &mut self.training {
                    for state in training.nodes_mut() {
                        state.clear_cache(runtime);
                    }
                }
            }

            fn execute_get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
                self.steps
                    .iter()
                    .flat_map(|step| step.get_data(runtime))
                    .collect()
            }

            fn execute_set_data(&mut self, data: &mut HostDataCursor, runtime: &CudaRuntime) {
                for step in &mut self.steps {
                    step.set_data(data, runtime);
                }
            }
        }
    };
}

impl_model!(Graph);
impl_model!(Branch);

impl<const TRAINING: bool> Graph<TRAINING> {
    pub fn forward(&mut self, input: Matrix, runtime: &mut CudaRuntime) -> Matrix {
        assert_eq!(
            (input.rows(), input.cols()),
            (self.input_config.rows, self.input_config.cols)
        );
        let mut outputs = self.execute_forward(vec![input], runtime);
        assert_eq!(outputs.len(), 1, "a complete Graph must produce one output");
        outputs.pop().unwrap()
    }
}

impl Graph<true> {
    pub fn backward(&mut self, gradient: Matrix, runtime: &mut CudaRuntime) -> Matrix {
        let mut gradients = self.execute_backward(vec![gradient], runtime);
        assert_eq!(
            gradients.len(),
            1,
            "a complete Graph must return one input gradient"
        );
        gradients.pop().unwrap()
    }

    pub fn step(&mut self, momentum: f32, batch_len: usize, runtime: &mut CudaRuntime) {
        let config = LearnConfig {
            learning_rate: self.learning_rate.current(),
            momentum,
            batch_len,
        };
        self.execute_learn(config, runtime);
        self.learning_rate.advance();
    }

    pub fn train_step(
        &mut self,
        input: Matrix,
        target: &Matrix,
        positive_weight: f32,
        dice_weight: f32,
        momentum: f32,
        runtime: &mut CudaRuntime,
    ) -> Vector {
        let output = self.forward(input, runtime);
        let (loss, gradient) = self.loss.loss_and_output_gradient(
            &output,
            target,
            positive_weight,
            dice_weight,
            runtime,
        );
        let input_gradient = self.backward(gradient, runtime);
        self.step(momentum, 1, runtime);
        runtime.recycle_matrix(output);
        runtime.recycle_matrix(input_gradient);
        loss
    }
}

impl<const TRAINING: bool> GraphNode for Graph<TRAINING> {
    fn forward(&mut self, inputs: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        self.execute_forward(inputs, runtime)
    }
    fn backward(&mut self, gradients: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        self.execute_backward(gradients, runtime)
    }
    fn learn(&mut self, config: LearnConfig, runtime: &mut CudaRuntime) {
        self.execute_learn(config, runtime)
    }
    fn clear_cache(&mut self, runtime: &mut CudaRuntime) {
        self.execute_clear_cache(runtime)
    }
    fn get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
        self.execute_get_data(runtime)
    }
    fn set_data(&mut self, data: &mut HostDataCursor, runtime: &CudaRuntime) {
        self.execute_set_data(data, runtime)
    }
}

impl<const TRAINING: bool> GraphNode for Branch<TRAINING> {
    fn forward(&mut self, inputs: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        self.execute_forward(inputs, runtime)
    }
    fn backward(&mut self, gradients: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        self.execute_backward(gradients, runtime)
    }
    fn learn(&mut self, config: LearnConfig, runtime: &mut CudaRuntime) {
        self.execute_learn(config, runtime)
    }
    fn clear_cache(&mut self, runtime: &mut CudaRuntime) {
        self.execute_clear_cache(runtime)
    }
    fn get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
        self.execute_get_data(runtime)
    }
    fn set_data(&mut self, data: &mut HostDataCursor, runtime: &CudaRuntime) {
        self.execute_set_data(data, runtime)
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
    fn create_train_state(&self) -> NodeTrainState {
        NodeTrainState::default()
    }
    fn output_configs(&self, inputs: &[MatrixConfig]) -> Vec<MatrixConfig> {
        assert_eq!(inputs.len(), 1, "shape-preserving node expects one input");
        inputs.to_vec()
    }
    fn forward(&mut self, inputs: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix>;
    fn backward(&mut self, gradients: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix>;
    fn learn(&mut self, config: LearnConfig, runtime: &mut CudaRuntime);
    fn clear_cache(&mut self, runtime: &mut CudaRuntime);
    fn get_data(&self, _runtime: &CudaRuntime) -> Vec<HostData> {
        Vec::new()
    }
    fn set_data(&mut self, _data: &mut HostDataCursor, _runtime: &CudaRuntime) {}
}
