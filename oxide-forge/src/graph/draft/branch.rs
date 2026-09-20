use super::{DraftStep, GraphDraft, root::compile_steps};
use crate::cuda::CudaRuntime;
use crate::graph::{Branch, InitConfig, LearningRateScheduler, Loss, MatrixConfig, TrainState};
use crate::net::checkpoint::CheckpointResult;

pub struct BranchDraft {
    pub(crate) input_config: Option<MatrixConfig>,
    pub(crate) output_config: Option<MatrixConfig>,
    pub(crate) steps: Vec<DraftStep>,
}

impl BranchDraft {
    pub fn get_input_config(&self) -> MatrixConfig {
        self.input_config
            .expect("BranchDraft has not been attached to an input")
    }

    pub fn get_output_config(&self) -> MatrixConfig {
        self.output_config
            .expect("BranchDraft has not been attached to an input")
    }

    pub(crate) fn resolve_output(&self, input: MatrixConfig) -> MatrixConfig {
        if let Some(expected) = self.input_config {
            assert_eq!(expected, input, "BranchDraft input shape mismatch");
        }
        self.steps
            .iter()
            .fold(vec![input], |configs, step| step.output_configs(&configs))
            .into_iter()
            .exactly_one()
    }

    pub(crate) fn attach(&mut self, input: MatrixConfig) -> MatrixConfig {
        let output = self.resolve_output(input);
        self.input_config = Some(input);
        self.output_config = Some(output);
        output
    }

    pub fn then(mut self, next: BranchDraft) -> BranchDraft {
        if let (Some(output), Some(input)) = (self.output_config, next.input_config) {
            assert_eq!(output, input, "BranchDraft shape mismatch");
        }
        self.steps.extend(next.steps);
        if let Some(input) = self.input_config {
            self.attach(input);
        }
        self
    }

    pub fn into_graph(self) -> GraphDraft {
        GraphDraft {
            input_config: self.get_input_config(),
            output_config: self.get_output_config(),
            steps: self.steps,
        }
    }

    pub fn init<const TRAINING: bool>(
        self,
        runtime: &mut CudaRuntime,
        config: InitConfig,
    ) -> CheckpointResult<Branch<TRAINING>> {
        let (loss, learning_rate) = match config {
            InitConfig::Random {
                loss,
                learning_rate,
            } => (loss, learning_rate),
            InitConfig::Load(path) => {
                return Err(format!(
                    "Branch checkpoint loading is not implemented for dynamic topology: {}",
                    path.display()
                )
                .into());
            }
        };
        Ok(self.compile::<TRAINING>(runtime, loss, learning_rate))
    }

    pub(crate) fn compile<const TRAINING: bool>(
        self,
        runtime: &mut CudaRuntime,
        loss: Loss,
        learning_rate: LearningRateScheduler,
    ) -> Branch<TRAINING> {
        let input_config = self.get_input_config();
        let output_config = self.get_output_config();
        let (steps, states) = compile_steps::<TRAINING>(self.steps, runtime, loss, &learning_rate);
        Branch {
            input_config,
            output_config,
            loss,
            learning_rate,
            steps,
            training: TRAINING.then(|| TrainState::new(states)),
        }
    }
}

trait ExactlyOne<T> {
    fn exactly_one(self) -> T;
}

impl<T, I: Iterator<Item = T>> ExactlyOne<T> for I {
    fn exactly_one(mut self) -> T {
        let value = self.next().expect("Branch must produce one output");
        assert!(self.next().is_none(), "Branch must produce one output");
        value
    }
}
