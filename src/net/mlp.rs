use crate::cuda::BinaryOp::Add;
use crate::cuda::container::Matrix;
use crate::cuda::runtime::CudaRuntime;
use crate::net::checkpoint;
use crate::net::linear::Linear;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Loss {
    MeanSquaredError,
}

pub struct MlpExecutor {
    pub(super) layers: Vec<Linear>,
    /// A residual from the input of `start` to the output of `end - 1`.
    pub(super) res_range: Option<(usize, usize)>,
    pub(super) loss: Loss,
}

impl MlpExecutor {
    pub fn new(layers: Vec<Linear>, res_range: Option<(usize, usize)>) -> Self {
        Self::with_loss(layers, res_range, Loss::MeanSquaredError)
    }

    pub fn with_loss(layers: Vec<Linear>, res_range: Option<(usize, usize)>, loss: Loss) -> Self {
        assert!(!layers.is_empty(), "MLP must contain at least one layer");
        if let Some((start, end)) = res_range {
            assert!(start < end && end <= layers.len(), "invalid residual range");
        }
        Self {
            layers,
            res_range,
            loss,
        }
    }

    pub fn forward(&self, input: &Matrix, runtime: &CudaRuntime) -> Matrix {
        let mut output: Option<Matrix> = None;
        let mut residual_cache: Option<Matrix> = None;

        for (index, layer) in self.layers.iter().enumerate() {
            let layer_input = output.as_ref().unwrap_or(input);
            if self.res_range.is_some_and(|(start, _)| index == start) {
                residual_cache = Some(runtime.matrix_copy(layer_input));
            }
            let consumes_residual = self.res_range.is_some_and(|(_, end)| index + 1 == end);
            let residual = if consumes_residual {
                residual_cache.as_ref()
            } else {
                None
            };
            let next = layer.forward(layer_input, residual, runtime);

            if let Some(previous) = output.take() {
                runtime.recycle_matrix(previous);
            }
            if consumes_residual {
                runtime.recycle_matrix(
                    residual_cache
                        .take()
                        .expect("residual cache missing at the end of its range"),
                );
            }
            output = Some(next);
        }

        debug_assert!(residual_cache.is_none());
        output.unwrap()
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    pub fn loss(&self) -> Loss {
        self.loss
    }

    pub fn backward(
        &mut self,
        layer_inputs: &[Matrix],
        output_gradient: &Matrix,
        learning_rate: f32,
        runtime: &CudaRuntime,
    ) -> Matrix {
        assert_eq!(layer_inputs.len(), self.layers.len());

        let mut gradient = runtime.matrix_copy(output_gradient);
        let mut residual_gradient: Option<(usize, Matrix)> = None;
        for index in (0..self.layers.len()).rev() {
            let residual_index = self
                .res_range
                .and_then(|(start, end)| (index + 1 == end).then_some(start));
            let pre_activation = self.layers[index].needs_pre_activation().then(|| {
                let residual = residual_index.map(|source| &layer_inputs[source]);
                self.layers[index].affine(&layer_inputs[index], residual, runtime)
            });
            let (layer_gradient, bias_gradient) =
                self.layers[index].backward(pre_activation.as_ref(), &gradient, runtime);
            let mut input_gradient = self.layers[index].input_gradient(&layer_gradient, runtime);

            if let Some(source) = residual_index {
                residual_gradient = Some((source, runtime.matrix_copy(&layer_gradient)));
            }
            if residual_gradient
                .as_ref()
                .is_some_and(|(source, _)| *source == index)
            {
                let (_, skip_gradient) = residual_gradient.take().unwrap();
                input_gradient.binary_assign(&skip_gradient, Add, runtime);
            }

            self.layers[index].learn(
                &layer_inputs[index],
                &layer_gradient,
                bias_gradient.as_ref(),
                learning_rate,
                runtime,
            );
            gradient = input_gradient;
        }
        assert!(residual_gradient.is_none(), "unresolved residual gradient");
        gradient
    }

    pub fn dump_to_file<P: AsRef<Path>>(
        &self,
        path: P,
        runtime: &CudaRuntime,
    ) -> Result<(), Box<dyn Error>> {
        checkpoint::dump_mlp_file(self, path.as_ref(), runtime)
    }

    pub fn load_from_file<P: AsRef<Path>>(
        path: P,
        runtime: &CudaRuntime,
    ) -> Result<Self, Box<dyn Error>> {
        checkpoint::load_mlp_file(path.as_ref(), runtime)
    }
}

pub struct InferenceMLP {
    pub(super) executor: MlpExecutor,
}

impl InferenceMLP {
    pub fn new(layers: Vec<Linear>, res_range: Option<(usize, usize)>) -> Self {
        Self::with_loss(layers, res_range, Loss::MeanSquaredError)
    }

    pub fn with_loss(layers: Vec<Linear>, res_range: Option<(usize, usize)>, loss: Loss) -> Self {
        Self {
            executor: MlpExecutor::with_loss(layers, res_range, loss),
        }
    }

    pub fn forward(&self, input: &Matrix, runtime: &CudaRuntime) -> Matrix {
        self.executor.forward(input, runtime)
    }

    pub fn dump_to_file<P: AsRef<Path>>(
        &self,
        path: P,
        runtime: &CudaRuntime,
    ) -> Result<(), Box<dyn Error>> {
        self.executor.dump_to_file(path, runtime)
    }

    pub fn load_from_file<P: AsRef<Path>>(
        path: P,
        runtime: &CudaRuntime,
    ) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            executor: MlpExecutor::load_from_file(path, runtime)?,
        })
    }
}

pub struct TrainingMlp {
    /// `layer_inputs[i]` owns the input consumed by layer `i`.
    pub(super) layer_inputs: Vec<Matrix>,
    pub(super) executor: MlpExecutor,
}

impl TrainingMlp {
    pub fn new(layers: Vec<Linear>, res_range: Option<(usize, usize)>) -> Self {
        Self::with_loss(layers, res_range, Loss::MeanSquaredError)
    }

    pub fn with_loss(layers: Vec<Linear>, res_range: Option<(usize, usize)>, loss: Loss) -> Self {
        Self {
            layer_inputs: Vec::new(),
            executor: MlpExecutor::with_loss(layers, res_range, loss),
        }
    }

    pub fn forward(&mut self, input: Matrix, runtime: &CudaRuntime) -> Matrix {
        self.layer_inputs.clear();
        self.layer_inputs.reserve(self.executor.layers.len());
        self.layer_inputs.push(input);

        for index in 0..self.executor.layers.len() {
            let residual_index = self
                .executor
                .res_range
                .and_then(|(start, end)| (index + 1 == end).then_some(start));
            let output = {
                let layer_input = &self.layer_inputs[index];
                let residual = residual_index.map(|source| &self.layer_inputs[source]);
                self.executor.layers[index].forward(layer_input, residual, runtime)
            };
            if index + 1 == self.executor.layers.len() {
                return output;
            }
            self.layer_inputs.push(output);
        }
        unreachable!("MLP always contains at least one layer")
    }

    pub fn backward(
        &mut self,
        output_gradient: &Matrix,
        learning_rate: f32,
        runtime: &CudaRuntime,
    ) -> Matrix {
        self.executor
            .backward(&self.layer_inputs, output_gradient, learning_rate, runtime)
    }

    pub fn input(&self) -> Option<&Matrix> {
        self.layer_inputs.first()
    }

    pub fn dump_to_file<P: AsRef<Path>>(
        &self,
        path: P,
        runtime: &CudaRuntime,
    ) -> Result<(), Box<dyn Error>> {
        self.executor.dump_to_file(path, runtime)
    }

    pub fn load_from_file<P: AsRef<Path>>(
        path: P,
        runtime: &CudaRuntime,
    ) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            layer_inputs: Vec::new(),
            executor: MlpExecutor::load_from_file(path, runtime)?,
        })
    }
}
