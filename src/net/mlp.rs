use crate::cuda::BinaryOp::Add;
use crate::cuda::container::Matrix;
use crate::cuda::runtime::CudaRuntime;
use crate::net::linear::Linear;

pub struct MlpExecutor {
    layers: Vec<Linear>,
    /// A residual from the input of `start` to the output of `end - 1`.
    res_range: Option<(usize, usize)>,
}

impl MlpExecutor {
    pub fn new(layers: Vec<Linear>, res_range: Option<(usize, usize)>) -> Self {
        assert!(!layers.is_empty(), "MLP must contain at least one layer");
        if let Some((start, end)) = res_range {
            assert!(start < end && end <= layers.len(), "invalid residual range");
        }
        Self { layers, res_range }
    }

    pub fn forward(&self, input: &Matrix, runtime: &CudaRuntime) -> Matrix {
        let mut output: Option<Matrix> = None;
        let mut residual_cache: Option<Matrix> = None;

        for (index, layer) in self.layers.iter().enumerate() {
            let layer_input = output.as_ref().unwrap_or(input);
            if self.res_range.is_some_and(|(start, _)| index == start) {
                residual_cache = Some(runtime.matrix_copy(layer_input));
            }
            let residual = if self.res_range.is_some_and(|(_, end)| index + 1 == end) {
                residual_cache.as_ref()
            } else {
                None
            };
            output = Some(layer.forward(layer_input, residual, runtime));
        }

        output.unwrap()
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
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
}

pub struct InferenceMLP {
    executor: MlpExecutor,
}

impl InferenceMLP {
    pub fn new(layers: Vec<Linear>, res_range: Option<(usize, usize)>) -> Self {
        Self {
            executor: MlpExecutor::new(layers, res_range),
        }
    }

    pub fn forward(&self, input: &Matrix, runtime: &CudaRuntime) -> Matrix {
        self.executor.forward(input, runtime)
    }
}

pub struct TrainingMlp {
    /// `layer_inputs[i]` owns the input consumed by layer `i`.
    layer_inputs: Vec<Matrix>,
    executor: MlpExecutor,
}

impl TrainingMlp {
    pub fn new(layers: Vec<Linear>, res_range: Option<(usize, usize)>) -> Self {
        Self {
            layer_inputs: Vec::new(),
            executor: MlpExecutor::new(layers, res_range),
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
}
