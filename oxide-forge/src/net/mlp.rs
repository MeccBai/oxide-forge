use crate::cuda::BinaryOp::{Add, Mul, Sub};
use crate::cuda::container::Matrix;
use crate::cuda::container::Vector;
use crate::cuda::runtime::CudaRuntime;
use crate::net::linear::{Linear, LinearMetadata, LinearMomentum};
use crate::net::metadata::{HostData, MetadataCursor};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Loss {
    MeanSquaredError,
    /// Numerically stable binary cross-entropy. Model outputs are logits;
    /// sigmoid is applied only when producing probabilities.
    BinaryCrossEntropyWithLogits,
}

impl Loss {
    pub fn activate_output(self, output: &mut Matrix, runtime: &CudaRuntime) {
        if matches!(self, Self::BinaryCrossEntropyWithLogits) {
            output.sigmoid(runtime);
        }
    }

    pub fn loss_rows(
        self,
        output: &Matrix,
        target: &Matrix,
        positive_weight: f32,
        runtime: &mut CudaRuntime,
    ) -> Vector {
        validate_binary_loss_inputs(output, target, positive_weight);
        let mut loss = runtime.clone_matrix(output);
        match self {
            Self::MeanSquaredError => {
                loss.binary_assign(target, Sub, runtime);
                loss.for_each(runtime, |value| 0.5 * value * value);
            }
            Self::BinaryCrossEntropyWithLogits => {
                // softplus(x) - x * target is stable BCE with logits.
                loss.for_each(runtime, |value| {
                    value.max(0.0) + (1.0 + (-value.abs()).exp()).ln()
                });
                let product = runtime.matrix_mul(output, target);
                loss.binary_assign(&product, Sub, runtime);
                runtime.recycle_matrix(product);

                if positive_weight != 1.0 {
                    // Add (positive_weight - 1) * target * softplus(-logit).
                    let mut positive = runtime.clone_matrix(output);
                    positive.for_each(runtime, move |value| {
                        (positive_weight - 1.0)
                            * ((-value).max(0.0) + (1.0 + (-value.abs()).exp()).ln())
                    });
                    positive.binary_assign(target, Mul, runtime);
                    loss.binary_assign(&positive, Add, runtime);
                    runtime.recycle_matrix(positive);
                }
            }
        }

        let cols = loss.cols() as f32;
        let mut rows = runtime.matrix_sum_rows(&loss);
        rows.scale(1.0 / cols, runtime);
        runtime.recycle_matrix(loss);
        rows
    }

    pub fn output_gradient(
        self,
        output: &Matrix,
        target: &Matrix,
        positive_weight: f32,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        validate_binary_loss_inputs(output, target, positive_weight);
        let mut gradient = runtime.clone_matrix(output);
        match self {
            Self::MeanSquaredError => gradient.binary_assign(target, Sub, runtime),
            Self::BinaryCrossEntropyWithLogits => {
                gradient.sigmoid(runtime);
                if positive_weight == 1.0 {
                    gradient.binary_assign(target, Sub, runtime);
                } else {
                    // sigmoid(logit) * (1 - target + weight * target)
                    //     - weight * target
                    // remains correct for both hard and soft target values.
                    let mut weights = runtime.clone_matrix(target);
                    weights.scale(positive_weight - 1.0, runtime);
                    weights.add_scalar(1.0, runtime);
                    gradient.binary_assign(&weights, Mul, runtime);
                    runtime.recycle_matrix(weights);

                    let mut weighted_target = runtime.clone_matrix(target);
                    weighted_target.scale(positive_weight, runtime);
                    gradient.binary_assign(&weighted_target, Sub, runtime);
                    runtime.recycle_matrix(weighted_target);
                }
            }
        }

        gradient.scale(1.0 / (output.rows() * output.cols()) as f32, runtime);
        gradient
    }

    pub fn loss_and_output_gradient(
        self,
        output: &Matrix,
        target: &Matrix,
        positive_weight: f32,
        dice_weight: f32,
        runtime: &mut CudaRuntime,
    ) -> (Vector, Matrix) {
        assert!(
            dice_weight.is_finite() && dice_weight >= 0.0,
            "Dice weight must be finite and non-negative"
        );
        let mut loss_rows = self.loss_rows(output, target, positive_weight, runtime);
        let mut gradient = self.output_gradient(output, target, positive_weight, runtime);
        if dice_weight == 0.0 {
            return (loss_rows, gradient);
        }
        assert!(
            matches!(self, Self::BinaryCrossEntropyWithLogits),
            "Dice loss requires binary cross-entropy with logits"
        );

        let mut probabilities = runtime.clone_matrix(output);
        probabilities.sigmoid(runtime);
        let intersection = probabilities.zip_map_reduce(
            target,
            runtime,
            0.0,
            move |probability, target| probability * target,
            move |lhs, rhs| lhs + rhs,
        );
        let numerator = 2.0 * intersection + 1.0;
        let denominator = matrix_sum(&probabilities, runtime) + matrix_sum(target, runtime) + 1.0;

        let dice_loss = 1.0 - numerator / denominator;
        loss_rows.add_scalar(dice_weight * dice_loss, runtime);

        // d(1 - Dice)/dp = (numerator - 2 * target * denominator) / denominator²
        let mut dice_gradient = runtime.clone_matrix(target);
        dice_gradient.scale(-2.0 * denominator, runtime);
        dice_gradient.add_scalar(numerator, runtime);
        dice_gradient.scale(dice_weight / (denominator * denominator), runtime);

        // Convert dL/dp to dL/dlogit with sigmoid'(logit) = p * (1 - p).
        let mut sigmoid_derivative = runtime.clone_matrix(&probabilities);
        sigmoid_derivative.scale(-1.0, runtime);
        sigmoid_derivative.add_scalar(1.0, runtime);
        sigmoid_derivative.binary_assign(&probabilities, Mul, runtime);
        dice_gradient.binary_assign(&sigmoid_derivative, Mul, runtime);
        gradient.binary_assign(&dice_gradient, Add, runtime);

        runtime.recycle_matrix(probabilities);
        runtime.recycle_matrix(sigmoid_derivative);
        runtime.recycle_matrix(dice_gradient);
        (loss_rows, gradient)
    }
}

fn matrix_sum(matrix: &Matrix, runtime: &mut CudaRuntime) -> f32 {
    matrix.sum(runtime)
}

fn validate_binary_loss_inputs(output: &Matrix, target: &Matrix, positive_weight: f32) {
    assert_eq!(output.rows(), target.rows());
    assert_eq!(output.cols(), target.cols());
    assert!(
        positive_weight.is_finite() && positive_weight > 0.0,
        "positive weight must be finite and greater than zero"
    );
}

pub struct MlpExecutor {
    layers: Vec<Linear>,
    /// A residual from the input of `start` to the output of `end - 1`.
    res_range: Option<(usize, usize)>,
    loss: Loss,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlpMetadata {
    pub layer_count: usize,
    pub loss: Loss,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub residual: Option<ResidualMetadata>,
    pub layers: Vec<LinearMetadata>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidualMetadata {
    pub start: usize,
    pub end: usize,
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

    pub fn forward(&self, input: &Matrix, runtime: &mut CudaRuntime) -> Matrix {
        let mut output: Option<Matrix> = None;
        let mut residual_cache: Option<Matrix> = None;

        for (index, layer) in self.layers.iter().enumerate() {
            let layer_input = output.as_ref().unwrap_or(input);
            if self.res_range.is_some_and(|(start, _)| index == start) {
                residual_cache = Some(runtime.clone_matrix(layer_input));
            }
            let consumes_residual = self.res_range.is_some_and(|(_, end)| index + 1 == end);
            let residual = if consumes_residual {
                residual_cache.as_ref()
            } else {
                None
            };
            let next = layer.forward(layer_input, residual, runtime, None);

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

    pub fn activate_output(&self, output: &mut Matrix, runtime: &CudaRuntime) {
        self.loss.activate_output(output, runtime);
    }

    pub fn loss_rows(
        &self,
        output: &Matrix,
        target: &Matrix,
        positive_weight: f32,
        runtime: &mut CudaRuntime,
    ) -> Vector {
        self.loss
            .loss_rows(output, target, positive_weight, runtime)
    }

    pub fn output_gradient(
        &self,
        output: &Matrix,
        target: &Matrix,
        positive_weight: f32,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        self.loss
            .output_gradient(output, target, positive_weight, runtime)
    }

    pub fn loss_and_output_gradient(
        &self,
        output: &Matrix,
        target: &Matrix,
        positive_weight: f32,
        dice_weight: f32,
        runtime: &mut CudaRuntime,
    ) -> (Vector, Matrix) {
        self.loss
            .loss_and_output_gradient(output, target, positive_weight, dice_weight, runtime)
    }

    pub fn get_meta_data(&self, cursor: &mut MetadataCursor) -> MlpMetadata {
        MlpMetadata {
            layer_count: self.layers.len(),
            loss: self.loss,
            residual: self
                .res_range
                .map(|(start, end)| ResidualMetadata { start, end }),
            layers: self
                .layers
                .iter()
                .map(|layer| layer.get_meta_data(cursor))
                .collect(),
        }
    }

    pub fn get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
        self.layers
            .iter()
            .flat_map(|layer| layer.get_data(runtime))
            .collect()
    }

    fn backward_accumulate(
        &mut self,
        layer_inputs: &[Matrix],
        output_gradient: &Matrix,
        optimizers: &mut [LinearMomentum],
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        assert_eq!(layer_inputs.len(), self.layers.len());
        assert_eq!(optimizers.len(), self.layers.len());

        let mut gradient = runtime.clone_matrix(output_gradient);
        let mut residual_gradient: Option<(usize, Matrix)> = None;
        for index in (0..self.layers.len()).rev() {
            let residual_index = self
                .res_range
                .and_then(|(start, end)| (index + 1 == end).then_some(start));
            let pre_activation = self.layers[index].needs_pre_activation().then(|| {
                let residual = residual_index.map(|source| &layer_inputs[source]);
                self.layers[index].affine(&layer_inputs[index], residual, runtime, None)
            });
            let (layer_gradient, bias_gradient) =
                self.layers[index].backward(pre_activation.as_ref(), &gradient, runtime, None);
            let mut input_gradient =
                self.layers[index].input_gradient(&layer_gradient, runtime, None);

            if let Some(source) = residual_index {
                residual_gradient = Some((source, runtime.clone_matrix(&layer_gradient)));
            }
            if residual_gradient
                .as_ref()
                .is_some_and(|(source, _)| *source == index)
            {
                let (_, skip_gradient) = residual_gradient.take().unwrap();
                input_gradient.binary_assign(&skip_gradient, Add, runtime);
            }

            optimizers[index].accumulate(
                &layer_inputs[index],
                &layer_gradient,
                bias_gradient.as_ref(),
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
        Self::with_loss(layers, res_range, Loss::MeanSquaredError)
    }

    pub fn with_loss(layers: Vec<Linear>, res_range: Option<(usize, usize)>, loss: Loss) -> Self {
        Self {
            executor: MlpExecutor::with_loss(layers, res_range, loss),
        }
    }

    pub fn get_meta_data(&self, cursor: &mut MetadataCursor) -> MlpMetadata {
        self.executor.get_meta_data(cursor)
    }

    pub fn get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
        self.executor.get_data(runtime)
    }

    pub fn loss(&self) -> Loss {
        self.executor.loss()
    }

    pub fn forward(&self, input: &Matrix, runtime: &mut CudaRuntime) -> Matrix {
        self.executor.forward(input, runtime)
    }

    pub fn predict(&self, input: &Matrix, runtime: &mut CudaRuntime) -> Matrix {
        let mut output = self.forward(input, runtime);
        self.executor.activate_output(&mut output, runtime);
        output
    }
}

pub struct TrainingMlp {
    /// `layer_inputs[i]` owns the input consumed by layer `i`.
    layer_inputs: Vec<Matrix>,
    executor: MlpExecutor,
    optimizers: Vec<LinearMomentum>,
}

impl TrainingMlp {
    pub fn new(layers: Vec<Linear>, res_range: Option<(usize, usize)>) -> Self {
        Self::with_loss(layers, res_range, Loss::MeanSquaredError)
    }

    pub fn with_loss(layers: Vec<Linear>, res_range: Option<(usize, usize)>, loss: Loss) -> Self {
        let layer_count = layers.len();
        Self {
            layer_inputs: Vec::new(),
            executor: MlpExecutor::with_loss(layers, res_range, loss),
            optimizers: (0..layer_count)
                .map(|_| LinearMomentum::default())
                .collect(),
        }
    }

    pub fn get_meta_data(&self, cursor: &mut MetadataCursor) -> MlpMetadata {
        self.executor.get_meta_data(cursor)
    }

    pub fn get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
        self.executor.get_data(runtime)
    }

    pub fn loss(&self) -> Loss {
        self.executor.loss()
    }

    pub fn activate_output(&self, output: &mut Matrix, runtime: &CudaRuntime) {
        self.executor.activate_output(output, runtime);
    }

    pub fn loss_rows(
        &self,
        output: &Matrix,
        target: &Matrix,
        positive_weight: f32,
        runtime: &mut CudaRuntime,
    ) -> Vector {
        self.executor
            .loss_rows(output, target, positive_weight, runtime)
    }

    pub fn output_gradient(
        &self,
        output: &Matrix,
        target: &Matrix,
        positive_weight: f32,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        self.executor
            .output_gradient(output, target, positive_weight, runtime)
    }

    pub fn loss_and_output_gradient(
        &self,
        output: &Matrix,
        target: &Matrix,
        positive_weight: f32,
        dice_weight: f32,
        runtime: &mut CudaRuntime,
    ) -> (Vector, Matrix) {
        self.executor.loss_and_output_gradient(
            output,
            target,
            positive_weight,
            dice_weight,
            runtime,
        )
    }

    pub fn forward(&mut self, input: Matrix, runtime: &mut CudaRuntime) -> Matrix {
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
                self.executor.layers[index].forward(layer_input, residual, runtime, None)
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
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        let input_gradient = self.backward_accumulate(output_gradient, runtime);
        self.step(learning_rate, 0.0, 1, runtime);
        input_gradient
    }

    /// Backpropagates one sample and adds its parameter gradients to the current
    /// mini-batch without changing any parameters.
    pub fn backward_accumulate(
        &mut self,
        output_gradient: &Matrix,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        self.executor.backward_accumulate(
            &self.layer_inputs,
            output_gradient,
            &mut self.optimizers,
            runtime,
        )
    }

    /// Applies the mean gradient of the accumulated mini-batch using classical
    /// momentum SGD, then clears only the gradient accumulators.
    pub fn step(
        &mut self,
        learning_rate: f32,
        momentum: f32,
        batch_len: usize,
        runtime: &mut CudaRuntime,
    ) {
        for (layer, optimizer) in self.executor.layers.iter_mut().zip(&mut self.optimizers) {
            optimizer.step(layer, learning_rate, momentum, batch_len, runtime);
        }
    }

    pub fn input(&self) -> Option<&Matrix> {
        self.layer_inputs.first()
    }
}
