use crate::cuda::{container::Matrix, runtime::CudaRuntime};
use crate::net::linear::{Linear, LinearMetadata, LinearMomentum};
use crate::net::metadata::{HostData, MetadataCursor};
use crate::net::node::{
    BinaryNode, BinaryOp, SingleNode, SingleType, TrainingBinaryNode, TrainingSingleNode,
};
use cuda_core::CudaStream;
use std::sync::Arc;

use super::NormType;

pub(super) struct QkvMetadata {
    pub query: LinearMetadata,
    pub key: LinearMetadata,
    pub value: LinearMetadata,
}

pub(super) struct QkvProjection {
    query: Linear,
    key: Linear,
    value: Linear,
    optimizers: [LinearMomentum; 3],
    streams: Option<[Arc<CudaStream>; 3]>,
}

pub(super) struct ProjectedQkv {
    pub query: Matrix,
    pub key: Matrix,
    pub value: Matrix,
}

impl QkvProjection {
    pub(super) fn new(
        query: Linear,
        key: Linear,
        value: Linear,
        streams: Option<Vec<Arc<CudaStream>>>,
    ) -> Self {
        let streams = streams.map(|streams| {
            streams
                .try_into()
                .unwrap_or_else(|_| panic!("Q/K/V execution requires three streams"))
        });
        Self {
            query,
            key,
            value,
            optimizers: std::array::from_fn(|_| LinearMomentum::default()),
            streams,
        }
    }

    pub(super) fn get_meta_data(&self, cursor: &mut MetadataCursor) -> QkvMetadata {
        QkvMetadata {
            query: self.query.get_meta_data(cursor),
            key: self.key.get_meta_data(cursor),
            value: self.value.get_meta_data(cursor),
        }
    }

    pub(super) fn get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
        let mut data = self.query.get_data(runtime);
        data.extend(self.key.get_data(runtime));
        data.extend(self.value.get_data(runtime));
        data
    }

    /// Projects query and key/value inputs independently. Passing the same Matrix
    /// for both is self-attention; passing different matrices is cross-attention.
    pub(super) fn project(
        &mut self,
        query_input: &Matrix,
        key_value_input: &Matrix,
        runtime: &mut CudaRuntime,
    ) -> ProjectedQkv {
        if let Some(streams) = &self.streams {
            runtime.fork_streams(streams);
        } else {
            self.streams = Some(
                runtime
                    .create_extra_streams(3)
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("three streams were requested")),
            );
        }
        let streams = self.streams.as_ref().unwrap();

        ProjectedQkv {
            query: self
                .query
                .forward(query_input, None, runtime, Some(streams[0].as_ref())),
            key: self
                .key
                .forward(key_value_input, None, runtime, Some(streams[1].as_ref())),
            value: self
                .value
                .forward(key_value_input, None, runtime, Some(streams[2].as_ref())),
        }
    }

    pub(super) fn wait_for_query_key(&self, runtime: &CudaRuntime) {
        runtime.join_streams(&self.streams.as_ref().unwrap()[..2]);
    }

    pub(super) fn wait_for_value(&self, runtime: &CudaRuntime) {
        runtime.join_streams(&self.streams.as_ref().unwrap()[2..]);
    }

    pub(super) fn backward_self_accumulate(
        &mut self,
        input: &Matrix,
        query_gradient: &Matrix,
        key_gradient: &Matrix,
        value_gradient: &Matrix,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        let mut input_gradients = Vec::with_capacity(3);
        let [query_optimizer, key_optimizer, value_optimizer] = &mut self.optimizers;

        for (linear, optimizer, projected_gradient) in [
            (&mut self.query, query_optimizer, query_gradient),
            (&mut self.key, key_optimizer, key_gradient),
            (&mut self.value, value_optimizer, value_gradient),
        ] {
            let pre_activation = linear
                .needs_pre_activation()
                .then(|| linear.affine(input, None, runtime, None));
            let (gradient, bias_gradient) =
                linear.backward(pre_activation.as_ref(), projected_gradient, runtime, None);
            let current_input_gradient = linear.input_gradient(&gradient, runtime, None);

            input_gradients.push(current_input_gradient);

            optimizer.accumulate(input, &gradient, bias_gradient.as_ref(), runtime);
        }

        BinaryNode::new(BinaryOp::Add).forward_owned(input_gradients, runtime)
    }

    pub(super) fn step(
        &mut self,
        learning_rate: f32,
        momentum: f32,
        batch_len: usize,
        runtime: &mut CudaRuntime,
    ) {
        let [query_optimizer, key_optimizer, value_optimizer] = &mut self.optimizers;
        for (linear, optimizer) in [
            (&mut self.query, query_optimizer),
            (&mut self.key, key_optimizer),
            (&mut self.value, value_optimizer),
        ] {
            optimizer.step(linear, learning_rate, momentum, batch_len, runtime);
        }
    }
}

pub(super) struct Attention {
    qkv: QkvProjection,
    norm_type: NormType,
    residual: BinaryNode,
    normalization: SingleNode,
    training_residual: TrainingBinaryNode,
    training_normalization: TrainingSingleNode,
    training_cache: Option<AttentionCache>,
}

struct AttentionCache {
    input: Matrix,
    query: Matrix,
    key: Matrix,
    value: Matrix,
    probabilities: Matrix,
}

impl Attention {
    pub(super) fn new(
        query: Linear,
        key: Linear,
        value: Linear,
        streams: Option<Vec<Arc<CudaStream>>>,
        norm_type: NormType,
    ) -> Self {
        let normalization = match norm_type {
            NormType::Layer => SingleType::LayerNorm,
            NormType::Rms => SingleType::RmsNorm,
        };
        Self {
            qkv: QkvProjection::new(query, key, value, streams),
            norm_type,
            residual: BinaryNode::new(BinaryOp::Add),
            normalization: SingleNode::new(normalization),
            training_residual: TrainingBinaryNode::new(BinaryOp::Add),
            training_normalization: TrainingSingleNode::new(normalization),
            training_cache: None,
        }
    }

    pub(super) fn norm_type(&self) -> NormType {
        self.norm_type
    }

    pub(super) fn get_meta_data(&self, cursor: &mut MetadataCursor) -> QkvMetadata {
        self.qkv.get_meta_data(cursor)
    }

    pub(super) fn get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
        self.qkv.get_data(runtime)
    }

    /// Runs attention without retaining backward state. `query_input` is also
    /// the residual source; `key_value_input` may differ for decoder cross-attention.
    pub(super) fn forward(
        &mut self,
        query_input: &Matrix,
        key_value_input: &Matrix,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        let projected = self.qkv.project(query_input, key_value_input, runtime);
        let attention = self.attention_value_inference(projected, runtime);
        self.finish_forward(query_input, attention, runtime)
    }

    pub(super) fn forward_mask(
        &mut self,
        query_input: &Matrix,
        key_value_input: &Matrix,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        let projected = self.qkv.project(query_input, key_value_input, runtime);
        let attention = self.attention_value_mask_inference(projected, runtime);
        self.finish_forward(query_input, attention, runtime)
    }

    fn finish_forward(
        &self,
        query_input: &Matrix,
        attention: Matrix,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        let output = self.residual.forward(&[query_input, &attention], runtime);
        runtime.recycle_matrix(attention);
        self.normalization.forward(output, runtime)
    }

    /// Self-attention variant that owns the input and retains exactly the state
    /// required by backward. A future cross-attention trainer can use the same
    /// QkvProjection while keeping separate query and key/value input caches.
    pub(super) fn forward_self_training(
        &mut self,
        input: Matrix,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        let projected = self.qkv.project(&input, &input, runtime);
        let probabilities = self.attention_probabilities(&projected, runtime);
        self.qkv.wait_for_value(runtime);
        let attention = runtime.matrix_multiply(&probabilities, &projected.value, None);
        let pre_norm = self
            .training_residual
            .forward_borrowed(&[&input, &attention], runtime);
        runtime.recycle_matrix(attention);
        let output = self.training_normalization.forward(pre_norm, runtime);

        self.training_cache = Some(AttentionCache {
            input,
            query: projected.query,
            key: projected.key,
            value: projected.value,
            probabilities,
        });
        output
    }

    pub(super) fn backward_self_accumulate(
        &mut self,
        output_gradient: Matrix,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        let cache = self
            .training_cache
            .take()
            .expect("attention forward must run before backward");
        let residual_gradient = self
            .training_normalization
            .backward(output_gradient, runtime);
        let [direct_gradient, attention_gradient]: [Matrix; 2] = self
            .training_residual
            .backward(residual_gradient, runtime)
            .try_into()
            .unwrap_or_else(|_| unreachable!("residual Add has two inputs"));

        let value_t = runtime.matrix_transpose(&cache.value, None);
        let probabilities_gradient = runtime.matrix_multiply(&attention_gradient, &value_t, None);
        let probabilities_t = runtime.matrix_transpose(&cache.probabilities, None);
        let value_gradient = runtime.matrix_multiply(&probabilities_t, &attention_gradient, None);

        let score_gradient =
            runtime.softmax_rows_backward(&cache.probabilities, &probabilities_gradient, None);
        let score_gradient =
            SingleNode::new(SingleType::Scale(1.0 / (cache.query.cols() as f32).sqrt()))
                .forward(score_gradient, runtime);

        let query_gradient = runtime.matrix_multiply(&score_gradient, &cache.key, None);
        let score_gradient_t = runtime.matrix_transpose(&score_gradient, None);
        let key_gradient = runtime.matrix_multiply(&score_gradient_t, &cache.query, None);

        let projection_gradient = self.qkv.backward_self_accumulate(
            &cache.input,
            &query_gradient,
            &key_gradient,
            &value_gradient,
            runtime,
        );
        runtime.recycle_matrix(attention_gradient);
        BinaryNode::new(BinaryOp::Add)
            .forward_owned(vec![direct_gradient, projection_gradient], runtime)
    }

    pub(super) fn step(
        &mut self,
        learning_rate: f32,
        momentum: f32,
        batch_len: usize,
        runtime: &mut CudaRuntime,
    ) {
        self.qkv.step(learning_rate, momentum, batch_len, runtime);
    }

    fn attention_value_inference(
        &self,
        projected: ProjectedQkv,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        let probabilities =
            self.attention_scores_inference(projected.query, projected.key, runtime);
        let probabilities = SingleNode::new(SingleType::Softmax).forward(probabilities, runtime);
        self.attention_value_from_probabilities(probabilities, projected.value, runtime)
    }

    fn attention_value_mask_inference(
        &self,
        projected: ProjectedQkv,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        let mut probabilities =
            self.attention_scores_inference(projected.query, projected.key, runtime);
        probabilities.causal_mask(runtime);
        let probabilities = SingleNode::new(SingleType::Softmax).forward(probabilities, runtime);
        self.attention_value_from_probabilities(probabilities, projected.value, runtime)
    }

    fn attention_scores_inference(
        &self,
        query: Matrix,
        key: Matrix,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        self.qkv.wait_for_query_key(runtime);
        let query_width = query.cols();
        let key_t = runtime.matrix_transpose(&key, None);
        runtime.recycle_matrix(key);
        let scores = runtime.matrix_multiply(&query, &key_t, None);
        runtime.recycle_matrix(query);
        runtime.recycle_matrix(key_t);
        SingleNode::new(SingleType::Scale(1.0 / (query_width as f32).sqrt()))
            .forward(scores, runtime)
    }

    fn attention_value_from_probabilities(
        &self,
        probabilities: Matrix,
        value: Matrix,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        self.qkv.wait_for_value(runtime);
        let attention = runtime.matrix_multiply(&probabilities, &value, None);
        runtime.recycle_matrix(probabilities);
        runtime.recycle_matrix(value);
        attention
    }

    fn attention_probabilities(
        &self,
        projected: &ProjectedQkv,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        self.qkv.wait_for_query_key(runtime);
        let key_t = runtime.matrix_transpose(&projected.key, None);
        let scores = runtime.matrix_multiply(&projected.query, &key_t, None);
        runtime.recycle_matrix(key_t);
        let scores = SingleNode::new(SingleType::Scale(
            1.0 / (projected.query.cols() as f32).sqrt(),
        ))
        .forward(scores, runtime);
        SingleNode::new(SingleType::Softmax).forward(scores, runtime)
    }
}
