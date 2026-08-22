use crate::cuda::{BinaryOp::Add, container::Matrix, runtime::CudaRuntime};
use crate::net::linear::{Linear, LinearMetadata};
use crate::net::metadata::{HostData, MetadataCursor};
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

    pub(super) fn backward_self(
        &mut self,
        input: &Matrix,
        query_gradient: &Matrix,
        key_gradient: &Matrix,
        value_gradient: &Matrix,
        learning_rate: f32,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        let mut input_gradient: Option<Matrix> = None;

        for (linear, projected_gradient) in [
            (&mut self.query, query_gradient),
            (&mut self.key, key_gradient),
            (&mut self.value, value_gradient),
        ] {
            let pre_activation = linear
                .needs_pre_activation()
                .then(|| linear.affine(input, None, runtime, None));
            let (gradient, bias_gradient) =
                linear.backward(pre_activation.as_ref(), projected_gradient, runtime, None);
            let current_input_gradient = linear.input_gradient(&gradient, runtime, None);

            if let Some(total) = &mut input_gradient {
                total.binary_assign(&current_input_gradient, Add, runtime);
            } else {
                input_gradient = Some(current_input_gradient);
            }

            linear.learn(
                input,
                &gradient,
                bias_gradient.as_ref(),
                learning_rate,
                runtime,
                None,
            );
        }

        input_gradient.unwrap()
    }
}

#[derive(Clone, Copy)]
struct Normalization {
    kind: NormType,
}

impl Normalization {
    fn new(kind: NormType) -> Self {
        Self { kind }
    }

    fn forward(self, matrix: &mut Matrix, runtime: &mut CudaRuntime) {
        match self.kind {
            NormType::Layer => matrix.layer_norm(runtime),
            NormType::Rms => matrix.rms_norm(runtime),
        }
    }

    fn backward(
        self,
        input: &Matrix,
        output_gradient: &Matrix,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        match self.kind {
            NormType::Layer => runtime.layer_norm_backward(input, output_gradient),
            NormType::Rms => runtime.rms_norm_backward(input, output_gradient),
        }
    }
}

pub(super) struct Attention {
    qkv: QkvProjection,
    normalization: Normalization,
    training_cache: Option<AttentionCache>,
}

struct AttentionCache {
    input: Matrix,
    query: Matrix,
    key: Matrix,
    value: Matrix,
    probabilities: Matrix,
    pre_norm: Matrix,
}

impl Attention {
    pub(super) fn new(
        query: Linear,
        key: Linear,
        value: Linear,
        streams: Option<Vec<Arc<CudaStream>>>,
        norm_type: NormType,
    ) -> Self {
        Self {
            qkv: QkvProjection::new(query, key, value, streams),
            normalization: Normalization::new(norm_type),
            training_cache: None,
        }
    }

    pub(super) fn norm_type(&self) -> NormType {
        self.normalization.kind
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
        let mut output = runtime.matrix_add(query_input, &attention);
        runtime.recycle_matrix(attention);
        self.normalize(&mut output, runtime);
        output
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
        let attention = runtime.matrix_multiply(&probabilities, &projected.value);
        let pre_norm = runtime.matrix_add(&input, &attention);
        runtime.recycle_matrix(attention);
        let mut output = runtime.clone_matrix(&pre_norm);
        self.normalize(&mut output, runtime);

        self.training_cache = Some(AttentionCache {
            input,
            query: projected.query,
            key: projected.key,
            value: projected.value,
            probabilities,
            pre_norm,
        });
        output
    }

    pub(super) fn backward_self(
        &mut self,
        output_gradient: &Matrix,
        learning_rate: f32,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        let cache = self
            .training_cache
            .as_ref()
            .expect("attention forward must run before backward");
        let residual_gradient =
            self.normalization
                .backward(&cache.pre_norm, output_gradient, runtime);

        let value_t = runtime.matrix_transpose(&cache.value);
        let probabilities_gradient = runtime.matrix_multiply(&residual_gradient, &value_t);
        let probabilities_t = runtime.matrix_transpose(&cache.probabilities);
        let value_gradient = runtime.matrix_multiply(&probabilities_t, &residual_gradient);

        let mut score_gradient =
            runtime.softmax_rows_backward(&cache.probabilities, &probabilities_gradient);
        score_gradient.scale(1.0 / (cache.query.cols() as f32).sqrt(), runtime);

        let query_gradient = runtime.matrix_multiply(&score_gradient, &cache.key);
        let score_gradient_t = runtime.matrix_transpose(&score_gradient);
        let key_gradient = runtime.matrix_multiply(&score_gradient_t, &cache.query);

        let projection_gradient = self.qkv.backward_self(
            &cache.input,
            &query_gradient,
            &key_gradient,
            &value_gradient,
            learning_rate,
            runtime,
        );
        let mut input_gradient = residual_gradient;
        input_gradient.binary_assign(&projection_gradient, Add, runtime);
        input_gradient
    }

    pub(super) fn normalize(&self, matrix: &mut Matrix, runtime: &mut CudaRuntime) {
        self.normalization.forward(matrix, runtime);
    }

    pub(super) fn normalization_backward(
        &self,
        input: &Matrix,
        output_gradient: &Matrix,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        self.normalization.backward(input, output_gradient, runtime)
    }

    fn attention_value_inference(
        &self,
        projected: ProjectedQkv,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        let mut probabilities =
            self.attention_scores_inference(projected.query, projected.key, runtime);
        probabilities.softmax_rows(runtime);
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
        probabilities.softmax_rows(runtime);
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
        let key_t = runtime.matrix_transpose(&key);
        runtime.recycle_matrix(key);
        let mut scores = runtime.matrix_multiply(&query, &key_t);
        runtime.recycle_matrix(query);
        runtime.recycle_matrix(key_t);
        scores.scale(1.0 / (query_width as f32).sqrt(), runtime);
        scores
    }

    fn attention_value_from_probabilities(
        &self,
        probabilities: Matrix,
        value: Matrix,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        self.qkv.wait_for_value(runtime);
        let attention = runtime.matrix_multiply(&probabilities, &value);
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
        let key_t = runtime.matrix_transpose(&projected.key);
        let mut scores = runtime.matrix_multiply(&projected.query, &key_t);
        runtime.recycle_matrix(key_t);
        scores.scale(1.0 / (projected.query.cols() as f32).sqrt(), runtime);
        scores.softmax_rows(runtime);
        scores
    }
}
