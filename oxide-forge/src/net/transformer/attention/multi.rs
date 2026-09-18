use crate::cuda::{container::Matrix, runtime::CudaRuntime};
use crate::net::linear::{Linear, LinearMetadata, LinearMomentum};
use crate::net::metadata::{HostData, MetadataCursor};
use crate::net::node::{
    BinaryNode, BinaryOp, SingleNode, SingleType, TrainingBinaryNode, TrainingSingleNode,
};
use crate::net::transformer::NormType;
use cuda_core::CudaStream;
use std::sync::Arc;

use super::{Qkv, QkvProjector};

impl QkvProjector {
    pub(crate) fn new(
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
            layers: Qkv { query, key, value },
            optimizers: Qkv {
                query: LinearMomentum::default(),
                key: LinearMomentum::default(),
                value: LinearMomentum::default(),
            },
            streams,
        }
    }

    pub(crate) fn get_meta_data(&self, cursor: &mut MetadataCursor) -> Qkv<LinearMetadata> {
        Qkv {
            query: self.layers.query.get_meta_data(cursor),
            key: self.layers.key.get_meta_data(cursor),
            value: self.layers.value.get_meta_data(cursor),
        }
    }

    pub(crate) fn get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
        let mut data = self.layers.query.get_data(runtime);
        data.extend(self.layers.key.get_data(runtime));
        data.extend(self.layers.value.get_data(runtime));
        data
    }

    /// Projects query and key/value inputs independently. Passing the same Matrix
    /// for both is self-attention; passing different matrices is cross-attention.
    pub(crate) fn project(
        &mut self,
        query_input: &Matrix,
        key_value_input: &Matrix,
        runtime: &mut CudaRuntime,
    ) -> Qkv<Matrix> {
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

        Qkv {
            query: self
                .layers
                .query
                .forward(query_input, None, runtime, Some(streams[0].as_ref())),
            key: self
                .layers
                .key
                .forward(key_value_input, None, runtime, Some(streams[1].as_ref())),
            value: self.layers.value.forward(
                key_value_input,
                None,
                runtime,
                Some(streams[2].as_ref()),
            ),
        }
    }

    pub(crate) fn wait_for_query_key(&self, runtime: &CudaRuntime) {
        runtime.join_streams(&self.streams.as_ref().unwrap()[..2]);
    }

    pub(crate) fn wait_for_value(&self, runtime: &CudaRuntime) {
        runtime.join_streams(&self.streams.as_ref().unwrap()[2..]);
    }

    pub(crate) fn backward_self_accumulate(
        &mut self,
        input: &Matrix,
        query_gradient: &Matrix,
        key_gradient: &Matrix,
        value_gradient: &Matrix,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        let mut input_gradients = Vec::with_capacity(3);
        let Qkv { query, key, value } = &mut self.layers;
        let Qkv {
            query: query_optimizer,
            key: key_optimizer,
            value: value_optimizer,
        } = &mut self.optimizers;

        for (linear, optimizer, projected_gradient) in [
            (query, query_optimizer, query_gradient),
            (key, key_optimizer, key_gradient),
            (value, value_optimizer, value_gradient),
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

    pub(crate) fn step(
        &mut self,
        learning_rate: f32,
        momentum: f32,
        batch_len: usize,
        runtime: &mut CudaRuntime,
    ) {
        let Qkv { query, key, value } = &mut self.layers;
        let Qkv {
            query: query_optimizer,
            key: key_optimizer,
            value: value_optimizer,
        } = &mut self.optimizers;
        for (linear, optimizer) in [
            (query, query_optimizer),
            (key, key_optimizer),
            (value, value_optimizer),
        ] {
            optimizer.step(linear, learning_rate, momentum, batch_len, runtime);
        }
    }
}

pub(crate) struct Attention<const HEADS: usize> {
    qkv: QkvProjector,
    norm_type: NormType,
    residual: BinaryNode,
    normalization: SingleNode,
    training_residual: TrainingBinaryNode,
    training_normalization: TrainingSingleNode,
    training_cache: Option<AttentionCache>,
}

struct AttentionCache {
    input: Matrix,
    projected: Qkv<Matrix>,
    probabilities: Matrix,
}

impl<const HEADS: usize> Attention<HEADS> {
    pub(crate) fn new(
        query: Linear,
        key: Linear,
        value: Linear,
        streams: Option<Vec<Arc<CudaStream>>>,
        norm_type: NormType,
    ) -> Self {
        assert!(HEADS > 0, "attention requires at least one head");
        let width = query.output_neurons();
        assert_eq!(
            width % HEADS,
            0,
            "attention width must be divisible by head count"
        );
        assert_eq!(key.output_neurons(), width, "query/key widths must match");
        assert_eq!(
            value.output_neurons(),
            width,
            "query/value widths must match"
        );
        let normalization = match norm_type {
            NormType::Layer => SingleType::LayerNorm,
            NormType::Rms => SingleType::RmsNorm,
        };
        Self {
            qkv: QkvProjector::new(query, key, value, streams),
            norm_type,
            residual: BinaryNode::new(BinaryOp::Add),
            normalization: SingleNode::new(normalization),
            training_residual: TrainingBinaryNode::new(BinaryOp::Add),
            training_normalization: TrainingSingleNode::new(normalization),
            training_cache: None,
        }
    }

    pub(crate) fn norm_type(&self) -> NormType {
        self.norm_type
    }

    pub(crate) fn get_meta_data(&self, cursor: &mut MetadataCursor) -> Qkv<LinearMetadata> {
        self.qkv.get_meta_data(cursor)
    }

    pub(crate) fn get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
        self.qkv.get_data(runtime)
    }

    /// Runs attention without retaining backward state. `query_input` is also
    /// the residual source; `key_value_input` may differ for decoder cross-attention.
    pub(crate) fn forward(
        &mut self,
        query_input: &Matrix,
        key_value_input: &Matrix,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        let projected = self.qkv.project(query_input, key_value_input, runtime);
        let attention = self.attention_value_inference(projected, runtime);
        self.finish_forward(query_input, attention, runtime)
    }

    pub(crate) fn forward_mask(
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
    /// QkvProjector while keeping separate query and key/value input caches.
    pub(crate) fn forward_self_training(
        &mut self,
        input: Matrix,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        let projected = self.qkv.project(&input, &input, runtime);
        let probabilities = self.attention_probabilities(&projected, false, runtime);
        let attention = self.attention_value(&probabilities, &projected.value, runtime);
        let pre_norm = self
            .training_residual
            .forward_borrowed(&[&input, &attention], runtime);
        runtime.recycle_matrix(attention);
        let output = self.training_normalization.forward(pre_norm, runtime);

        self.training_cache = Some(AttentionCache {
            input,
            projected,
            probabilities,
        });
        output
    }

    pub(crate) fn backward_self_accumulate(
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

        let width = cache.projected.query.cols();
        let head_width = width / HEADS;
        let query_rows = cache.projected.query.rows();
        let key_rows = cache.projected.key.rows();

        let value_t = runtime.matrix_transpose(&cache.projected.value, None);
        let mut probabilities_gradient = runtime.new_uninit_matrix(HEADS * query_rows, key_rows);
        runtime.matrix_multiply_batched_strided_into(
            &attention_gradient,
            &value_t,
            &mut probabilities_gradient,
            head_width,
            query_rows,
            key_rows,
            HEADS,
            0,
            width,
            head_width,
            0,
            key_rows,
            head_width * key_rows,
            0,
            key_rows,
            query_rows * key_rows,
        );

        let probabilities_t = runtime.matrix_transpose_batches(
            &cache.probabilities,
            HEADS,
            query_rows,
            key_rows,
            None,
        );
        let mut value_gradient = runtime.new_uninit_matrix(key_rows, width);
        runtime.matrix_multiply_batched_strided_into(
            &probabilities_t,
            &attention_gradient,
            &mut value_gradient,
            query_rows,
            key_rows,
            head_width,
            HEADS,
            0,
            query_rows,
            key_rows * query_rows,
            0,
            width,
            head_width,
            0,
            width,
            head_width,
        );

        let score_gradient =
            runtime.softmax_rows_backward(&cache.probabilities, &probabilities_gradient, None);
        let score_gradient = SingleNode::new(SingleType::Scale(1.0 / (head_width as f32).sqrt()))
            .forward(score_gradient, runtime);

        let mut query_gradient = runtime.new_uninit_matrix(query_rows, width);
        runtime.matrix_multiply_batched_strided_into(
            &score_gradient,
            &cache.projected.key,
            &mut query_gradient,
            key_rows,
            query_rows,
            head_width,
            HEADS,
            0,
            key_rows,
            query_rows * key_rows,
            0,
            width,
            head_width,
            0,
            width,
            head_width,
        );

        let score_gradient_t =
            runtime.matrix_transpose_batches(&score_gradient, HEADS, query_rows, key_rows, None);
        let mut key_gradient = runtime.new_uninit_matrix(key_rows, width);
        runtime.matrix_multiply_batched_strided_into(
            &score_gradient_t,
            &cache.projected.query,
            &mut key_gradient,
            query_rows,
            key_rows,
            head_width,
            HEADS,
            0,
            query_rows,
            key_rows * query_rows,
            0,
            width,
            head_width,
            0,
            width,
            head_width,
        );

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

    pub(crate) fn step(
        &mut self,
        learning_rate: f32,
        momentum: f32,
        batch_len: usize,
        runtime: &mut CudaRuntime,
    ) {
        self.qkv.step(learning_rate, momentum, batch_len, runtime);
    }

    fn attention_value_inference(
        &mut self,
        projected: Qkv<Matrix>,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        let probabilities =
            self.attention_scores_inference(projected.query, projected.key, false, runtime);
        self.attention_value_from_probabilities(probabilities, projected.value, runtime)
    }

    fn attention_value_mask_inference(
        &mut self,
        projected: Qkv<Matrix>,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        let probabilities =
            self.attention_scores_inference(projected.query, projected.key, true, runtime);
        self.attention_value_from_probabilities(probabilities, projected.value, runtime)
    }

    fn attention_scores_inference(
        &mut self,
        query: Matrix,
        key: Matrix,
        masked: bool,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        self.qkv.wait_for_query_key(runtime);
        let query_width = query.cols();
        let head_width = query_width / HEADS;
        let query_rows = query.rows();
        let key_rows = key.rows();
        let key_t = runtime.matrix_transpose(&key, None);
        let mut scores = runtime.new_uninit_matrix(HEADS * query_rows, key_rows);
        runtime.matrix_multiply_batched_strided_into(
            &query,
            &key_t,
            &mut scores,
            head_width,
            query_rows,
            key_rows,
            HEADS,
            0,
            query_width,
            head_width,
            0,
            key_rows,
            head_width * key_rows,
            0,
            key_rows,
            query_rows * key_rows,
        );
        runtime.recycle_matrix(query);
        runtime.recycle_matrix(key);
        runtime.recycle_matrix(key_t);
        let mut scores = SingleNode::new(SingleType::Scale(1.0 / (head_width as f32).sqrt()))
            .forward(scores, runtime);
        if masked {
            scores.causal_mask_heads(query_rows, runtime);
        }
        SingleNode::new(SingleType::Softmax).forward(scores, runtime)
    }

    fn attention_value_from_probabilities(
        &mut self,
        probabilities: Matrix,
        value: Matrix,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        let attention = self.attention_value(&probabilities, &value, runtime);
        runtime.recycle_matrix(probabilities);
        runtime.recycle_matrix(value);
        attention
    }

    fn attention_probabilities(
        &mut self,
        projected: &Qkv<Matrix>,
        masked: bool,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        self.qkv.wait_for_query_key(runtime);
        let width = projected.query.cols();
        let head_width = width / HEADS;
        let query_rows = projected.query.rows();
        let key_rows = projected.key.rows();
        let key_t = runtime.matrix_transpose(&projected.key, None);
        let mut scores = runtime.new_uninit_matrix(HEADS * query_rows, key_rows);
        runtime.matrix_multiply_batched_strided_into(
            &projected.query,
            &key_t,
            &mut scores,
            head_width,
            query_rows,
            key_rows,
            HEADS,
            0,
            width,
            head_width,
            0,
            key_rows,
            head_width * key_rows,
            0,
            key_rows,
            query_rows * key_rows,
        );
        runtime.recycle_matrix(key_t);
        let mut scores = SingleNode::new(SingleType::Scale(1.0 / (head_width as f32).sqrt()))
            .forward(scores, runtime);
        if masked {
            scores.causal_mask_heads(query_rows, runtime);
        }
        SingleNode::new(SingleType::Softmax).forward(scores, runtime)
    }

    fn attention_value(
        &mut self,
        probabilities: &Matrix,
        value: &Matrix,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        self.qkv.wait_for_value(runtime);
        let query_rows = probabilities.rows() / HEADS;
        let key_rows = probabilities.cols();
        let width = value.cols();
        let head_width = width / HEADS;
        let mut attention = runtime.new_uninit_matrix(query_rows, width);
        runtime.matrix_multiply_batched_strided_into(
            probabilities,
            value,
            &mut attention,
            key_rows,
            query_rows,
            head_width,
            HEADS,
            0,
            key_rows,
            query_rows * key_rows,
            0,
            width,
            head_width,
            0,
            width,
            head_width,
        );
        attention
    }
}
