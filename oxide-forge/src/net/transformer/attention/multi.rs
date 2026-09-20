use crate::cuda::{container::Matrix, runtime::CudaRuntime};
use crate::net::linear::{Linear, LinearMetadata};
use crate::net::metadata::{HostData, HostDataCursor, MetadataCursor};
use crate::net::node::{SingleNode, SingleType};
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

    pub(crate) fn set_data(&mut self, data: &mut HostDataCursor, runtime: &CudaRuntime) {
        self.layers.query.set_data(data, runtime);
        self.layers.key.set_data(data, runtime);
        self.layers.value.set_data(data, runtime);
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
}

pub(crate) struct Attention<const HEADS: usize> {
    qkv: QkvProjector,
}

impl<const HEADS: usize> Attention<HEADS> {
    pub(crate) fn new(
        query: Linear,
        key: Linear,
        value: Linear,
        streams: Option<Vec<Arc<CudaStream>>>,
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
        Self {
            qkv: QkvProjector::new(query, key, value, streams),
        }
    }

    pub(crate) fn get_meta_data(&self, cursor: &mut MetadataCursor) -> Qkv<LinearMetadata> {
        self.qkv.get_meta_data(cursor)
    }

    pub(crate) fn get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
        self.qkv.get_data(runtime)
    }

    pub(crate) fn set_data(&mut self, data: &mut HostDataCursor, runtime: &CudaRuntime) {
        self.qkv.set_data(data, runtime);
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
        self.attention_value_inference(projected, runtime)
    }

    pub(crate) fn forward_mask(
        &mut self,
        query_input: &Matrix,
        key_value_input: &Matrix,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        let projected = self.qkv.project(query_input, key_value_input, runtime);
        self.attention_value_mask_inference(projected, runtime)
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
            query.batch_span(0, query_rows, head_width, query_width, head_width, HEADS),
            key_t.batch_span(
                0,
                head_width,
                key_rows,
                key_rows,
                head_width * key_rows,
                HEADS,
            ),
            scores.batch_span_mut(
                0,
                query_rows,
                key_rows,
                key_rows,
                query_rows * key_rows,
                HEADS,
            ),
            None,
        );
        runtime.recycle_matrix(query);
        runtime.recycle_matrix(key);
        runtime.recycle_matrix(key_t);
        let mut scores = SingleNode::new(SingleType::Scale(1.0 / (head_width as f32).sqrt()))
            .forward(scores, runtime);
        if masked {
            scores.causal_mask_heads(query_rows, runtime, None);
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
            probabilities.batch_span(
                0,
                query_rows,
                key_rows,
                key_rows,
                query_rows * key_rows,
                HEADS,
            ),
            value.batch_span(0, key_rows, head_width, width, head_width, HEADS),
            attention.batch_span_mut(0, query_rows, head_width, width, head_width, HEADS),
            None,
        );
        attention
    }
}
