use crate::cuda::container::Matrix;
use crate::cuda::runtime::CudaRuntime;
use crate::net::linear::{Linear, LinearMetadata};
use crate::net::metadata::{HostData, MetadataCursor};
use crate::net::mlp::{InferenceMLP, MlpMetadata};
use crate::net::node::{BinaryNode, BinaryOp, SingleNode, SingleType};
use cuda_core::CudaStream;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::{NormType, PositionEncoding, attention::Attention};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformerMetadata {
    pub block_count: usize,
    pub attention_residual: bool,
    pub feed_forward_residual: bool,
    pub normalization: NormType,
    pub query: LinearMetadata,
    pub key: LinearMetadata,
    pub value: LinearMetadata,
    pub feed_forward: MlpMetadata,
    pub output: LinearMetadata,
}

pub(super) struct InferenceBlock {
    attention: Attention,
    position_encoding: PositionEncoding,
    fcs: InferenceMLP,
    output_matrix: Linear,
    feed_forward_residual: BinaryNode,
    feed_forward_normalization: SingleNode,
}

impl InferenceBlock {
    pub(super) fn new(
        query: Linear,
        key: Linear,
        value: Linear,
        position_encoding: PositionEncoding,
        fcs: InferenceMLP,
        output_matrix: Linear,
        qkv_streams: Option<Vec<Arc<CudaStream>>>,
        norm_type: NormType,
    ) -> Self {
        Self {
            attention: Attention::new(query, key, value, qkv_streams, norm_type),
            position_encoding,
            fcs,
            output_matrix,
            feed_forward_residual: BinaryNode::new(BinaryOp::Add),
            feed_forward_normalization: SingleNode::new(match norm_type {
                NormType::Layer => SingleType::LayerNorm,
                NormType::Rms => SingleType::RmsNorm,
            }),
        }
    }

    pub(super) fn get_meta_data(&self, cursor: &mut MetadataCursor) -> TransformerMetadata {
        let qkv = self.attention.get_meta_data(cursor);
        TransformerMetadata {
            block_count: 1,
            attention_residual: true,
            feed_forward_residual: true,
            normalization: self.attention.norm_type(),
            query: qkv.query,
            key: qkv.key,
            value: qkv.value,
            feed_forward: self.fcs.get_meta_data(cursor),
            output: self.output_matrix.get_meta_data(cursor),
        }
    }

    pub(super) fn get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
        let mut data = self.attention.get_data(runtime);
        data.extend(self.fcs.get_data(runtime));
        data.extend(self.output_matrix.get_data(runtime));
        data
    }

    pub(super) fn forward(&mut self, input: &Matrix, runtime: &mut CudaRuntime) -> Matrix {
        let positioned = self.position(input, runtime);
        let x = self.attention.forward(&positioned, &positioned, runtime);
        self.finish_forward(positioned, x, runtime)
    }

    pub(super) fn forward_mask(&mut self, input: &Matrix, runtime: &mut CudaRuntime) -> Matrix {
        let positioned = self.position(input, runtime);
        let x = self
            .attention
            .forward_mask(&positioned, &positioned, runtime);
        self.finish_forward(positioned, x, runtime)
    }

    fn finish_forward(
        &mut self,
        positioned: Matrix,
        x: Matrix,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        runtime.recycle_matrix(positioned);

        let ffn = self.fcs.forward(&x, runtime);
        let output = self
            .feed_forward_residual
            .forward_owned(vec![x, ffn], runtime);
        let output = self.feed_forward_normalization.forward(output, runtime);

        let result = self.output_matrix.forward(&output, None, runtime, None);
        runtime.recycle_matrix(output);
        result
    }

    fn position(&self, input: &Matrix, runtime: &mut CudaRuntime) -> Matrix {
        let positioned = (self.position_encoding)(input, runtime);
        assert_eq!(positioned.rows(), input.rows());
        assert_eq!(positioned.cols(), input.cols());
        positioned
    }
}
