use crate::cuda::{container::Matrix, runtime::CudaRuntime};
use crate::graph::{GraphNode, LearnConfig};
use crate::net::linear::Linear;
use crate::net::linear::LinearMetadata;
use crate::net::metadata::{HostData, HostDataCursor, MetadataCursor};
use crate::net::mlp::Mlp;
use crate::net::mlp::MlpMetadata;
use crate::net::node::{BinaryNode, BinaryOp, SingleNode, SingleType};
use cuda_core::CudaStream;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::{NormType, PositionEncoding, PositionEncodingMetadata, attention::multi::Attention};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformerMetadata {
    pub block_count: usize,
    pub attention_heads: usize,
    pub attention_residual: bool,
    pub feed_forward_residual: bool,
    pub normalization: NormType,
    pub position_encoding: PositionEncodingMetadata,
    pub query: LinearMetadata,
    pub key: LinearMetadata,
    pub value: LinearMetadata,
    pub feed_forward: MlpMetadata,
    pub output: LinearMetadata,
}

pub struct Transformer<const HEADS: usize = 1, const MASKED: bool = false> {
    attention: Attention<HEADS>,
    norm_type: NormType,
    attention_residual: BinaryNode,
    attention_normalization: SingleNode,
    position_encoding: PositionEncoding,
    fcs: Mlp,
    output_matrix: Linear,
    feed_forward_residual: BinaryNode,
    feed_forward_normalization: SingleNode,
}

impl<const HEADS: usize, const MASKED: bool> Transformer<HEADS, MASKED> {
    pub fn new(
        q_matrix: Linear,
        k_matrix: Linear,
        v_matrix: Linear,
        position_encoding: PositionEncoding,
        fcs: Mlp,
        output_matrix: Linear,
        qkv_streams: Option<Vec<Arc<CudaStream>>>,
        norm_type: NormType,
    ) -> Self {
        Self {
            attention: Attention::new(q_matrix, k_matrix, v_matrix, qkv_streams),
            norm_type,
            attention_residual: BinaryNode::new(BinaryOp::Add),
            attention_normalization: SingleNode::new(match norm_type {
                NormType::Layer => SingleType::LayerNorm,
                NormType::Rms => SingleType::RmsNorm,
            }),
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

    pub fn get_meta_data(&self, cursor: &mut MetadataCursor) -> TransformerMetadata {
        let qkv = self.attention.get_meta_data(cursor);
        TransformerMetadata {
            block_count: 1,
            attention_heads: HEADS,
            attention_residual: true,
            feed_forward_residual: true,
            normalization: self.norm_type,
            position_encoding: self.position_encoding.get_meta_data(cursor),
            query: qkv.query,
            key: qkv.key,
            value: qkv.value,
            feed_forward: self.fcs.get_meta_data(cursor),
            output: self.output_matrix.get_meta_data(cursor),
        }
    }

    pub fn get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
        let mut data = self.attention.get_data(runtime);
        data.extend(self.position_encoding.get_data(runtime));
        data.extend(self.fcs.get_data(runtime));
        data.extend(self.output_matrix.get_data(runtime));
        data
    }

    pub fn set_data(&mut self, data: &mut HostDataCursor, runtime: &CudaRuntime) {
        self.attention.set_data(data, runtime);
        self.position_encoding.set_data(data, runtime);
        self.fcs.set_data(data, runtime);
        self.output_matrix.set_data(data, runtime);
    }

    pub fn forward(&mut self, input: &Matrix, runtime: &mut CudaRuntime) -> Matrix {
        let positioned = self.position_encoding.forward(input, runtime);
        assert_eq!(positioned.rows(), input.rows());
        assert_eq!(positioned.cols(), input.cols());
        let attention = if MASKED {
            self.attention
                .forward_mask(&positioned, &positioned, runtime)
        } else {
            self.attention.forward(&positioned, &positioned, runtime)
        };
        let x = self
            .attention_residual
            .forward(&[&positioned, &attention], runtime);
        runtime.recycle_matrix(attention);
        let x = self.attention_normalization.forward(x, runtime);
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
}

impl<const HEADS: usize, const MASKED: bool> GraphNode for Transformer<HEADS, MASKED> {
    fn forward(&mut self, mut inputs: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        assert_eq!(inputs.len(), 1, "Transformer forward expects one matrix");
        let input = inputs.pop().unwrap();
        let output = Transformer::forward(self, &input, runtime);
        runtime.recycle_matrix(input);
        vec![output]
    }

    fn backward(&mut self, _gradients: Vec<Matrix>, _runtime: &mut CudaRuntime) -> Vec<Matrix> {
        panic!("Transformer training is assembled and owned by Graph<true>")
    }

    fn learn(&mut self, _config: LearnConfig, _runtime: &mut CudaRuntime) {
        panic!("Transformer training is assembled and owned by Graph<true>")
    }

    fn clear_cache(&mut self, _runtime: &mut CudaRuntime) {}

    fn get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
        Transformer::get_data(self, runtime)
    }

    fn set_data(&mut self, data: &mut HostDataCursor, runtime: &CudaRuntime) {
        Transformer::set_data(self, data, runtime);
    }
}
