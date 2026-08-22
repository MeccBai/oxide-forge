use crate::cuda::container::Matrix;
use crate::cuda::runtime::CudaRuntime;
use crate::net::linear::{Linear, LinearMetadata};
use crate::net::metadata::{HostData, MatrixMetadata, MetadataCursor};
use crate::net::mlp::{InferenceMLP, MlpMetadata};
use cuda_core::CudaStream;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::{NormType, attention::Attention};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformerMetadata {
    pub block_count: usize,
    pub attention_residual: bool,
    pub feed_forward_residual: bool,
    pub normalization: NormType,
    pub position: MatrixMetadata,
    pub query: LinearMetadata,
    pub key: LinearMetadata,
    pub value: LinearMetadata,
    pub feed_forward: MlpMetadata,
    pub output: LinearMetadata,
}

pub(super) struct InferenceBlock {
    attention: Attention,
    position_matrix: Matrix,
    fcs: InferenceMLP,
    output_matrix: Linear,
}

impl InferenceBlock {
    pub(super) fn new(
        query: Linear,
        key: Linear,
        value: Linear,
        position_matrix: Matrix,
        fcs: InferenceMLP,
        output_matrix: Linear,
        qkv_streams: Option<Vec<Arc<CudaStream>>>,
        norm_type: NormType,
    ) -> Self {
        Self {
            attention: Attention::new(query, key, value, qkv_streams, norm_type),
            position_matrix,
            fcs,
            output_matrix,
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
            position: cursor.matrix(self.position_matrix.rows(), self.position_matrix.cols()),
            feed_forward: self.fcs.get_meta_data(cursor),
            output: self.output_matrix.get_meta_data(cursor),
        }
    }

    pub(super) fn get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
        let mut data = self.attention.get_data(runtime);
        data.push(HostData::new(self.position_matrix.to_host(runtime)));
        data.extend(self.fcs.get_data(runtime));
        data.extend(self.output_matrix.get_data(runtime));
        data
    }

    pub(super) fn forward(&mut self, input: &Matrix, runtime: &mut CudaRuntime) -> Matrix {
        let positioned = runtime.matrix_add(input, &self.position_matrix);
        let x = self.attention.forward(&positioned, &positioned, runtime);
        self.finish_forward(positioned, x, runtime)
    }

    pub(super) fn forward_mask(&mut self, input: &Matrix, runtime: &mut CudaRuntime) -> Matrix {
        let positioned = runtime.matrix_add(input, &self.position_matrix);
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
        let mut output = runtime.matrix_add(&x, &ffn);
        runtime.recycle_matrix(x);
        runtime.recycle_matrix(ffn);

        self.attention.normalize(&mut output, runtime);

        let result = self.output_matrix.forward(&output, None, runtime, None);
        runtime.recycle_matrix(output);
        result
    }
}
