use crate::cuda::container::Matrix;
use crate::cuda::runtime::CudaRuntime;
use crate::net::linear::Linear;
use crate::net::mlp::InferenceMLP;
use cuda_core::CudaStream;
use std::sync::Arc;

use crate::net::metadata::{HostData, MetadataCursor};

pub use super::inference::TransformerMetadata;
use super::{NormType, inference::InferenceBlock};

pub struct InferenceTransformer {
    block: InferenceBlock,
}

impl InferenceTransformer {
    pub fn new<F>(
        q_matrix: Linear,
        k_matrix: Linear,
        v_matrix: Linear,
        position_encoding: F,
        fcs: InferenceMLP,
        output_matrix: Linear,
        qkv_streams: Option<Vec<Arc<CudaStream>>>,
        norm_type: NormType,
    ) -> Self
    where
        F: Fn(&Matrix, &mut CudaRuntime) -> Matrix + 'static,
    {
        Self {
            block: InferenceBlock::new(
                q_matrix,
                k_matrix,
                v_matrix,
                Box::new(position_encoding),
                fcs,
                output_matrix,
                qkv_streams,
                norm_type,
            ),
        }
    }

    pub fn get_meta_data(&self, cursor: &mut MetadataCursor) -> TransformerMetadata {
        self.block.get_meta_data(cursor)
    }

    pub fn get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
        self.block.get_data(runtime)
    }

    pub fn forward(&mut self, input: &Matrix, runtime: &mut CudaRuntime) -> Matrix {
        self.block.forward_mask(input, runtime)
    }
}
