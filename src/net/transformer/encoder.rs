use crate::cuda::{
    BinaryOp::{Add, Sub},
    container::Matrix,
    runtime::CudaRuntime,
};
use crate::net::linear::Linear;
use crate::net::metadata::{HostData, MetadataCursor};
use crate::net::mlp::{InferenceMLP, TrainingMlp};
use cuda_core::CudaStream;

use std::sync::Arc;

pub use super::inference::TransformerMetadata;
use super::{NormType, attention::Attention, inference::InferenceBlock};

pub struct InferenceTransformer {
    block: InferenceBlock,
}

impl InferenceTransformer {
    pub fn new(
        q_matrix: Linear,
        k_matrix: Linear,
        v_matrix: Linear,
        position_matrix: Matrix,
        fcs: InferenceMLP,
        output_matrix: Linear,
        qkv_streams: Option<Vec<Arc<CudaStream>>>,
        norm_type: NormType,
    ) -> Self {
        Self {
            block: InferenceBlock::new(
                q_matrix,
                k_matrix,
                v_matrix,
                position_matrix,
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
        self.block.forward(input, runtime)
    }
}

pub struct TrainingTransformer {
    attention: Attention,
    position_matrix: Matrix,
    fcs: TrainingMlp,
    output_matrix: Linear,
    cache: Option<TransformerCache>,
}

struct TransformerCache {
    second_pre_norm: Matrix,
    encoded: Matrix,
}

impl TrainingTransformer {
    pub fn new(
        q_matrix: Linear,
        k_matrix: Linear,
        v_matrix: Linear,
        position_matrix: Matrix,
        fcs: TrainingMlp,
        output_matrix: Linear,
        norm_type: NormType,
    ) -> Self {
        Self {
            attention: Attention::new(q_matrix, k_matrix, v_matrix, None, norm_type),
            position_matrix,
            fcs,
            output_matrix,
            cache: None,
        }
    }

    pub fn get_meta_data(&self, cursor: &mut MetadataCursor) -> TransformerMetadata {
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

    pub fn get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
        let mut data = self.attention.get_data(runtime);
        data.push(HostData::new(self.position_matrix.to_host(runtime)));
        data.extend(self.fcs.get_data(runtime));
        data.extend(self.output_matrix.get_data(runtime));
        data
    }

    pub fn forward(&mut self, input: &Matrix, runtime: &mut CudaRuntime) -> Matrix {
        let positioned = runtime.matrix_add(input, &self.position_matrix);
        let first_output = self.attention.forward_self_training(positioned, runtime);
        let ffn = self.fcs.forward(first_output, runtime);
        let second_pre_norm = runtime.matrix_add(
            self.fcs.input().expect("training MLP input cache missing"),
            &ffn,
        );
        let mut encoded = runtime.clone_matrix(&second_pre_norm);
        self.attention.normalize(&mut encoded, runtime);

        let output = self.output_matrix.forward(&encoded, None, runtime, None);
        self.cache = Some(TransformerCache {
            second_pre_norm,
            encoded,
        });
        output
    }

    pub fn backward(
        &mut self,
        output_gradient: &Matrix,
        learning_rate: f32,
        runtime: &mut CudaRuntime,
    ) -> Matrix {
        let cache = self
            .cache
            .as_ref()
            .expect("forward must run before backward");

        let output_pre_activation = self.output_matrix.needs_pre_activation().then(|| {
            self.output_matrix
                .affine(&cache.encoded, None, runtime, None)
        });
        let (output_gradient, output_bias_gradient) = self.output_matrix.backward(
            output_pre_activation.as_ref(),
            output_gradient,
            runtime,
            None,
        );

        let encoded_gradient = self
            .output_matrix
            .input_gradient(&output_gradient, runtime, None);
        self.output_matrix.learn(
            &cache.encoded,
            &output_gradient,
            output_bias_gradient.as_ref(),
            learning_rate,
            runtime,
            None,
        );

        let second_gradient = self.attention.normalization_backward(
            &cache.second_pre_norm,
            &encoded_gradient,
            runtime,
        );
        let ffn_input_gradient = self.fcs.backward(&second_gradient, learning_rate, runtime);
        let mut first_output_gradient = second_gradient;
        first_output_gradient.binary_assign(&ffn_input_gradient, Add, runtime);

        let positioned_gradient =
            self.attention
                .backward_self(&first_output_gradient, learning_rate, runtime);

        let input_gradient = runtime.clone_matrix(&positioned_gradient);
        let mut position_update = positioned_gradient;
        position_update.scale(learning_rate, runtime);
        self.position_matrix
            .binary_assign(&position_update, Sub, runtime);
        input_gradient
    }
}
