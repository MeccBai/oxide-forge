use crate::cuda::{container::Matrix, runtime::CudaRuntime};
use crate::net::linear::{Linear, LinearMomentum};
use crate::net::metadata::{HostData, MetadataCursor};
use crate::net::mlp::{InferenceMLP, TrainingMlp};
use crate::net::node::{BinaryNode, BinaryOp, SingleType, TrainingBinaryNode, TrainingSingleNode};
use cuda_core::CudaStream;

use std::sync::Arc;

pub use super::inference::TransformerMetadata;
use super::{NormType, PositionEncoding, attention::Attention, inference::InferenceBlock};

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
        self.block.forward(input, runtime)
    }
}

pub struct TrainingTransformer {
    attention: Attention,
    position_encoding: PositionEncoding,
    fcs: TrainingMlp,
    output_matrix: Linear,
    output_optimizer: LinearMomentum,
    feed_forward_residual: TrainingBinaryNode,
    feed_forward_normalization: TrainingSingleNode,
    cache: Option<TransformerCache>,
}

struct TransformerCache {
    encoded: Matrix,
}

impl TrainingTransformer {
    pub fn new<F>(
        q_matrix: Linear,
        k_matrix: Linear,
        v_matrix: Linear,
        position_encoding: F,
        fcs: TrainingMlp,
        output_matrix: Linear,
        norm_type: NormType,
    ) -> Self
    where
        F: Fn(&Matrix, &mut CudaRuntime) -> Matrix + 'static,
    {
        Self {
            attention: Attention::new(q_matrix, k_matrix, v_matrix, None, norm_type),
            position_encoding: Box::new(position_encoding),
            fcs,
            output_matrix,
            output_optimizer: LinearMomentum::default(),
            feed_forward_residual: TrainingBinaryNode::new(BinaryOp::Add),
            feed_forward_normalization: TrainingSingleNode::new(match norm_type {
                NormType::Layer => SingleType::LayerNorm,
                NormType::Rms => SingleType::RmsNorm,
            }),
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
            feed_forward: self.fcs.get_meta_data(cursor),
            output: self.output_matrix.get_meta_data(cursor),
        }
    }

    pub fn get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
        let mut data = self.attention.get_data(runtime);
        data.extend(self.fcs.get_data(runtime));
        data.extend(self.output_matrix.get_data(runtime));
        data
    }

    pub fn forward(&mut self, input: &Matrix, runtime: &mut CudaRuntime) -> Matrix {
        let positioned = (self.position_encoding)(input, runtime);
        assert_eq!(positioned.rows(), input.rows());
        assert_eq!(positioned.cols(), input.cols());
        let first_output = self.attention.forward_self_training(positioned, runtime);
        let ffn = self.fcs.forward(first_output, runtime);
        let second_pre_norm = self.feed_forward_residual.forward_borrowed(
            &[
                self.fcs.input().expect("training MLP input cache missing"),
                &ffn,
            ],
            runtime,
        );
        runtime.recycle_matrix(ffn);
        let encoded = self
            .feed_forward_normalization
            .forward(second_pre_norm, runtime);

        let output = self.output_matrix.forward(&encoded, None, runtime, None);
        self.cache = Some(TransformerCache { encoded });
        output
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

    /// Backpropagates one sample while retaining parameter gradients for a
    /// later mini-batch optimizer step.
    pub fn backward_accumulate(
        &mut self,
        output_gradient: &Matrix,
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
        self.output_optimizer.accumulate(
            &cache.encoded,
            &output_gradient,
            output_bias_gradient.as_ref(),
            runtime,
        );

        let second_gradient = self
            .feed_forward_normalization
            .backward(encoded_gradient, runtime);
        let [direct_gradient, ffn_output_gradient]: [Matrix; 2] = self
            .feed_forward_residual
            .backward(second_gradient, runtime)
            .try_into()
            .unwrap_or_else(|_| unreachable!("residual Add has two inputs"));
        let ffn_input_gradient = self.fcs.backward_accumulate(&ffn_output_gradient, runtime);
        runtime.recycle_matrix(ffn_output_gradient);
        let first_output_gradient = BinaryNode::new(BinaryOp::Add)
            .forward_owned(vec![direct_gradient, ffn_input_gradient], runtime);

        let positioned_gradient = self
            .attention
            .backward_self_accumulate(first_output_gradient, runtime);

        positioned_gradient
    }

    /// Applies every accumulated transformer gradient in one Momentum SGD step.
    pub fn step(
        &mut self,
        learning_rate: f32,
        momentum: f32,
        batch_len: usize,
        runtime: &mut CudaRuntime,
    ) {
        self.output_optimizer.step(
            &mut self.output_matrix,
            learning_rate,
            momentum,
            batch_len,
            runtime,
        );
        self.fcs.step(learning_rate, momentum, batch_len, runtime);
        self.attention
            .step(learning_rate, momentum, batch_len, runtime);
    }
}
