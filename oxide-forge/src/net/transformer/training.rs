use crate::cuda::{container::Matrix, runtime::CudaRuntime};
use crate::graph::{GraphNode, LearnConfig};
use crate::net::linear::{Linear, LinearTrainingState};
use crate::net::metadata::{HostData, HostDataCursor, MetadataCursor};
use crate::net::mlp::TrainingMlp;
use crate::net::node::{BinaryNode, BinaryOp, SingleType, TrainingBinaryNode, TrainingSingleNode};
use crate::net::transformer::attention::multi::Attention;

use super::inference::TransformerMetadata;
use super::{NormType, PositionEncoding};

pub struct TrainingTransformer<const HEADS: usize = 1, const MASKED: bool = false> {
    attention: Attention<HEADS>,
    norm_type: NormType,
    attention_residual: TrainingBinaryNode,
    attention_normalization: TrainingSingleNode,
    position_encoding: PositionEncoding,
    fcs: TrainingMlp,
    output_matrix: Linear,
    output_training: LinearTrainingState,
    feed_forward_residual: TrainingBinaryNode,
    feed_forward_normalization: TrainingSingleNode,
    cache: Option<TransformerCache>,
}

struct TransformerCache {
    encoded: Matrix,
}

impl<const HEADS: usize, const MASKED: bool> TrainingTransformer<HEADS, MASKED> {
    pub fn new(
        q_matrix: Linear,
        k_matrix: Linear,
        v_matrix: Linear,
        position_encoding: PositionEncoding,
        fcs: TrainingMlp,
        output_matrix: Linear,
        norm_type: NormType,
    ) -> Self {
        Self {
            attention: Attention::new(q_matrix, k_matrix, v_matrix, None),
            norm_type,
            attention_residual: TrainingBinaryNode::new(BinaryOp::Add),
            attention_normalization: TrainingSingleNode::new(match norm_type {
                NormType::Layer => SingleType::LayerNorm,
                NormType::Rms => SingleType::RmsNorm,
            }),
            position_encoding,
            fcs,
            output_matrix,
            output_training: LinearTrainingState::default(),
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
        self.clear_cache(runtime);
        let positioned = self.position_encoding.forward(input, runtime);
        assert_eq!(positioned.rows(), input.rows());
        assert_eq!(positioned.cols(), input.cols());
        let (attention, attention_state) = self
            .attention
            .forward_self_training::<MASKED>(positioned, runtime);
        let first_pre_norm = self
            .attention_residual
            .forward_borrowed(&[&attention_state.input, &attention], runtime);
        runtime.recycle_matrix(attention);
        self.attention.retain_training_state(attention_state);
        let first_output = self
            .attention_normalization
            .forward(first_pre_norm, runtime);
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

    pub fn backward(&mut self, output_gradient: &Matrix, runtime: &mut CudaRuntime) -> Matrix {
        self.backward_accumulate(output_gradient, runtime)
    }

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
        self.output_training.accumulate(
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

        let attention_residual_gradient = self
            .attention_normalization
            .backward(first_output_gradient, runtime);
        let [direct_gradient, attention_gradient]: [Matrix; 2] = self
            .attention_residual
            .backward(attention_residual_gradient, runtime)
            .try_into()
            .unwrap_or_else(|_| unreachable!("attention residual Add has two inputs"));
        let projection_gradient = self
            .attention
            .backward_self_accumulate(attention_gradient, runtime);
        BinaryNode::new(BinaryOp::Add)
            .forward_owned(vec![direct_gradient, projection_gradient], runtime)
    }

    pub fn learn(
        &mut self,
        learning_rate: f32,
        momentum: f32,
        batch_len: usize,
        runtime: &mut CudaRuntime,
    ) {
        self.output_training.learn(
            &mut self.output_matrix,
            learning_rate,
            momentum,
            batch_len,
            runtime,
        );
        self.fcs.learn(learning_rate, momentum, batch_len, runtime);
        self.attention
            .learn(learning_rate, momentum, batch_len, runtime);
    }

    pub fn clear_cache(&mut self, runtime: &mut CudaRuntime) {
        if let Some(cache) = self.cache.take() {
            runtime.recycle_matrix(cache.encoded);
        }
        self.attention.clear_training_cache(runtime);
        self.fcs.clear_cache(runtime);
        self.attention_residual.clear_cache(runtime);
        self.attention_normalization.clear_cache(runtime);
        self.feed_forward_residual.clear_cache(runtime);
        self.feed_forward_normalization.clear_cache(runtime);
    }
}

impl<const HEADS: usize, const MASKED: bool> GraphNode for TrainingTransformer<HEADS, MASKED> {
    fn forward(&mut self, mut inputs: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        assert_eq!(inputs.len(), 1, "Transformer forward expects one matrix");
        let input = inputs.pop().unwrap();
        let output = TrainingTransformer::forward(self, &input, runtime);
        runtime.recycle_matrix(input);
        vec![output]
    }

    fn backward(&mut self, mut gradients: Vec<Matrix>, runtime: &mut CudaRuntime) -> Vec<Matrix> {
        assert_eq!(
            gradients.len(),
            1,
            "Transformer backward expects one gradient"
        );
        let output_gradient = gradients.pop().unwrap();
        let input_gradient = self.backward_accumulate(&output_gradient, runtime);
        runtime.recycle_matrix(output_gradient);
        self.clear_cache(runtime);
        vec![input_gradient]
    }

    fn learn(&mut self, config: LearnConfig, runtime: &mut CudaRuntime) {
        TrainingTransformer::learn(
            self,
            config.learning_rate,
            config.momentum,
            config.batch_len,
            runtime,
        );
    }

    fn clear_cache(&mut self, runtime: &mut CudaRuntime) {
        TrainingTransformer::clear_cache(self, runtime);
    }

    fn get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
        TrainingTransformer::get_data(self, runtime)
    }

    fn set_data(&mut self, data: &mut HostDataCursor, runtime: &CudaRuntime) {
        TrainingTransformer::set_data(self, data, runtime);
    }
}
