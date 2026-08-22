use crate::cuda::{
    BinaryOp::{Add, Sub},
    container::Matrix,
    runtime::CudaRuntime,
};
use crate::net::checkpoint;
use crate::net::linear::Linear;
use crate::net::mlp::{InferenceMLP, TrainingMlp};
use cuda_core::CudaStream;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormType {
    Rms,
    #[default]
    Layer,
}

pub struct InferenceTransformer {
    pub(super) q_matrix: Linear,
    pub(super) k_matrix: Linear,
    pub(super) v_matrix: Linear,
    pub(super) position_matrix: Matrix,
    pub(super) fcs: InferenceMLP,
    pub(super) output_matrix: Linear,
    qkv_streams: Option<Vec<Arc<CudaStream>>>,
    pub(super) norm_type: NormType,
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
        if let Some(streams) = &qkv_streams {
            assert_eq!(streams.len(), 3, "Q/K/V execution requires three streams");
        }
        Self {
            q_matrix,
            k_matrix,
            v_matrix,
            position_matrix,
            fcs,
            output_matrix,
            qkv_streams,
            norm_type,
        }
    }

    pub fn forward(&mut self, input: &Matrix, runtime: &mut CudaRuntime) -> Matrix {
        let positioned = runtime.matrix_add(input, &self.position_matrix);

        if let Some(streams) = &self.qkv_streams {
            runtime.fork_streams(streams);
        } else {
            self.qkv_streams = Some(runtime.create_extra_streams(3));
        }
        let streams = self.qkv_streams.as_ref().unwrap();

        let q = self
            .q_matrix
            .forward(&positioned, None, runtime, Some(streams[0].as_ref()));
        let k = self
            .k_matrix
            .forward(&positioned, None, runtime, Some(streams[1].as_ref()));
        let v = self
            .v_matrix
            .forward(&positioned, None, runtime, Some(streams[2].as_ref()));

        runtime.join_streams(&streams[..2]);

        let k_t = runtime.matrix_transpose(&k);
        runtime.recycle_matrix(k);
        let mut scores = runtime.matrix_multiply(&q, &k_t);
        let attention_scale = 1.0 / (q.cols() as f32).sqrt();
        runtime.recycle_matrix(q);
        runtime.recycle_matrix(k_t);

        scores.scale(attention_scale, runtime);
        scores.softmax_rows(runtime);

        runtime.join_streams(&streams[2..]);
        let attention = runtime.matrix_multiply(&scores, &v);
        runtime.recycle_matrix(scores);
        runtime.recycle_matrix(v);

        let mut x = runtime.matrix_add(&positioned, &attention);
        runtime.recycle_matrix(positioned);
        runtime.recycle_matrix(attention);
        match self.norm_type {
            NormType::Rms => x.rms_norm(runtime),
            NormType::Layer => x.layer_norm(runtime),
        }

        let ffn = self.fcs.forward(&x, runtime);
        let mut output = runtime.matrix_add(&x, &ffn);
        runtime.recycle_matrix(x);
        runtime.recycle_matrix(ffn);

        match self.norm_type {
            NormType::Rms => output.rms_norm(runtime),
            NormType::Layer => output.layer_norm(runtime),
        }

        let result = self.output_matrix.forward(&output, None, runtime, None);
        runtime.recycle_matrix(output);
        result
    }

    pub fn dump_to_file<P: AsRef<Path>>(
        &self,
        path: P,
        runtime: &CudaRuntime,
    ) -> Result<(), Box<dyn Error>> {
        checkpoint::dump_inference_transformer_file(self, path.as_ref(), runtime)
    }

    pub fn load_from_file<P: AsRef<Path>>(
        path: P,
        runtime: &mut CudaRuntime,
    ) -> Result<Self, Box<dyn Error>> {
        checkpoint::load_inference_transformer_file(path.as_ref(), runtime)
    }
}

pub struct TrainingTransformer {
    pub(super) q_matrix: Linear,
    pub(super) k_matrix: Linear,
    pub(super) v_matrix: Linear,
    pub(super) position_matrix: Matrix,
    pub(super) fcs: TrainingMlp,
    pub(super) output_matrix: Linear,
    cache: Option<TransformerCache>,
    qkv_streams: Option<Vec<Arc<CudaStream>>>,
}

struct TransformerCache {
    positioned: Matrix,
    q: Matrix,
    k: Matrix,
    v: Matrix,
    probabilities: Matrix,
    first_pre_norm: Matrix,
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
    ) -> Self {
        Self {
            q_matrix,
            k_matrix,
            v_matrix,
            position_matrix,
            fcs,
            output_matrix,
            cache: None,
            qkv_streams: None,
        }
    }

    pub fn forward(&mut self, input: &Matrix, runtime: &mut CudaRuntime) -> Matrix {
        let positioned = runtime.matrix_add(input, &self.position_matrix);

        if let Some(streams) = &self.qkv_streams {
            runtime.fork_streams(streams);
        } else {
            self.qkv_streams = Some(runtime.create_extra_streams(3));
        }
        let streams = self.qkv_streams.as_ref().unwrap();

        let q = self
            .q_matrix
            .forward(&positioned, None, runtime, Some(streams[0].as_ref()));
        let k = self
            .k_matrix
            .forward(&positioned, None, runtime, Some(streams[1].as_ref()));
        let v = self
            .v_matrix
            .forward(&positioned, None, runtime, Some(streams[2].as_ref()));

        runtime.join_streams(&streams[..2]);

        let k_t = runtime.matrix_transpose(&k);
        let mut probabilities = runtime.matrix_multiply(&q, &k_t);
        probabilities.scale(1.0 / (q.cols() as f32).sqrt(), runtime);
        probabilities.softmax_rows(runtime);

        runtime.join_streams(&streams[2..]);
        let attention = runtime.matrix_multiply(&probabilities, &v);
        let first_pre_norm = runtime.matrix_add(&positioned, &attention);
        let mut first_output = runtime.clone_matrix(&first_pre_norm);
        first_output.layer_norm(runtime);

        let ffn = self.fcs.forward(first_output, runtime);
        let second_pre_norm = runtime.matrix_add(
            self.fcs.input().expect("training MLP input cache missing"),
            &ffn,
        );
        let mut encoded = runtime.clone_matrix(&second_pre_norm);
        encoded.layer_norm(runtime);

        let output = self.output_matrix.forward(&encoded, None, runtime, None);
        self.cache = Some(TransformerCache {
            positioned,
            q,
            k,
            v,
            probabilities,
            first_pre_norm,
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

        let second_gradient =
            runtime.layer_norm_backward(&cache.second_pre_norm, &encoded_gradient);
        let ffn_input_gradient = self.fcs.backward(&second_gradient, learning_rate, runtime);
        let mut first_output_gradient = second_gradient;
        first_output_gradient.binary_assign(&ffn_input_gradient, Add, runtime);

        let first_gradient =
            runtime.layer_norm_backward(&cache.first_pre_norm, &first_output_gradient);

        let v_t = runtime.matrix_transpose(&cache.v);
        let probabilities_gradient = runtime.matrix_multiply(&first_gradient, &v_t);
        let probabilities_t = runtime.matrix_transpose(&cache.probabilities);
        let v_gradient = runtime.matrix_multiply(&probabilities_t, &first_gradient);

        let mut score_gradient =
            runtime.softmax_rows_backward(&cache.probabilities, &probabilities_gradient);
        score_gradient.scale(1.0 / (cache.q.cols() as f32).sqrt(), runtime);

        let q_gradient = runtime.matrix_multiply(&score_gradient, &cache.k);
        let score_gradient_t = runtime.matrix_transpose(&score_gradient);
        let k_gradient = runtime.matrix_multiply(&score_gradient_t, &cache.q);

        let mut positioned_gradient = first_gradient;
        for (linear, projected_gradient) in [
            (&mut self.q_matrix, &q_gradient),
            (&mut self.k_matrix, &k_gradient),
            (&mut self.v_matrix, &v_gradient),
        ] {
            let pre_activation = linear
                .needs_pre_activation()
                .then(|| linear.affine(&cache.positioned, None, runtime, None));
            let (gradient, bias_gradient) =
                linear.backward(pre_activation.as_ref(), projected_gradient, runtime, None);
            let input_gradient = linear.input_gradient(&gradient, runtime, None);
            positioned_gradient.binary_assign(&input_gradient, Add, runtime);
            linear.learn(
                &cache.positioned,
                &gradient,
                bias_gradient.as_ref(),
                learning_rate,
                runtime,
                None,
            );
        }

        let input_gradient = runtime.clone_matrix(&positioned_gradient);
        let mut position_update = positioned_gradient;
        position_update.scale(learning_rate, runtime);
        self.position_matrix
            .binary_assign(&position_update, Sub, runtime);
        input_gradient
    }

    pub fn dump_to_file<P: AsRef<Path>>(
        &self,
        path: P,
        runtime: &CudaRuntime,
    ) -> Result<(), Box<dyn Error>> {
        checkpoint::dump_training_transformer_file(self, path.as_ref(), runtime)
    }

    pub fn load_from_file<P: AsRef<Path>>(
        path: P,
        runtime: &mut CudaRuntime,
    ) -> Result<Self, Box<dyn Error>> {
        checkpoint::load_training_transformer_file(path.as_ref(), runtime)
    }
}
