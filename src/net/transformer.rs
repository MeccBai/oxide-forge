use crate::cuda::{
    BinaryOp::{Add, Sub},
    container::Matrix,
    runtime::CudaRuntime,
};
use crate::net::linear::Linear;
use crate::net::mlp::{InferenceMLP, TrainingMlp};

pub struct InferenceTransformer {
    q_matrix: Linear,
    k_matrix: Linear,
    v_matrix: Linear,
    position_matrix: Matrix,
    fcs: InferenceMLP,
    output_matrix: Linear,
}

impl InferenceTransformer {
    pub fn new(
        q_matrix: Linear,
        k_matrix: Linear,
        v_matrix: Linear,
        position_matrix: Matrix,
        fcs: InferenceMLP,
        output_matrix: Linear,
    ) -> Self {
        Self {
            q_matrix,
            k_matrix,
            v_matrix,
            position_matrix,
            fcs,
            output_matrix,
        }
    }

    pub fn forward(&mut self, input: &Matrix, runtime: &CudaRuntime) -> Matrix {
        let positioned = runtime.matrix_add(input, &self.position_matrix);

        let q = self.q_matrix.forward(&positioned, None, runtime);
        let k = self.k_matrix.forward(&positioned, None, runtime);
        let v = self.v_matrix.forward(&positioned, None, runtime);

        let k_t = runtime.matrix_transpose(&k);
        let mut scores = runtime.matrix_multiply(&q, &k_t);

        scores.scale(1.0 / (q.cols() as f32).sqrt(), runtime);
        scores.softmax_rows(runtime);

        let attention = runtime.matrix_multiply(&scores, &v);

        let mut x = runtime.matrix_add(&positioned, &attention);
        x.layer_norm(runtime);

        let ffn = self.fcs.forward(&x, runtime);
        let mut output = runtime.matrix_add(&x, &ffn);
        output.layer_norm(runtime);

        let output = self.output_matrix.forward(&output, None, runtime);
        runtime.sync();
        output
    }
}

pub struct TrainingTransformer {
    q_matrix: Linear,
    k_matrix: Linear,
    v_matrix: Linear,
    position_matrix: Matrix,
    fcs: TrainingMlp,
    output_matrix: Linear,
    cache: Option<TransformerCache>,
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
        }
    }

    pub fn forward(&mut self, input: &Matrix, runtime: &CudaRuntime) -> Matrix {
        let positioned = runtime.matrix_add(input, &self.position_matrix);
        let q = self.q_matrix.forward(&positioned, None, runtime);
        let k = self.k_matrix.forward(&positioned, None, runtime);
        let v = self.v_matrix.forward(&positioned, None, runtime);

        let k_t = runtime.matrix_transpose(&k);
        let mut probabilities = runtime.matrix_multiply(&q, &k_t);
        probabilities.scale(1.0 / (q.cols() as f32).sqrt(), runtime);
        probabilities.softmax_rows(runtime);

        let attention = runtime.matrix_multiply(&probabilities, &v);
        let first_pre_norm = runtime.matrix_add(&positioned, &attention);
        let mut first_output = runtime.matrix_copy(&first_pre_norm);
        first_output.layer_norm(runtime);

        let ffn = self.fcs.forward(first_output, runtime);
        let second_pre_norm = runtime.matrix_add(
            self.fcs.input().expect("training MLP input cache missing"),
            &ffn,
        );
        let mut encoded = runtime.matrix_copy(&second_pre_norm);
        encoded.layer_norm(runtime);

        let output = self.output_matrix.forward(&encoded, None, runtime);
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
        runtime.sync();
        output
    }

    pub fn backward(
        &mut self,
        output_gradient: &Matrix,
        learning_rate: f32,
        runtime: &CudaRuntime,
    ) -> Matrix {
        let cache = self
            .cache
            .as_ref()
            .expect("forward must run before backward");

        let output_pre_activation = self
            .output_matrix
            .needs_pre_activation()
            .then(|| self.output_matrix.affine(&cache.encoded, None, runtime));
        let (output_gradient, output_bias_gradient) =
            self.output_matrix
                .backward(output_pre_activation.as_ref(), output_gradient, runtime);
        let encoded_gradient = self.output_matrix.input_gradient(&output_gradient, runtime);
        self.output_matrix.learn(
            &cache.encoded,
            &output_gradient,
            output_bias_gradient.as_ref(),
            learning_rate,
            runtime,
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
                .then(|| linear.affine(&cache.positioned, None, runtime));
            let (gradient, bias_gradient) =
                linear.backward(pre_activation.as_ref(), projected_gradient, runtime);
            let input_gradient = linear.input_gradient(&gradient, runtime);
            positioned_gradient.binary_assign(&input_gradient, Add, runtime);
            linear.learn(
                &cache.positioned,
                &gradient,
                bias_gradient.as_ref(),
                learning_rate,
                runtime,
            );
        }

        let input_gradient = runtime.matrix_copy(&positioned_gradient);
        let mut position_update = positioned_gradient;
        position_update.scale(learning_rate, runtime);
        self.position_matrix
            .binary_assign(&position_update, Sub, runtime);
        runtime.sync();
        input_gradient
    }
}
