use crate::cuda::{container::Matrix, runtime::CudaRuntime};
use crate::net::linear::Linear;
use crate::net::mlp::Mlp;

pub struct Transformer {
    q_matrix: Linear,
    k_matrix: Linear,
    v_matrix: Linear,
    position_matrix: Matrix,
    fcs: Mlp,
    output_matrix: Linear,
}

impl Transformer {
    pub fn new(
        q_matrix: Linear,
        k_matrix: Linear,
        v_matrix: Linear,
        position_matrix: Matrix,
        fcs: Mlp,
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

    pub fn forward(&self, input: &Matrix, runtime: &CudaRuntime) -> Matrix {
        let positioned = runtime.matrix_add(input, &self.position_matrix);

        let q = self.q_matrix.forward(&positioned, runtime);
        let k = self.k_matrix.forward(&positioned, runtime);
        let v = self.v_matrix.forward(&positioned, runtime);

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

        let output = self.output_matrix.forward(&output, runtime);
        runtime.sync();
        output
    }
}
