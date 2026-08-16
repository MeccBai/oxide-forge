use crate::cuda;

pub struct Transformer<const R: usize, const C: usize> {
    q_matrix: cuda::matrix::Matrix,
    k_matrix: cuda::matrix::Matrix,
    v_matrix: cuda::matrix::Matrix,
    position_matrix: cuda::matrix::Matrix,
}

impl<const R: usize, const C: usize> Transformer<R, C> {
    pub fn new(
        q_matrix: cuda::matrix::Matrix,
        k_matrix: cuda::matrix::Matrix,
        v_matrix: cuda::matrix::Matrix,
        position_matrix: cuda::matrix::Matrix,
    ) -> Self {
        assert!(q_matrix.rows() == R && q_matrix.cols() == C);
        assert!(k_matrix.rows() == R && k_matrix.cols() == C);
        assert!(v_matrix.rows() == R && v_matrix.cols() == C);
        assert!(position_matrix.rows() == R && position_matrix.cols() == C);

        Self {
            q_matrix,
            k_matrix,
            v_matrix,
            position_matrix,
        }
    }

    pub fn forward(
        &self,
        input: &cuda::matrix::Matrix,
        runtime: &cuda::runtime::CudaRuntime,
    ) -> cuda::matrix::Matrix {
        let input_with_position = runtime.matrix_add(input, &self.position_matrix);

        let qs = runtime.matrix_multiply(&input_with_position, &self.q_matrix);
        let ks = runtime.matrix_multiply(&input_with_position, &self.k_matrix);
        let vs = runtime.matrix_multiply(&input_with_position, &self.v_matrix);

        let kst = runtime.matrix_transpose(&ks);
        let mut scores = runtime.matrix_multiply(&qs, &kst);

        let scale = 1.0 / (qs.cols() as f32).sqrt();
        scores.scale(scale, runtime);
        scores.softmax_rows(runtime);
        let attention = runtime.matrix_multiply(&scores, &vs);

        let mut net_input = runtime.matrix_add(&attention, &input_with_position);

        net_input.layer_norm(&runtime);

        println!("[{}:{}]", net_input.rows(), net_input.cols());

        let fc1 = runtime.new_matrix(cuda::InitType::Random, 768, 3072);

        let fc2 = runtime.new_matrix(cuda::InitType::Random, 3072, 768);

        let mut hidden = runtime.matrix_multiply(&net_input, &fc1);

        let gelu =
            move |x: f32| 0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh());

        hidden.for_each(&runtime, gelu);

        let mlp_output = runtime.matrix_multiply(&hidden, &fc2);
        let mut output = runtime.matrix_add(&net_input, &mlp_output);

        output.layer_norm(&runtime);

        output
    }

}
