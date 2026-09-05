use oxide_forge::cuda::{CudaRuntime, InitType::Random};

fn main() {
    let mut runtime = CudaRuntime::new().unwrap();
    let mat1 = runtime.new_matrix(Random, 512, 512);
    let mat2 = runtime.new_matrix(Random, 512, 512);

    let _mat3 = runtime.matrix_multiply(&mat1, &mat2);

    runtime.sync();
}
