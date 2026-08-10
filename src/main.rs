mod cuda;

use crate::cuda::InitType;

fn main() {
    let mut cuda_runtime = cuda::CudaRuntime::new().unwrap();

    let mut vec1 = cuda_runtime.new_vector(InitType::Random, 2048);

    vec1.scale(100.0, &cuda_runtime);

    let matrix1 = cuda_runtime.broadcast(&vec1, 10);

    let vec1_host = vec1.to_host(&cuda_runtime);

    for i in 0..10 {
        println!("vec1[{}] = {}", i, vec1_host[i]);
    }

    print!("[");
    for i in 0..10 {
        print!("[");
        for j in 0..10 {
            print!("{},", matrix1.to_host(&cuda_runtime)[i * matrix1.cols() + j]);
        }
        println!("],");
    }
    print!("]");
}
