mod cuda;
mod net;

use crate::net::linear::Activation::{Gelu, Identity};
use crate::net::linear::Linear;
use crate::net::mlp::InferenceMLP;

fn main() {
    let runtime = cuda::CudaRuntime::new().unwrap();

    // seq × hidden
    let input = runtime.new_matrix(cuda::InitType::Random, 1024, 768);

    // hidden × hidden
    let matrix_q = Linear::new(
        runtime.new_matrix(cuda::InitType::Random, 768, 768),
        None,
        Identity,
    );
    let matrix_k = Linear::new(
        runtime.new_matrix(cuda::InitType::Random, 768, 768),
        None,
        Identity,
    );
    let matrix_v = Linear::new(
        runtime.new_matrix(cuda::InitType::Random, 768, 768),
        None,
        Identity,
    );

    // seq × hidden
    let matrix_position = runtime.new_matrix(cuda::InitType::Random, 1024, 768);

    let fcs = InferenceMLP::new(
        vec![
            Linear::new(
                runtime.new_matrix(cuda::InitType::Random, 768, 3072),
                None,
                Gelu,
            ),
            Linear::new(
                runtime.new_matrix(cuda::InitType::Random, 3072, 768),
                None,
                Identity,
            ),
        ],
        None,
    );

    let output_matrix = Linear::new(
        runtime.new_matrix(cuda::InitType::Random, 768, 256),
        None,
        Identity,
    );

    let mut transformer = net::transformer::InferenceTransformer::new(
        matrix_q,
        matrix_k,
        matrix_v,
        matrix_position,
        fcs,
        output_matrix,
    );

    for i in 0..10 {
        let time_now = std::time::Instant::now();
        let output = transformer.forward(&input, &runtime);
        let time_duration = time_now.elapsed();
        println!("Time taken: {:?}", time_duration);
    }
}
