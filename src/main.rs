mod cuda;
mod net;

use crate::cuda::InitType::Random;
use crate::net::linear::Activation::{Gelu, Identity};
use crate::net::linear::Linear;
use crate::net::mlp::InferenceMLP;

fn main() {
    let mut runtime = cuda::CudaRuntime::new().unwrap();

    // seq × hidden
    let input = runtime.new_matrix(Random, 1024, 768);

    // hidden × hidden
    let matrix_q = Linear::new(runtime.new_matrix(Random, 768, 768), None, Identity);
    let matrix_k = Linear::new(runtime.new_matrix(Random, 768, 768), None, Identity);
    let matrix_v = Linear::new(runtime.new_matrix(Random, 768, 768), None, Identity);

    // seq × hidden
    let matrix_position = runtime.new_matrix(Random, 1024, 768);

    let fcs = InferenceMLP::new(
        vec![
            Linear::new(runtime.new_matrix(Random, 768, 3072), None, Gelu),
            Linear::new(runtime.new_matrix(Random, 3072, 768), None, Identity),
        ],
        None,
    );

    let output_matrix = Linear::new(runtime.new_matrix(Random, 768, 256), None, Identity);

    let mut transformer = net::transformer::encoder::InferenceTransformer::new(
        matrix_q,
        matrix_k,
        matrix_v,
        matrix_position,
        fcs,
        output_matrix,
        None,
        net::transformer::NormType::Layer,
    );

    // InferenceTransformer::forward needs at most five live seq × hidden
    // workspaces. The other shapes have one live temporary each. The final
    // seq × output buffer is owned by the caller and recycled below only after
    // it is no longer needed.
    runtime.reserve_buffers(1024 * 768, 5);
    runtime.reserve_buffers(1024 * 1024, 1);
    runtime.reserve_buffers(1024 * 3072, 1);
    runtime.reserve_buffers(1024 * 256, 1);
    runtime.sync();

    let mut average = std::time::Duration::ZERO;

    for _ in 0..100 {
        let time_now = std::time::Instant::now();
        let output = transformer.forward(&input, &mut runtime);
        runtime.sync();
        let time_duration = time_now.elapsed();
        average += time_duration;
        println!("Time taken: {:?}", time_duration);
        runtime.recycle_matrix(output);
    }
    println!("Average time taken: {:?}", average / 100);
}
