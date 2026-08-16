mod cuda;
mod net;

use crate::net::linear::Activation::{Gelu, Identity};
use crate::net::linear::Linear;
use crate::net::mlp::Mlp;

/*
fn main() {
    let runtime = cuda::CudaRuntime::new().unwrap();

    // seq × hidden
    let input = runtime.new_matrix(cuda::InitType::Random, 256, 768);

    // hidden × hidden
    let matrix_q = runtime.new_matrix(cuda::InitType::Random, 768, 768);
    let matrix_k = runtime.new_matrix(cuda::InitType::Random, 768, 768);
    let matrix_v = runtime.new_matrix(cuda::InitType::Random, 768, 768);

    // seq × hidden
    let matrix_position = runtime.new_matrix(cuda::InitType::Random, 256, 768);

    let input_with_position = runtime.matrix_add(&input, &matrix_position);

    // seq × hidden
    let qs = runtime.matrix_multiply(&input_with_position, &matrix_q);

    let ks = runtime.matrix_multiply(&input_with_position, &matrix_k);

    let vs = runtime.matrix_multiply(&input_with_position, &matrix_v);

    // K^T:
    // hidden × seq
    let kst = runtime.matrix_transpose(&ks);

    // seq × seq
    let mut scores = runtime.matrix_multiply(&qs, &kst);

    // sqrt(hidden)
    let scale = 1.0 / (qs.cols() as f32).sqrt();

    scores.scale(scale, &runtime);

    // 每个token对其他token归一化
    scores.softmax_rows(&runtime);

    // seq×seq  *  seq×hidden
    //
    // = seq×hidden
    let attention = runtime.matrix_multiply(&scores, &vs);

    let mut net_input = runtime.matrix_add(&attention, &input_with_position);

    net_input.layer_norm(&runtime);

    println!("[{}:{}]", net_input.rows(), net_input.cols());

    let fc1 = runtime.new_matrix(cuda::InitType::Random, 768, 3072);

    let fc2 = runtime.new_matrix(cuda::InitType::Random, 3072, 768);

    let mut hidden = runtime.matrix_multiply(&net_input, &fc1);

    let gelu = move |x: f32| 0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh());

    hidden.for_each(&runtime, gelu);

    let mlp_output = runtime.matrix_multiply(&hidden, &fc2);
    let mut output = runtime.matrix_add(&net_input, &mlp_output);

    output.layer_norm(&runtime);
}
*/

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

    let fcs = Mlp::new(
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

    let transformer = net::transformer::Transformer::new(
        matrix_q,
        matrix_k,
        matrix_v,
        matrix_position,
        fcs,
        output_matrix,
    );

    let output = transformer.forward(&input, &runtime);

    println!("[{}:{}]", output.rows(), output.cols());
}
