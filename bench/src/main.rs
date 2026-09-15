use oxide_forge::cuda::{
    CudaRuntime,
    InitType::{Random, Reserve},
};
use std::time::Instant;

const SIZE: usize = 512;
const WARMUP_ITERATIONS: usize = 10;
const BENCHMARK_ITERATIONS: usize = 100;

fn main() {
    let mut runtime = CudaRuntime::new().unwrap();
    let mat1 = runtime.new_matrix(Random, SIZE, SIZE);
    let mat2 = runtime.new_matrix(Random, SIZE, SIZE);
    let mut result = runtime.new_matrix(Reserve, SIZE, SIZE);

    for _ in 0..WARMUP_ITERATIONS {
        runtime.matrix_multiply_into(&mat1, &mat2, &mut result);
    }
    runtime.sync();

    let start = Instant::now();
    for _ in 0..BENCHMARK_ITERATIONS {
        runtime.matrix_multiply_into(&mat1, &mat2, &mut result);
    }
    runtime.sync();
    let elapsed = start.elapsed();
    let average = elapsed / u32::try_from(BENCHMARK_ITERATIONS).unwrap();

    println!("matrix:            {SIZE}x{SIZE} @ {SIZE}x{SIZE}");
    println!("iterations:        {BENCHMARK_ITERATIONS}");
    println!("total:             {elapsed:?}");
    println!("GEMM average:      {average:?}");
}
