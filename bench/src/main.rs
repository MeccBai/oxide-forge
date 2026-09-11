use oxide_forge::cuda::{
    CudaRuntime,
    InitType::{Random, Reserve},
};
use std::time::{Duration, Instant};

const SIZE: usize = 512;
const WARMUP_ITERATIONS: usize = 10;
const BENCHMARK_ITERATIONS: usize = 1_000;

fn average(duration: Duration, iterations: usize) -> Duration {
    duration / u32::try_from(iterations).unwrap()
}

fn main() {
    let mut runtime = CudaRuntime::new().unwrap();
    let mat1 = runtime.new_matrix(Random, SIZE, SIZE);
    let mat2 = runtime.new_matrix(Random, SIZE, SIZE);
    let mut baseline = runtime.new_matrix(Reserve, SIZE, SIZE);
    let mut async_tensor = runtime.new_matrix(Reserve, SIZE, SIZE);

    runtime.matrix_multiply_into(&mat1, &mat2, &mut baseline);
    runtime.matrix_multiply_at_into(&mat1, &mat2, &mut async_tensor);
    runtime.sync();

    let baseline_host = baseline.to_host(&runtime);
    let async_tensor_host = async_tensor.to_host(&runtime);
    let (max_absolute_error, max_relative_error) = baseline_host
        .iter()
        .zip(&async_tensor_host)
        .fold((0.0f32, 0.0f32), |(max_abs, max_rel), (&lhs, &rhs)| {
            let absolute = (lhs - rhs).abs();
            let relative = absolute / lhs.abs().max(1.0e-6);
            (max_abs.max(absolute), max_rel.max(relative))
        });

    println!("matrix: {SIZE}x{SIZE} @ {SIZE}x{SIZE}");
    println!("max absolute error: {max_absolute_error:.6}");
    println!("max relative error: {max_relative_error:.6}");

    for _ in 0..WARMUP_ITERATIONS {
        runtime.matrix_multiply_into(&mat1, &mat2, &mut baseline);
        runtime.matrix_multiply_at_into(&mat1, &mat2, &mut async_tensor);
    }
    runtime.sync();

    let start = Instant::now();
    for _ in 0..BENCHMARK_ITERATIONS {
        runtime.matrix_multiply_into(&mat1, &mat2, &mut baseline);
    }
    runtime.sync();
    let baseline_elapsed = start.elapsed();

    let start = Instant::now();
    for _ in 0..BENCHMARK_ITERATIONS {
        runtime.matrix_multiply_at_into(&mat1, &mat2, &mut async_tensor);
    }
    runtime.sync();
    let async_tensor_elapsed = start.elapsed();

    let baseline_average = average(baseline_elapsed, BENCHMARK_ITERATIONS);
    let async_tensor_average = average(async_tensor_elapsed, BENCHMARK_ITERATIONS);
    println!("existing GEMM:     {baseline_average:?} per iteration");
    println!("async tensor GEMM: {async_tensor_average:?} per iteration");
    println!(
        "speedup: {:.3}x",
        baseline_elapsed.as_secs_f64() / async_tensor_elapsed.as_secs_f64()
    );
}
