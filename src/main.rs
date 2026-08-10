use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

mod cuda;

fn main() {
    let mut cuda_runtime = cuda::CudaRuntime::new().unwrap();

    let vec1 = cuda_runtime.new_vector(cuda::InitType::Random, 2048);

    let vec1_host = vec1.to_host(&cuda_runtime);

    for i in 0..10 {
        println!("vec1[{}] = {}", i, vec1_host[i]);
    }
}
