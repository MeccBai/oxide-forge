use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D, PreparedLaunch};
use cuda_device::{
    DisjointSlice, SharedArray, device, gpu_printf, kernel, launch_bounds, launch_contract, shared,
    thread, warp,
};
use cuda_host::cuda_module;

#[device]
pub fn vector_add_device(buffer: &[f32]) -> f32 {
    let idx = thread::index_1d();
    let index = idx.get();
    static mut SHARED: SharedArray<f32, 32> = SharedArray::UNINIT;
    let mut value = if index < buffer.len() {
        buffer[index]
    } else {
        0.0
    };
    for delta in [1, 2, 4, 8, 16] {
        value += warp::shuffle_down_f32(value, delta) as f32;
    }
    if (thread::threadIdx_x() % 32) == 0 {
        unsafe {
            SHARED[thread::threadIdx_x() as usize / 32] = value;
        }
    }
    thread::sync_threads();
    if thread::threadIdx_x() < 32 {
        value = unsafe { SHARED[thread::threadIdx_x() as usize % 32] };
        for delta in [1, 2, 4, 8, 16] {
            value += warp::shuffle_down_f32(value, delta) as f32;
        }
    }
    return value;
}

#[device]
pub fn vector_max_device(buffer: &[f32]) -> f32 {
    let idx = thread::index_1d();
    let index = idx.get();
    static mut SHARED: SharedArray<f32, 32> = SharedArray::UNINIT;
    let mut value = if index < buffer.len() {
        buffer[index]
    } else {
        f32::MIN
    };
    for delta in [1, 2, 4, 8, 16] {
        value = value.max(warp::shuffle_down_f32(value, delta) as f32);
    }
    if (thread::threadIdx_x() % 32) == 0 {
        unsafe {
            SHARED[thread::threadIdx_x() as usize / 32] = value;
        }
    }
    thread::sync_threads();
    if thread::threadIdx_x() < 32 {
        value = unsafe { SHARED[thread::threadIdx_x() as usize % 32] };
        for delta in [1, 2, 4, 8, 16] {
            value = value.max(warp::shuffle_down_f32(value, delta) as f32);
        }
    }
    return value;
}

#[device]
pub fn random(mut x: u32) -> u32 {
    x += x << 13;
    x -= x >> 17;
    x *= x << 5;
    x
}
