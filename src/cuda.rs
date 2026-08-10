use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D, PreparedLaunch};
use cuda_device::{
    DisjointSlice, SharedArray, device, gpu_printf, kernel, launch_bounds, launch_contract, shared,
    thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

const DEFAULT_BLOCK_SIZE: usize = 1024;

mod device;
mod span;
pub use span::DeviceSpanMut;
#[cuda_module]
mod kernels {
    const DEFAULT_BLOCK_SIZE_U32: u32 = 1024;

    use super::*;

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1)]
    pub fn vec_set(mut buffer: DisjointSlice<f32>, value: f32) {
        let idx = thread::index_1d();
        if let Some(elem) = buffer.get_mut(idx) {
            *elem = value;
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn vec_set_seq(mut buffer: DisjointSlice<f32>, dir: bool) {
        let idx = thread::index_1d();
        let max = buffer.len();
        let val = if dir {
            idx.get() as f32
        } else {
            (max - idx.get()) as f32
        };
        if let Some(elem) = buffer.get_mut(idx) {
            *elem = val;
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn vector_add(buffer1: &[f32], buffer2: &[f32], mut result: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let index = idx.get();
        if let Some(elem) = result.get_mut(idx) {
            *elem = buffer1[index] + buffer2[index];
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn vector_sum(buffer: &[f32], mut result: DisjointSlice<f32>) {
        let value = device::vector_add_device(buffer);
        if thread::threadIdx_x() == 0 {
            let block_id = thread::blockIdx_x() as usize;
            if block_id < result.len() {
                unsafe {
                    *result.get_unchecked_mut(block_id) = value;
                }
            }
        }
    }

    #[kernel]
    #[launch_bounds(1024)]
    #[launch_contract(domain = 1)]
    pub fn vector_for_each<F>(mut buffer: DisjointSlice<f32>, f: F)
    where
        F: Fn(f32) -> f32 + Copy,
    {
        let idx = thread::index_1d();

        if let Some(element) = buffer.get_mut(idx) {
            *element = f(*element);
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub unsafe fn span_for_each<F>(span: span::DeviceSpanDescriptor<f32>, f: F)
    where
        F: Fn(f32) -> f32 + Copy,
    {
        let index = thread::index_1d().get();
        if index < span.len {
            let element = unsafe { span.ptr.add(index * span.stride) };
            unsafe {
                element.write(f(element.read()));
            }
        }
    }

    #[kernel]
    #[launch_bounds(1024)]
    #[launch_contract(domain = 1)]
    pub fn vector_set_random(mut buffer: DisjointSlice<f32>, seed: u32) {
        let idx = thread::index_1d();
        let val = idx.get() as u32;
        if let Some(elem) = buffer.get_mut(idx) {
            let rand = device::random(seed + val as u32);
            *elem = (rand as f32) / (u32::MAX as f32);
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn vector_max(buffer: &[f32], mut result: DisjointSlice<f32>) {
        let value = device::vector_max_device(buffer);
        if thread::threadIdx_x() == 0 {
            let block_id = thread::blockIdx_x() as usize;
            if block_id < result.len() {
                unsafe {
                    *result.get_unchecked_mut(block_id) = value;
                }
            }
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn copy_slice(buffer: &[f32], mut result: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let index = idx.get();
        if let Some(elem) = result.get_mut(idx) {
            *elem = buffer[index];
        }
    }
}

mod matrix;
mod vector;
pub enum InitType {
    Sequence,
    Reserve,
    Random,
    Zero,
}

impl InitType {
    pub fn is_zero(&self) -> bool {
        match self {
            Self::Sequence => false,
            Self::Reserve => false,
            Self::Random => false,
            Self::Zero => true,
        }
    }
}

pub struct CudaRuntime {
    module: kernels::LoadedModule,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    temp_stream: Vec<Arc<CudaStream>>,
}

impl CudaRuntime {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let temp_stream = Vec::new();
        let module = unsafe { kernels::load(&ctx)? };

        Ok(Self {
            module,
            ctx,
            stream,
            temp_stream,
        })
    }

    pub fn get_uninit_buffer(&self, size: usize) -> DeviceBuffer<f32> {
        let buffer = unsafe { DeviceBuffer::<f32>::uninitialized_async(&self.stream, size) };
        buffer.unwrap()
    }

    pub fn get_zerod_buffer(&self, size: usize) -> DeviceBuffer<f32> {
        let buffer = DeviceBuffer::<f32>::zeroed(&self.stream, size);
        buffer.unwrap()
    }

    pub fn sync(&self) {
        self.stream.synchronize().unwrap();
    }
}
