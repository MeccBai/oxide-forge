const DEFAULT_BLOCK_SIZE: usize = 1024;


#[inline(always)]
pub(crate) fn sigmoid_f32(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}

pub mod container;
mod device;
pub mod runtime;
//mod tensor;

mod span;

pub use runtime::CudaRuntime;
pub use runtime::InitType;
pub(crate) use span::{DeviceSpan, DeviceSpanMut};

pub(in crate::cuda) use device::kernels;
