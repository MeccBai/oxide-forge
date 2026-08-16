use crate::cuda::{CudaRuntime, DeviceSpan, DeviceSpanMut};

use super::Vector;

impl Vector {
    pub fn to_host(&self, runtime: &CudaRuntime) -> Vec<f32> {
        self.buffer.to_host_vec(runtime.stream()).unwrap()
    }

    pub fn add(&mut self, value: f32, runtime: &CudaRuntime) {
        let len = self.buffer.len();
        let mut span = DeviceSpanMut::from_buffer(&mut self.buffer, 0, len);
        span.for_each(runtime, move |x| x + value);
    }

    pub fn scale(&mut self, value: f32, runtime: &CudaRuntime) {
        let len = self.buffer.len();
        let mut span = DeviceSpanMut::from_buffer(&mut self.buffer, 0, len);
        span.scale(value, runtime);
    }

    pub fn sum(&self, runtime: &CudaRuntime) -> f32 {
        DeviceSpan::from_buffer(&self.buffer, 0, self.buffer.len()).sum(runtime)
    }

    pub fn max(&self, runtime: &CudaRuntime) -> f32 {
        DeviceSpan::from_buffer(&self.buffer, 0, self.buffer.len()).max(runtime)
    }

    pub fn exp(&mut self, value: f32, runtime: &CudaRuntime) {
        let len = self.buffer.len();
        let mut span = DeviceSpanMut::from_buffer(&mut self.buffer, 0, len);
        span.for_each(runtime, move |x| (x - value).exp());
    }

    pub fn softmax(&mut self, runtime: &CudaRuntime) {
        let max = self.max(runtime);
        self.exp(max, runtime);
        let sum = self.sum(runtime);
        self.scale(1.0 / sum, runtime);
    }
}
