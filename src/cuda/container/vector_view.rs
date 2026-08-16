use cuda_core::DeviceBuffer;

use crate::cuda::CudaRuntime;

use super::{Vector, VectorView};
use crate::cuda::span::{DeviceSpan, DeviceSpanMut};

impl Vector {
    pub fn as_span(&self) -> DeviceSpan<'_, f32> {
        DeviceSpan::from_buffer(&self.buffer, 0, self.buffer.len())
    }

    pub fn span(&self, offset: usize, len: usize) -> DeviceSpan<'_, f32> {
        DeviceSpan::from_buffer(&self.buffer, offset, len)
    }

    pub fn new(buffer: DeviceBuffer<f32>) -> Self {
        Vector { buffer }
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

impl<'a> VectorView<'a> {
    pub fn new(span: DeviceSpanMut<'a, f32>) -> Self {
        VectorView { span }
    }

    pub fn len(&self) -> usize {
        self.span.len()
    }

    pub fn add(&mut self, value: f32, runtime: &CudaRuntime) {
        self.span.for_each(runtime, move |x| x + value);
    }

    pub fn scale(&mut self, value: f32, runtime: &CudaRuntime) {
        self.span.for_each(runtime, move |x| x * value);
    }

    pub fn for_each<F>(&mut self, runtime: &CudaRuntime, f: F)
    where
        F: Fn(f32) -> f32 + Copy,
    {
        self.span.for_each(runtime, f);
    }

    pub fn sum(&self, runtime: &CudaRuntime) -> f32 {
        self.span.sum(runtime)
    }

    pub fn map_sum<F>(&self, runtime: &CudaRuntime, f: F) -> f32
    where
        F: Fn(f32) -> f32 + Copy,
    {
        self.span.map_sum(runtime, f)
    }

    pub fn softmax(&mut self, runtime: &CudaRuntime) {
        let max = self.span.max(runtime);
        self.span.for_each(runtime, move |x| (x - max).exp());
        let sum = self.span.sum(runtime);
        self.span.scale(1.0 / sum, runtime);
    }
}
