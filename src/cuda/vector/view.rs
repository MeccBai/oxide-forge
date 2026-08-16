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

    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    pub fn new(buffer: DeviceBuffer<f32>) -> Self {
        Vector { buffer }
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
}
