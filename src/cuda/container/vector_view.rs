use crate::cuda::CudaRuntime;

use super::VectorView;
use crate::cuda::span::DeviceSpanMut;

impl<'a> VectorView<'a> {
    pub(crate) fn new(span: DeviceSpanMut<'a, f32>) -> Self {
        VectorView { span }
    }

    pub fn len(&self) -> usize {
        self.span.len()
    }

    pub fn add_scalar(&mut self, value: f32, runtime: &CudaRuntime) {
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

    pub fn sum(&self, runtime: &mut CudaRuntime) -> f32 {
        self.span.sum(runtime)
    }

    pub fn map_sum<F>(&self, runtime: &mut CudaRuntime, f: F) -> f32
    where
        F: Fn(f32) -> f32 + Copy,
    {
        self.span.map_sum(runtime, f)
    }

    pub fn softmax(&mut self, runtime: &mut CudaRuntime) {
        let max = self.span.max(runtime);
        self.span.for_each(runtime, move |x| (x - max).exp());
        let sum = self.span.sum(runtime);
        self.span.scale(1.0 / sum, runtime);
    }
}
