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
        self.map_reduce(runtime, 0.0, move |value| value, move |lhs, rhs| lhs + rhs)
    }

    pub fn max(&self, runtime: &mut CudaRuntime) -> f32 {
        self.map_reduce(
            runtime,
            f32::NEG_INFINITY,
            move |value| value,
            move |lhs, rhs| lhs.max(rhs),
        )
    }

    pub fn map_sum<F>(&self, runtime: &mut CudaRuntime, map: F) -> f32
    where
        F: Fn(f32) -> f32 + Copy,
    {
        self.map_reduce(runtime, 0.0, map, move |lhs, rhs| lhs + rhs)
    }

    pub fn map_reduce<FM, FR>(
        &self,
        runtime: &mut CudaRuntime,
        identity: f32,
        map: FM,
        reduce: FR,
    ) -> f32
    where
        FM: Fn(f32) -> f32 + Copy,
        FR: Fn(f32, f32) -> f32 + Copy,
    {
        self.span.map_reduce(runtime, identity, map, reduce)
    }

    pub fn softmax(&mut self, runtime: &mut CudaRuntime) {
        let max = self.span.max(runtime);
        self.span.for_each(runtime, move |x| (x - max).exp());
        let sum = self.span.sum(runtime);
        self.span.scale(1.0 / sum, runtime);
    }
}
