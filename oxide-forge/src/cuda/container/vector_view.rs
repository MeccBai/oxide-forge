use crate::cuda::CudaRuntime;
use cuda_core::CudaStream;

use super::VectorView;
use crate::cuda::span::DeviceSpanMut;

impl<'a> VectorView<'a> {
    pub(crate) fn new(span: DeviceSpanMut<'a, f32>) -> Self {
        VectorView { span }
    }

    pub fn len(&self) -> usize {
        self.span.len()
    }

    pub fn add_scalar(&mut self, value: f32, runtime: &CudaRuntime, stream: Option<&CudaStream>) {
        self.span.for_each(runtime, move |x| x + value, stream);
    }

    pub fn scale(&mut self, value: f32, runtime: &CudaRuntime, stream: Option<&CudaStream>) {
        self.span.for_each(runtime, move |x| x * value, stream);
    }

    pub fn for_each<F>(&mut self, runtime: &CudaRuntime, f: F, stream: Option<&CudaStream>)
    where
        F: Fn(f32) -> f32 + Copy,
    {
        self.span.for_each(runtime, f, stream);
    }

    pub fn sum(&self, runtime: &mut CudaRuntime, stream: Option<&CudaStream>) -> f32 {
        self.map_reduce(
            runtime,
            0.0,
            move |value| value,
            move |lhs, rhs| lhs + rhs,
            stream,
        )
    }

    pub fn max(&self, runtime: &mut CudaRuntime, stream: Option<&CudaStream>) -> f32 {
        self.map_reduce(
            runtime,
            f32::NEG_INFINITY,
            move |value| value,
            move |lhs, rhs| lhs.max(rhs),
            stream,
        )
    }

    pub fn map_sum<F>(&self, runtime: &mut CudaRuntime, map: F, stream: Option<&CudaStream>) -> f32
    where
        F: Fn(f32) -> f32 + Copy,
    {
        self.map_reduce(runtime, 0.0, map, move |lhs, rhs| lhs + rhs, stream)
    }

    pub fn map_reduce<FM, FR>(
        &self,
        runtime: &mut CudaRuntime,
        identity: f32,
        map: FM,
        reduce: FR,
        stream: Option<&CudaStream>,
    ) -> f32
    where
        FM: Fn(f32) -> f32 + Copy,
        FR: Fn(f32, f32) -> f32 + Copy,
    {
        self.span.map_reduce(runtime, identity, map, reduce, stream)
    }

    pub fn zip_map_reduce<FM, FR>(
        &self,
        rhs: &VectorView<'_>,
        runtime: &mut CudaRuntime,
        identity: f32,
        map: FM,
        reduce: FR,
        stream: Option<&CudaStream>,
    ) -> f32
    where
        FM: Fn(f32, f32) -> f32 + Copy,
        FR: Fn(f32, f32) -> f32 + Copy,
    {
        self.span
            .zip_map_reduce(&rhs.span, runtime, identity, map, reduce, stream)
    }

    pub fn softmax(&mut self, runtime: &mut CudaRuntime, stream: Option<&CudaStream>) {
        let max = self.span.max(runtime, stream);
        self.span
            .for_each(runtime, move |x| (x - max).exp(), stream);
        let sum = self.span.sum(runtime, stream);
        self.span.scale(1.0 / sum, runtime, stream);
    }
}
