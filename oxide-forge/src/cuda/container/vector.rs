use crate::cuda::{CudaRuntime, DEFAULT_BLOCK_SIZE, DeviceSpan, DeviceSpanMut, span};
use cuda_core::CudaStream;

use super::Vector;

impl Vector {
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn as_span(&self) -> DeviceSpan<'_, f32> {
        DeviceSpan::from_buffer(&self.buffer, 0, self.buffer.len())
    }

    pub fn span(&self, offset: usize, len: usize) -> DeviceSpan<'_, f32> {
        DeviceSpan::from_buffer(&self.buffer, offset, len)
    }

    pub fn to_host(&self, runtime: &CudaRuntime) -> Vec<f32> {
        self.buffer.to_host_vec(runtime.stream()).unwrap()
    }

    pub fn add_scalar(&mut self, value: f32, runtime: &CudaRuntime) {
        let len = self.buffer.len();
        let mut span = DeviceSpanMut::from_buffer(&mut self.buffer, 0, len);
        span.for_each(runtime, move |x| x + value);
    }

    pub fn scale(&mut self, value: f32, runtime: &CudaRuntime) {
        self.scale_on(value, runtime, runtime.stream());
    }

    pub(crate) fn scale_on(&mut self, value: f32, runtime: &CudaRuntime, stream: &CudaStream) {
        let len = self.buffer.len();
        let mut span = DeviceSpanMut::from_buffer(&mut self.buffer, 0, len);
        span.for_each_on(runtime, stream, move |x| x * value);
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
        self.as_span().map_reduce(runtime, identity, map, reduce)
    }

    pub fn zip_map_reduce<FM, FR>(
        &self,
        rhs: &Vector,
        runtime: &mut CudaRuntime,
        identity: f32,
        map: FM,
        reduce: FR,
    ) -> f32
    where
        FM: Fn(f32, f32) -> f32 + Copy,
        FR: Fn(f32, f32) -> f32 + Copy,
    {
        self.as_span()
            .zip_map_reduce(&rhs.as_span(), runtime, identity, map, reduce)
    }

    pub fn exp_shifted(&mut self, offset: f32, runtime: &CudaRuntime) {
        let len = self.buffer.len();
        let mut span = DeviceSpanMut::from_buffer(&mut self.buffer, 0, len);
        span.for_each(runtime, move |x| (x - offset).exp());
    }

    /// Applies the numerically stable logistic sigmoid in place.
    pub fn sigmoid(&mut self, runtime: &CudaRuntime) {
        self.for_each(runtime, crate::cuda::sigmoid_f32);
    }

    pub fn softmax(&mut self, runtime: &mut CudaRuntime) {
        let max = self.max(runtime);
        self.exp_shifted(max, runtime);
        let sum = self.sum(runtime);
        self.scale(1.0 / sum, runtime);
    }

    pub fn binary_assign<F>(&mut self, rhs: &Vector, runtime: &CudaRuntime, f: F)
    where
        F: Fn(f32, f32) -> f32 + Copy,
    {
        self.binary_assign_on(rhs, f, runtime, runtime.stream());
    }

    pub(crate) fn binary_assign_on<F>(
        &mut self,
        rhs: &Vector,
        f: F,
        runtime: &CudaRuntime,
        stream: &CudaStream,
    ) where
        F: Fn(f32, f32) -> f32 + Copy,
    {
        let len = self.buffer.len();
        assert_eq!(len, rhs.buffer.len());
        let span = DeviceSpanMut::from_buffer(&mut self.buffer, 0, len);
        let rhs_span = DeviceSpan::from_buffer(&rhs.buffer, 0, len);

        let (config, elements_per_thread) =
            runtime.get_elementwise_launch_config(len, DEFAULT_BLOCK_SIZE);
        let prepared = runtime
            .module()
            .prepare_slice_binary_assign(config)
            .unwrap();
        runtime
            .module()
            .slice_binary_assign(
                stream,
                &prepared,
                span.descriptor(),
                rhs_span.descriptor(),
                elements_per_thread,
                f,
            )
            .unwrap();
    }

    pub fn for_each<F>(&mut self, runtime: &CudaRuntime, f: F)
    where
        F: Fn(f32) -> f32 + Copy,
    {
        self.for_each_on(runtime, runtime.stream(), f);
    }

    pub(crate) fn for_each_on<F>(&mut self, runtime: &CudaRuntime, stream: &CudaStream, f: F)
    where
        F: Fn(f32) -> f32 + Copy,
    {
        let len = self.buffer.len();
        let mut span = DeviceSpanMut::from_buffer(&mut self.buffer, 0, len);
        span.for_each_on(runtime, stream, f);
    }

    pub fn dot(&self, rhs: &Vector, runtime: &mut CudaRuntime) -> f32 {
        self.zip_map_reduce(
            rhs,
            runtime,
            0.0,
            move |lhs, rhs| lhs * rhs,
            move |lhs, rhs| lhs + rhs,
        )
    }

    pub fn equals(&self, rhs: &Vector, runtime: &mut CudaRuntime) -> bool {
        assert_eq!(self.len(), rhs.len());
        let config = cuda_core::LaunchConfig1D::new(1, DEFAULT_BLOCK_SIZE as u32, 0);
        let elements_per_thread = self.len().div_ceil(DEFAULT_BLOCK_SIZE);
        let prepared = runtime.module().prepare_compare_vectors(config).unwrap();
        let mut result = runtime.get_u32_uninit_buffer(1);
        let view = span::DeviceSpanMut::from_buffer(&mut result, 0, 1);
        runtime
            .module()
            .compare_vectors(
                runtime.stream(),
                &prepared,
                self.as_span().descriptor(),
                rhs.as_span().descriptor(),
                elements_per_thread,
                view.descriptor(),
            )
            .unwrap();

        let host_result = result.to_host_vec(runtime.stream()).unwrap();
        host_result[0] != 0
    }
}
