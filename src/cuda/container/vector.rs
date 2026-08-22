use crate::cuda::{BinaryOp, CudaRuntime, DEFAULT_BLOCK_SIZE, DeviceSpan, DeviceSpanMut};
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
        DeviceSpan::from_buffer(&self.buffer, 0, self.buffer.len()).sum(runtime)
    }

    pub fn max(&self, runtime: &mut CudaRuntime) -> f32 {
        DeviceSpan::from_buffer(&self.buffer, 0, self.buffer.len()).max(runtime)
    }

    pub fn exp_shifted(&mut self, offset: f32, runtime: &CudaRuntime) {
        let len = self.buffer.len();
        let mut span = DeviceSpanMut::from_buffer(&mut self.buffer, 0, len);
        span.for_each(runtime, move |x| (x - offset).exp());
    }

    pub fn softmax(&mut self, runtime: &mut CudaRuntime) {
        let max = self.max(runtime);
        self.exp_shifted(max, runtime);
        let sum = self.sum(runtime);
        self.scale(1.0 / sum, runtime);
    }

    pub fn binary_assign(&mut self, rhs: &Vector, op: BinaryOp, runtime: &CudaRuntime) {
        self.binary_assign_on(rhs, op, runtime, runtime.stream());
    }

    pub(crate) fn binary_assign_on(
        &mut self,
        rhs: &Vector,
        op: BinaryOp,
        runtime: &CudaRuntime,
        stream: &CudaStream,
    ) {
        let len = self.buffer.len();
        assert_eq!(len, rhs.buffer.len());
        let span = DeviceSpanMut::from_buffer(&mut self.buffer, 0, len);
        let rhs_span = DeviceSpan::from_buffer(&rhs.buffer, 0, len);

        let config = runtime.get_launch_config(len, DEFAULT_BLOCK_SIZE);
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
                op,
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
        assert_eq!(self.len(), rhs.len());
        let product = runtime.vector_mul(self, rhs);
        let result = product.sum(runtime);
        runtime.recycle_vector(product);
        result
    }
}
