use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use crate::cuda::{CudaRuntime, DEFAULT_BLOCK_SIZE, DeviceSpan, DeviceSpanMut, runtime::InitType};

use super::Vector;

impl CudaRuntime {
    pub fn vector_from_host(
        &self,
        values: &[f32],
        stream: Option<&CudaStream>,
    ) -> Result<Vector, DriverError> {
        let stream = self.execution_stream(stream);
        Ok(Vector {
            buffer: DeviceBuffer::from_host(stream, values)?,
        })
    }

    /// Consumes a vector and returns its allocation to the runtime pool.
    pub fn recycle_vector(&mut self, vector: Vector) {
        self.recycle_buffer(vector.buffer);
    }

    pub(crate) fn create_vector(&self, buffer: DeviceBuffer<f32>) -> Vector {
        Vector { buffer }
    }

    pub fn new_vector(
        &mut self,
        init_type: InitType,
        size: usize,
        stream: Option<&CudaStream>,
    ) -> Vector {
        let mut buffer = self.get_uninit_buffer(size);
        let mut span = DeviceSpanMut::from_buffer(&mut buffer, 0, size);
        init_type.initialize(&mut span, self, stream);
        Vector { buffer }
    }

    pub fn clone_vector(&mut self, vec: &Vector, stream: Option<&CudaStream>) -> Vector {
        let mut buffer = self.get_uninit_buffer(vec.buffer.len());
        let stream = self.execution_stream(stream);
        buffer.copy_from_device_async(&vec.buffer, stream).unwrap();
        Vector { buffer }
    }

    pub fn vector_add(
        &mut self,
        vec1: &Vector,
        vec2: &Vector,
        stream: Option<&CudaStream>,
    ) -> Vector {
        self.vector_binary(vec1, vec2, move |lhs, rhs| lhs + rhs, stream)
    }

    pub fn vector_sub(
        &mut self,
        vec1: &Vector,
        vec2: &Vector,
        stream: Option<&CudaStream>,
    ) -> Vector {
        self.vector_binary(vec1, vec2, move |lhs, rhs| lhs - rhs, stream)
    }

    pub fn vector_mul(
        &mut self,
        vec1: &Vector,
        vec2: &Vector,
        stream: Option<&CudaStream>,
    ) -> Vector {
        self.vector_binary(vec1, vec2, move |lhs, rhs| lhs * rhs, stream)
    }

    pub fn vector_div(
        &mut self,
        vec1: &Vector,
        vec2: &Vector,
        stream: Option<&CudaStream>,
    ) -> Vector {
        self.vector_binary(vec1, vec2, move |lhs, rhs| lhs / rhs, stream)
    }

    pub fn vector_binary<F>(
        &mut self,
        vec1: &Vector,
        vec2: &Vector,
        f: F,
        stream: Option<&CudaStream>,
    ) -> Vector
    where
        F: Fn(f32, f32) -> f32 + Copy,
    {
        assert_eq!(vec1.buffer.len(), vec2.buffer.len());
        let mut result_buffer = self.get_uninit_buffer(vec1.buffer.len());

        let (config, elements_per_thread) =
            self.get_elementwise_launch_config(vec1.buffer.len(), DEFAULT_BLOCK_SIZE);

        let prepared = self.module().prepare_slice_binary(config).unwrap();
        let lhs = DeviceSpan::from_buffer(&vec1.buffer, 0, vec1.buffer.len());
        let rhs = DeviceSpan::from_buffer(&vec2.buffer, 0, vec2.buffer.len());
        let result_len = result_buffer.len();
        let output = DeviceSpanMut::from_buffer(&mut result_buffer, 0, result_len);
        let stream = self.execution_stream(stream);

        self.module()
            .slice_binary(
                stream,
                &prepared,
                lhs.descriptor(),
                rhs.descriptor(),
                output.descriptor(),
                elements_per_thread,
                f,
            )
            .unwrap();
        Vector {
            buffer: result_buffer,
        }
    }
}
