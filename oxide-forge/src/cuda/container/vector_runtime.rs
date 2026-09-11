use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use crate::cuda::{CudaRuntime, DEFAULT_BLOCK_SIZE, DeviceSpan, DeviceSpanMut, runtime::InitType};

use super::Vector;

impl CudaRuntime {
    pub fn vector_from_host(&self, values: &[f32]) -> Result<Vector, DriverError> {
        Ok(Vector {
            buffer: DeviceBuffer::from_host(self.stream(), values)?,
        })
    }

    /// Consumes a vector and returns its allocation to the runtime pool.
    pub fn recycle_vector(&mut self, vector: Vector) {
        self.recycle_buffer(vector.buffer);
    }

    pub(crate) fn create_vector(&self, buffer: DeviceBuffer<f32>) -> Vector {
        Vector { buffer }
    }

    pub fn new_vector(&mut self, init_type: InitType, size: usize) -> Vector {
        if init_type.is_zero() {
            return Vector {
                buffer: self.get_zerod_buffer(size),
            };
        }
        let mut buffer = self.get_uninit_buffer(size);
        let (config, elements_per_thread) =
            self.get_elementwise_launch_config(buffer.len(), DEFAULT_BLOCK_SIZE);
        let span = DeviceSpanMut::from_buffer(&mut buffer, 0, size);
        match init_type {
            InitType::Sequence => {
                let prepared = self.module().prepare_slice_set_seq(config).unwrap();
                self.module()
                    .slice_set_seq(
                        self.stream(),
                        &prepared,
                        span.descriptor(),
                        elements_per_thread,
                        true,
                        0.0,
                        1.0,
                    )
                    .unwrap();
                Vector { buffer }
            }
            InitType::Reserve => {
                let prepared = self.module().prepare_slice_set_seq(config).unwrap();
                self.module()
                    .slice_set_seq(
                        self.stream(),
                        &prepared,
                        span.descriptor(),
                        elements_per_thread,
                        false,
                        0.0,
                        1.0,
                    )
                    .unwrap();
                Vector { buffer }
            }
            InitType::Random => {
                let seed = rand::random();
                let prepared = self.module().prepare_slice_set_random(config).unwrap();
                self.module()
                    .slice_set_random(
                        self.stream(),
                        &prepared,
                        span.descriptor(),
                        elements_per_thread,
                        seed,
                    )
                    .unwrap();
                Vector { buffer }
            }
            InitType::Zero => Vector { buffer },
        }
    }

    pub fn clone_vector(&mut self, vec: &Vector) -> Vector {
        Vector {
            buffer: self.clone_buffer(&vec.buffer),
        }
    }

    pub(crate) fn clone_vector_on(&mut self, vec: &Vector, stream: &CudaStream) -> Vector {
        let mut buffer = self.get_uninit_buffer(vec.buffer.len());
        stream.join(self.stream()).unwrap();
        buffer.copy_from_device_async(&vec.buffer, stream).unwrap();
        Vector { buffer }
    }

    pub fn vector_add(&mut self, vec1: &Vector, vec2: &Vector) -> Vector {
        self.vector_binary(vec1, vec2, move |lhs, rhs| lhs + rhs)
    }

    pub fn vector_sub(&mut self, vec1: &Vector, vec2: &Vector) -> Vector {
        self.vector_binary(vec1, vec2, move |lhs, rhs| lhs - rhs)
    }

    pub fn vector_mul(&mut self, vec1: &Vector, vec2: &Vector) -> Vector {
        self.vector_binary(vec1, vec2, move |lhs, rhs| lhs * rhs)
    }

    pub fn vector_div(&mut self, vec1: &Vector, vec2: &Vector) -> Vector {
        self.vector_binary(vec1, vec2, move |lhs, rhs| lhs / rhs)
    }

    pub fn vector_binary<F>(&mut self, vec1: &Vector, vec2: &Vector, f: F) -> Vector
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

        self.module()
            .slice_binary(
                self.stream(),
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
