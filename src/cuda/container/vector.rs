use cuda_core::DeviceBuffer;

use crate::cuda::{CudaRuntime, DEFAULT_BLOCK_SIZE, DeviceSpan, DeviceSpanMut, runtime::InitType};

use super::Vector;

impl CudaRuntime {
    pub(super) fn create_vector(&self, buffer: DeviceBuffer<f32>) -> Vector {
        Vector { buffer }
    }

    pub fn new_vector(&self, init_type: InitType, size: usize) -> Vector {
        if init_type.is_zero() {
            return Vector {
                buffer: self.get_zerod_buffer(size),
            };
        }
        let mut buffer = self.get_uninit_buffer(size);
        let config = self.get_launch_config(buffer.len(), DEFAULT_BLOCK_SIZE);
        match init_type {
            InitType::Sequence => {
                let prepared = self.module().prepare_slice_set_seq(config).unwrap();
                let span = DeviceSpanMut::from_buffer(&mut buffer, 0, size);
                self.module()
                    .slice_set_seq(self.stream(), &prepared, span.descriptor(), true)
                    .unwrap();
                self.sync();
                Vector { buffer }
            }
            InitType::Reserve => {
                let prepared = self.module().prepare_slice_set_seq(config).unwrap();
                let span = DeviceSpanMut::from_buffer(&mut buffer, 0, size);
                self.module()
                    .slice_set_seq(self.stream(), &prepared, span.descriptor(), false)
                    .unwrap();
                self.sync();
                Vector { buffer }
            }
            InitType::Random => {
                let seed = rand::random();
                let prepared = self.module().prepare_slice_set_random(config).unwrap();
                let span = DeviceSpanMut::from_buffer(&mut buffer, 0, size);
                self.module()
                    .slice_set_random(self.stream(), &prepared, span.descriptor(), seed)
                    .unwrap();
                self.sync();
                Vector { buffer }
            }
            InitType::Zero => Vector { buffer },
        }
    }

    pub fn clone_vector(&self, vec: &Vector) -> Vector {
        Vector {
            buffer: self.clone_buffer(&vec.buffer),
        }
    }

    pub fn vector_add(&self, vec1: &Vector, vec2: &Vector) -> Vector {
        assert_eq!(vec1.buffer.len(), vec2.buffer.len());
        let mut result_buffer =
            DeviceBuffer::<f32>::zeroed(&self.stream(), vec1.buffer.len()).unwrap();

        let config = self.get_launch_config(vec1.buffer.len(), DEFAULT_BLOCK_SIZE);

        let prepared = self.module().prepare_slice_add(config).unwrap();
        let lhs = DeviceSpan::from_buffer(&vec1.buffer, 0, vec1.buffer.len());
        let rhs = DeviceSpan::from_buffer(&vec2.buffer, 0, vec2.buffer.len());
        let result_len = result_buffer.len();
        let output = DeviceSpanMut::from_buffer(&mut result_buffer, 0, result_len);

        self.module()
            .slice_add(
                self.stream(),
                &prepared,
                lhs.descriptor(),
                rhs.descriptor(),
                output.descriptor(),
            )
            .unwrap();
        self.sync();
        Vector {
            buffer: result_buffer,
        }
    }

    pub fn vector_dot_product(&self, vec1: &Vector, vec2: &Vector) -> f32 {
        assert!(
            vec1.buffer.len() == vec2.buffer.len(),
            "Vectors must have the same length for dot product."
        );
        let mut buffer = self.get_uninit_buffer(vec1.buffer.len());
        let config = self.get_launch_config(vec1.buffer.len(), DEFAULT_BLOCK_SIZE);
        let prepared = self.module().prepare_slice_mul(config).unwrap();
        let lhs = DeviceSpan::from_buffer(&vec1.buffer, 0, vec1.buffer.len());
        let rhs = DeviceSpan::from_buffer(&vec2.buffer, 0, vec2.buffer.len());
        let output_len = buffer.len();
        let output = DeviceSpanMut::from_buffer(&mut buffer, 0, output_len);
        self.module()
            .slice_mul(
                self.stream(),
                &prepared,
                lhs.descriptor(),
                rhs.descriptor(),
                output.descriptor(),
            )
            .unwrap();
        self.sync();
        let vector = Vector::new(buffer);
        vector.sum(self)
    }
}
