use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D, LaunchConfig2D, sys::random};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

use crate::cuda::{CudaRuntime, DEFAULT_BLOCK_SIZE, DeviceSpan, DeviceSpanMut, runtime::InitType};

pub struct Vector {
    buffer: DeviceBuffer<f32>,
}

pub struct VectorView<'a> {
    span: DeviceSpanMut<'a, f32>,
}

impl VectorView<'_> {
    pub fn sum(&self, runtime: &CudaRuntime) -> f32 {
        self.span.sum(runtime)
    }

    pub fn softmax(&mut self, runtime: &CudaRuntime) {
        let max = self.span.max(runtime);
        self.span.for_each(runtime, move|x| (x-max).exp());
        let sum = self.span.sum(runtime);
        self.span.scale(1.0 / sum, runtime);
    }

    pub fn map_sum<F>(&self, runtime: &CudaRuntime, f: F) -> f32
    where
        F: Fn(f32) -> f32 + Copy,
    {
        self.span.map_sum(runtime, f)
    }
}

pub mod compute;
pub mod view;

impl CudaRuntime {
    pub(super) fn create_vector(&self, buffer: DeviceBuffer<f32>) -> Vector {
        Vector { buffer }
    }

    pub fn new_vector(&mut self, init_type: InitType, size: usize) -> Vector {
        if init_type.is_zero() {
            return Vector {
                buffer: self.get_zerod_buffer(size),
            };
        }
        let mut buffer = self.get_uninit_buffer(size);
        let config = self.get_launch_config(buffer.len(), DEFAULT_BLOCK_SIZE);
        let prepared = self.module().prepare_vec_set_seq(config).unwrap();
        match init_type {
            InitType::Sequence => {
                self.module()
                    .vec_set_seq(self.stream(), &prepared, &mut buffer, true)
                    .unwrap();
                self.sync();
                Vector { buffer }
            }
            InitType::Reserve => {
                self.module()
                    .vec_set_seq(self.stream(), &prepared, &mut buffer, false)
                    .unwrap();
                self.sync();
                Vector { buffer }
            }
            InitType::Random => {
                let seed = rand::random();
                let prepared = self.module().prepare_vector_set_random(config).unwrap();
                self.module()
                    .vector_set_random(self.stream(), &prepared, &mut buffer, seed)
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
        let mut result_buffer =
            DeviceBuffer::<f32>::zeroed(&self.stream(), vec1.buffer.len()).unwrap();

        let config = self.get_launch_config(vec1.buffer.len(), DEFAULT_BLOCK_SIZE);

        let prepared = self.module().prepare_vector_add(config).unwrap();

        self.module()
            .vector_add(
                self.stream(),
                &prepared,
                &vec1.buffer,
                &vec2.buffer,
                &mut result_buffer,
            )
            .unwrap();
        self.sync();
        Vector {
            buffer: result_buffer,
        }
    }

    pub fn get_launch_config(&self, size: usize, block_size: usize) -> LaunchConfig1D {
        LaunchConfig1D::new(
            size.div_ceil(block_size).max(1) as u32,
            block_size as u32,
            0,
        )
    }

    pub fn vector_dot_product(&self, vec1: &Vector, vec2: &Vector) -> f32 {
        assert!(
            vec1.buffer.len() == vec2.buffer.len(),
            "Vectors must have the same length for dot product."
        );
        let mut buffer = self.get_uninit_buffer(vec1.buffer.len());
        let config = self.get_launch_config(vec1.buffer.len(), DEFAULT_BLOCK_SIZE);
        let prepared = self
            .module()
            .prepare_vector_pre_dot_product(config)
            .unwrap();
        self.module()
            .vector_pre_dot_product(
                self.stream(),
                &prepared,
                &vec1.buffer,
                &vec2.buffer,
                &mut buffer,
            )
            .unwrap();
        self.sync();
        let vector = Vector::new(buffer);
        return vector.sum(self);
    }
}
