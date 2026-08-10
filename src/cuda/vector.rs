use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D, LaunchConfig2D, sys::random};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

use crate::cuda::{
    CudaRuntime, DEFAULT_BLOCK_SIZE, DeviceSpanMut,
    InitType::{self, Random},
    matrix::Matrix,
};

pub struct Vector {
    buffer: DeviceBuffer<f32>,
}

pub struct VectorView<'a> {
    span: DeviceSpanMut<'a, f32>,
}

impl Vector {
    pub fn to_host(&self, runtime: &CudaRuntime) -> Vec<f32> {
        self.buffer.to_host_vec(&runtime.stream).unwrap()
    }

    pub fn add(&mut self, value: f32, runtime: &CudaRuntime) {
        let config = runtime.get_launch_config(self.buffer.len(), DEFAULT_BLOCK_SIZE);

        let prepared = runtime.module.prepare_vector_for_each(config).unwrap();

        runtime
            .module
            .vector_for_each(&runtime.stream, &prepared, &mut self.buffer, |x| x + value)
            .unwrap();
        runtime.sync();
    }

    pub fn scale(&mut self, value: f32, runtime: &CudaRuntime) {
        let config = runtime.get_launch_config(self.buffer.len(), DEFAULT_BLOCK_SIZE);
        let prepared = runtime.module.prepare_vector_for_each(config).unwrap();
        runtime
            .module
            .vector_for_each(&runtime.stream, &prepared, &mut self.buffer, |x| x * value)
            .unwrap();
        runtime.sync();
    }

    pub fn sum(&self, runtime: &CudaRuntime) -> f32 {
        if self.buffer.len() == 0 {
            return 0.0;
        }
        let mut input = runtime.clone_buffer(&self.buffer);
        while input.len() > 1 {
            let output_len = input.len().div_ceil(DEFAULT_BLOCK_SIZE);
            let mut output = runtime.get_uninit_buffer(output_len);
            let config = runtime.get_launch_config(input.len(), DEFAULT_BLOCK_SIZE);
            let prepared = runtime.module.prepare_vector_sum(config).unwrap();
            runtime
                .module
                .vector_sum(&runtime.stream, &prepared, &input, &mut output)
                .unwrap();
            runtime.sync();
            input = output;
        }
        input.to_host_vec(&runtime.stream).unwrap()[0]
    }

    pub fn max(&self, runtime: &CudaRuntime) -> f32 {
        if self.buffer.len() == 0 {
            return f32::MIN;
        }
        let mut input = runtime.clone_buffer(&self.buffer);
        while input.len() > 1 {
            let output_len = input.len().div_ceil(DEFAULT_BLOCK_SIZE);
            let mut output = runtime.get_uninit_buffer(output_len);
            let config = runtime.get_launch_config(input.len(), DEFAULT_BLOCK_SIZE);
            let prepared = runtime.module.prepare_vector_max(config).unwrap();
            runtime
                .module
                .vector_max(&runtime.stream, &prepared, &input, &mut output)
                .unwrap();
            runtime.sync();
            input = output;
        }
        input.to_host_vec(&runtime.stream).unwrap()[0]
    }

    pub fn exp(&mut self, value: f32, runtime: &CudaRuntime) {
        let config = runtime.get_launch_config(self.buffer.len(), DEFAULT_BLOCK_SIZE);
        let prepared = runtime.module.prepare_vector_for_each(config).unwrap();
        let exp = |x: f32| (x - value).exp();
        runtime
            .module
            .vector_for_each(&runtime.stream, &prepared, &mut self.buffer, exp)
            .unwrap();
        runtime.sync();
    }

    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }
}

impl<'a> VectorView<'a> {
    pub(super) fn new(span: DeviceSpanMut<'a, f32>) -> Self {
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

impl CudaRuntime {
    pub fn new_vector(&mut self, init_type: InitType, size: usize) -> Vector {
        if init_type.is_zero() {
            return Vector {
                buffer: self.get_zerod_buffer(size),
            };
        }
        let mut buffer = self.get_uninit_buffer(size);
        let config = self.get_launch_config(buffer.len(), DEFAULT_BLOCK_SIZE);
        let prepared = self.module.prepare_vec_set_seq(config).unwrap();
        match init_type {
            InitType::Sequence => {
                self.module
                    .vec_set_seq(&self.stream, &prepared, &mut buffer, true)
                    .unwrap();
                self.sync();
                Vector { buffer }
            }
            InitType::Reserve => {
                self.module
                    .vec_set_seq(&self.stream, &prepared, &mut buffer, false)
                    .unwrap();
                self.sync();
                Vector { buffer }
            }
            InitType::Random => {
                let seed = rand::random();
                let prepared = self.module.prepare_vector_set_random(config).unwrap();
                self.module
                    .vector_set_random(&self.stream, &prepared, &mut buffer, seed)
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

    pub fn clone_buffer(&self, buffer: &DeviceBuffer<f32>) -> DeviceBuffer<f32> {
        let mut new_buffer = self.get_uninit_buffer(buffer.len());
        new_buffer
            .copy_from_device_async(buffer, &self.stream)
            .unwrap();
        self.stream.synchronize().unwrap();
        new_buffer
    }

    pub fn vector_add(&self, vec1: &Vector, vec2: &Vector) -> Vector {
        let mut result_buffer =
            DeviceBuffer::<f32>::zeroed(&self.stream, vec1.buffer.len()).unwrap();

        let config = self.get_launch_config(vec1.buffer.len(), DEFAULT_BLOCK_SIZE);

        let prepared = self.module.prepare_vector_add(config).unwrap();

        self.module
            .vector_add(
                &self.stream,
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
}
