use crate::cuda::{CudaRuntime, DEFAULT_BLOCK_SIZE};

use crate::cuda::vector::Vector;

impl Vector {
    pub fn to_host(&self, runtime: &CudaRuntime) -> Vec<f32> {
        self.buffer.to_host_vec(runtime.stream()).unwrap()
    }

    pub fn add(&mut self, value: f32, runtime: &CudaRuntime) {
        let config = runtime.get_launch_config(self.buffer.len(), DEFAULT_BLOCK_SIZE);

        let prepared = runtime.module().prepare_vector_for_each(config).unwrap();

        runtime
            .module()
            .vector_for_each(runtime.stream(), &prepared, &mut self.buffer, move |x| {
                x + value
            })
            .unwrap();
        runtime.sync();
    }

    pub fn scale(&mut self, value: f32, runtime: &CudaRuntime) {
        let config = runtime.get_launch_config(self.buffer.len(), DEFAULT_BLOCK_SIZE);
        let prepared = runtime.module().prepare_vector_for_each(config).unwrap();
        runtime
            .module()
            .vector_for_each(runtime.stream(), &prepared, &mut self.buffer, move |x| {
                x * value
            })
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
            let prepared = runtime.module().prepare_vector_sum(config).unwrap();
            runtime
                .module()
                .vector_sum(runtime.stream(), &prepared, &input, &mut output)
                .unwrap();
            runtime.sync();
            input = output;
        }
        input.to_host_vec(runtime.stream()).unwrap()[0]
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
            let prepared: cuda_core::PreparedLaunch<
                super::super::kernels::__vector_max_CudaKernel,
            > = runtime.module().prepare_vector_max(config).unwrap();
            runtime
                .module()
                .vector_max(runtime.stream(), &prepared, &input, &mut output)
                .unwrap();
            runtime.sync();
            input = output;
        }
        input.to_host_vec(runtime.stream()).unwrap()[0]
    }

    pub fn exp(&mut self, value: f32, runtime: &CudaRuntime) {
        let config = runtime.get_launch_config(self.buffer.len(), DEFAULT_BLOCK_SIZE);
        let prepared = runtime.module().prepare_vector_for_each(config).unwrap();
        let exp = move |x: f32| (x - value).exp();
        runtime
            .module()
            .vector_for_each(runtime.stream(), &prepared, &mut self.buffer, exp)
            .unwrap();
        runtime.sync();
    }
}
