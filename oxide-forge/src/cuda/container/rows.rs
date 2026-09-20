use cuda_core::{CudaStream, LaunchConfig1D};

use crate::cuda::{DEFAULT_BLOCK_SIZE, DeviceSpan, DeviceSpanMut, runtime::CudaRuntime};

use super::{Matrix, Vector};

impl Matrix {
    pub fn binary_assign_by_rows(
        &mut self,
        vec: &Vector,
        f: impl Fn(f32, f32) -> f32 + Copy,
        runtime: &CudaRuntime,
        stream: Option<&CudaStream>,
    ) {
        assert_eq!(self.cols, vec.len());
        if self.buffer.is_empty() {
            return;
        }
        let config = runtime.get_launch_config(self.buffer.len(), DEFAULT_BLOCK_SIZE);
        let prepared = runtime
            .module()
            .prepare_matrix_binary_assign_by_rows(config)
            .unwrap();
        let len = self.buffer.len();
        let matrix = DeviceSpanMut::from_buffer(&mut self.buffer, 0, len);
        let rhs = DeviceSpan::from_buffer(&vec.buffer, 0, vec.buffer.len());
        let stream = runtime.execution_stream(stream);
        runtime
            .module()
            .matrix_binary_assign_by_rows(
                stream,
                &prepared,
                matrix.descriptor(),
                rhs.descriptor(),
                self.cols,
                f,
            )
            .unwrap();
    }
}

impl CudaRuntime {
    pub fn matrix_sum_rows(&mut self, matrix: &Matrix, stream: Option<&CudaStream>) -> Vector {
        if matrix.rows == 0 {
            let buffer = self.get_uninit_buffer(0);
            return self.create_vector(buffer);
        }
        let mut buffer = self.get_uninit_buffer(matrix.rows);
        assert!(matrix.cols > 0);
        let elements_per_thread = matrix.cols.div_ceil(DEFAULT_BLOCK_SIZE);
        let config = LaunchConfig1D::new(matrix.rows as u32, DEFAULT_BLOCK_SIZE as u32, 128);
        let prepared = self.module().prepare_matrix_sum_rows(config).unwrap();
        let input = DeviceSpan::from_buffer(&matrix.buffer, 0, matrix.buffer.len());
        let result_len = buffer.len();
        let result = DeviceSpanMut::from_buffer(&mut buffer, 0, result_len);
        let stream = self.execution_stream(stream);
        self.module()
            .matrix_sum_rows(
                stream,
                &prepared,
                input.descriptor(),
                result.descriptor(),
                matrix.cols,
                elements_per_thread,
            )
            .unwrap();
        self.create_vector(buffer)
    }
}
