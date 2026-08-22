use cuda_core::{CudaStream, LaunchConfig1D};

use crate::cuda::{BinaryOp, DEFAULT_BLOCK_SIZE, DeviceSpan, DeviceSpanMut, runtime::CudaRuntime};

use super::{Matrix, Vector};

impl Matrix {
    pub fn binary_assign_by_rows(&mut self, vec: &Vector, op: BinaryOp, runtime: &CudaRuntime) {
        self.binary_assign_by_rows_on(vec, op, runtime, runtime.stream());
    }

    pub(crate) fn binary_assign_by_rows_on(
        &mut self,
        vec: &Vector,
        op: BinaryOp,
        runtime: &CudaRuntime,
        stream: &CudaStream,
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
        runtime
            .module()
            .matrix_binary_assign_by_rows(
                stream,
                &prepared,
                matrix.descriptor(),
                rhs.descriptor(),
                self.cols,
                op,
            )
            .unwrap();
    }
}

impl CudaRuntime {
    pub fn matrix_sum_rows(&mut self, matrix: &Matrix) -> Vector {
        if matrix.rows == 0 {
            let buffer = self.get_uninit_buffer(0);
            return self.create_vector(buffer);
        }
        let mut buffer = self.get_uninit_buffer(matrix.rows);
        self.matrix_sum_rows_into_on(matrix, &mut buffer, self.stream());
        self.create_vector(buffer)
    }

    pub(crate) fn matrix_sum_rows_on(&mut self, matrix: &Matrix, stream: &CudaStream) -> Vector {
        if matrix.rows == 0 {
            let buffer = self.get_uninit_buffer(0);
            return self.create_vector(buffer);
        }
        let mut buffer = self.get_uninit_buffer(matrix.rows);
        stream.join(self.stream()).unwrap();
        self.matrix_sum_rows_into_on(matrix, &mut buffer, stream);
        self.create_vector(buffer)
    }

    fn matrix_sum_rows_into_on(
        &self,
        matrix: &Matrix,
        buffer: &mut cuda_core::DeviceBuffer<f32>,
        stream: &CudaStream,
    ) {
        assert!(matrix.cols > 0 && matrix.cols <= DEFAULT_BLOCK_SIZE);
        let config = LaunchConfig1D::new(matrix.rows as u32, DEFAULT_BLOCK_SIZE as u32, 0);
        let prepared = self.module().prepare_matrix_sum_rows(config).unwrap();
        let input = DeviceSpan::from_buffer(&matrix.buffer, 0, matrix.buffer.len());
        let result_len = buffer.len();
        let result = DeviceSpanMut::from_buffer(buffer, 0, result_len);
        self.module()
            .matrix_sum_rows(
                stream,
                &prepared,
                input.descriptor(),
                result.descriptor(),
                matrix.cols,
            )
            .unwrap();
    }
}
