use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig1D, LaunchConfig2D};

use crate::cuda::{
    BinaryOp, DEFAULT_BLOCK_SIZE, DeviceSpan, DeviceSpanMut,
    runtime::{CudaRuntime, InitType},
};

use super::Matrix;

impl Matrix {
    pub fn to_host(&self, runtime: &CudaRuntime) -> Vec<f32> {
        self.buffer.to_host_vec(runtime.stream()).unwrap()
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn for_each<F>(&mut self, runtime: &CudaRuntime, f: F)
    where
        F: Fn(f32) -> f32 + Copy,
    {
        self.for_each_on(runtime, runtime.stream(), f);
    }

    pub fn scale(&mut self, value: f32, runtime: &CudaRuntime) {
        self.for_each(runtime, move |x| x * value);
    }

    pub fn add_scalar(&mut self, value: f32, runtime: &CudaRuntime) {
        self.for_each(runtime, move |x| x + value);
    }

    pub fn binary_assign(&mut self, rhs: &Matrix, op: BinaryOp, runtime: &CudaRuntime) {
        self.binary_assign_on(rhs, op, runtime, runtime.stream());
    }

    pub(crate) fn binary_assign_on(
        &mut self,
        rhs: &Matrix,
        op: BinaryOp,
        runtime: &CudaRuntime,
        stream: &CudaStream,
    ) {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        let len = self.buffer.len();
        let config = runtime.get_launch_config(len, DEFAULT_BLOCK_SIZE);
        let prepared = runtime
            .module()
            .prepare_slice_binary_assign(config)
            .unwrap();
        let target = DeviceSpanMut::from_buffer(&mut self.buffer, 0, len);
        let rhs = DeviceSpan::from_buffer(&rhs.buffer, 0, len);
        runtime
            .module()
            .slice_binary_assign(stream, &prepared, target.descriptor(), rhs.descriptor(), op)
            .unwrap();
    }

    pub(crate) fn for_each_on<F>(&mut self, runtime: &CudaRuntime, stream: &CudaStream, f: F)
    where
        F: Fn(f32) -> f32 + Copy,
    {
        if self.buffer.is_empty() {
            return;
        }

        let len = self.buffer.len();
        let mut span = DeviceSpanMut::from_buffer(&mut self.buffer, 0, len);
        span.for_each_on(runtime, stream, f);
    }

    pub fn causal_mask(&mut self, runtime: &CudaRuntime) {
        if self.rows == 0 {
            return;
        }
        assert!(self.cols > 0 && self.cols <= DEFAULT_BLOCK_SIZE);
        let config = LaunchConfig1D::new(self.rows as u32, self.cols as u32, 0);
        let prepared = runtime.module().prepare_matrix_causal_mask(config).unwrap();
        let len = self.buffer.len();
        let matrix = DeviceSpanMut::from_buffer(&mut self.buffer, 0, len);
        runtime
            .module()
            .matrix_causal_mask(runtime.stream(), &prepared, matrix.descriptor(), self.cols)
            .unwrap();
    }
}

impl CudaRuntime {
    pub fn matrix_from_host(
        &self,
        values: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Matrix, DriverError> {
        let len = rows
            .checked_mul(cols)
            .expect("matrix element count overflow");
        assert_eq!(values.len(), len);
        Ok(Matrix {
            buffer: DeviceBuffer::from_host(self.stream(), values)?,
            rows,
            cols,
        })
    }

    pub fn matrix_multiply(&mut self, mat1: &Matrix, mat2: &Matrix) -> Matrix {
        assert_eq!(mat1.cols, mat2.rows);
        let mut result = self.new_uninit_matrix(mat1.rows, mat2.cols);
        self.matrix_multiply_into_on(self.stream(), mat1, mat2, &mut result);
        result
    }

    pub(crate) fn matrix_multiply_on(
        &mut self,
        stream: &CudaStream,
        mat1: &Matrix,
        mat2: &Matrix,
    ) -> Matrix {
        assert_eq!(mat1.cols, mat2.rows);
        let mut result = self.new_uninit_matrix(mat1.rows, mat2.cols);
        stream.join(self.stream()).unwrap();
        self.matrix_multiply_into_on(stream, mat1, mat2, &mut result);
        result
    }

    pub(crate) fn matrix_multiply_into_on(
        &self,
        stream: &CudaStream,
        mat1: &Matrix,
        mat2: &Matrix,
        result: &mut Matrix,
    ) {
        assert_eq!(mat1.cols, mat2.rows);

        let rows = mat1.rows;
        let cols = mat2.cols;
        let len = mat1.cols;
        assert_eq!(result.rows, rows);
        assert_eq!(result.cols, cols);
        assert_eq!(rows % 16, 0);
        assert_eq!(cols % 16, 0);
        assert_eq!(len % 16, 0);

        const TILE_SIZE: usize = 32;
        const BLOCK_SIZE: u32 = 128;
        let grid = (
            cols.div_ceil(TILE_SIZE) as u32,
            rows.div_ceil(TILE_SIZE) as u32,
        );
        let config = LaunchConfig2D::new(grid, (BLOCK_SIZE, 1), 0);
        let prepared = self.module().prepare_matrix_multiply(config).unwrap();

        let lhs = DeviceSpan::from_buffer(&mat1.buffer, 0, mat1.buffer.len());
        let rhs = DeviceSpan::from_buffer(&mat2.buffer, 0, mat2.buffer.len());
        let output = DeviceSpanMut::from_buffer(&mut result.buffer, 0, rows * cols);

        self.module()
            .matrix_multiply(
                stream,
                &prepared,
                lhs.descriptor(),
                rhs.descriptor(),
                output.descriptor(),
                len,
                rows,
                cols,
            )
            .unwrap();
    }

    pub fn matrix_add(&mut self, mat1: &Matrix, mat2: &Matrix) -> Matrix {
        self.matrix_binary(mat1, mat2, BinaryOp::Add)
    }

    pub fn matrix_sub(&mut self, mat1: &Matrix, mat2: &Matrix) -> Matrix {
        self.matrix_binary(mat1, mat2, BinaryOp::Sub)
    }

    pub fn matrix_mul(&mut self, mat1: &Matrix, mat2: &Matrix) -> Matrix {
        self.matrix_binary(mat1, mat2, BinaryOp::Mul)
    }

    pub fn matrix_div(&mut self, mat1: &Matrix, mat2: &Matrix) -> Matrix {
        self.matrix_binary(mat1, mat2, BinaryOp::Div)
    }

    pub fn matrix_binary(&mut self, mat1: &Matrix, mat2: &Matrix, op: BinaryOp) -> Matrix {
        assert_eq!(mat1.rows, mat2.rows);
        assert_eq!(mat1.cols, mat2.cols);

        let rows = mat1.rows;
        let cols = mat1.cols;
        let mut result_buffer = self.get_uninit_buffer(rows * cols);

        let config = self.get_launch_config(mat1.buffer.len(), DEFAULT_BLOCK_SIZE);
        let prepared = self.module().prepare_slice_binary(config).unwrap();
        let lhs = DeviceSpan::from_buffer(&mat1.buffer, 0, mat1.buffer.len());
        let rhs = DeviceSpan::from_buffer(&mat2.buffer, 0, mat2.buffer.len());
        let output = DeviceSpanMut::from_buffer(&mut result_buffer, 0, rows * cols);

        self.module()
            .slice_binary(
                self.stream(),
                &prepared,
                lhs.descriptor(),
                rhs.descriptor(),
                output.descriptor(),
                op,
            )
            .unwrap();
        self.create_matrix(result_buffer, rows, cols)
    }

    pub(crate) fn matrix_binary_on(
        &mut self,
        stream: &CudaStream,
        mat1: &Matrix,
        mat2: &Matrix,
        op: BinaryOp,
    ) -> Matrix {
        assert_eq!(mat1.rows, mat2.rows);
        assert_eq!(mat1.cols, mat2.cols);

        let rows = mat1.rows;
        let cols = mat1.cols;
        let mut result_buffer = self.get_uninit_buffer(rows * cols);
        stream.join(self.stream()).unwrap();

        let config = self.get_launch_config(mat1.buffer.len(), DEFAULT_BLOCK_SIZE);
        let prepared = self.module().prepare_slice_binary(config).unwrap();
        let lhs = DeviceSpan::from_buffer(&mat1.buffer, 0, mat1.buffer.len());
        let rhs = DeviceSpan::from_buffer(&mat2.buffer, 0, mat2.buffer.len());
        let output = DeviceSpanMut::from_buffer(&mut result_buffer, 0, rows * cols);

        self.module()
            .slice_binary(
                stream,
                &prepared,
                lhs.descriptor(),
                rhs.descriptor(),
                output.descriptor(),
                op,
            )
            .unwrap();
        self.create_matrix(result_buffer, rows, cols)
    }

    /// Consumes a matrix and returns its allocation to the runtime pool.
    pub fn recycle_matrix(&mut self, matrix: Matrix) {
        self.recycle_buffer(matrix.buffer);
    }

    pub fn new_matrix(&mut self, init_type: InitType, rows: usize, cols: usize) -> Matrix {
        let size = rows * cols;
        if init_type.is_zero() {
            let buffer = self.get_zerod_buffer(size);
            return self.create_matrix(buffer, rows, cols);
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
            }
            InitType::Reserve => {
                let prepared = self.module().prepare_slice_set_seq(config).unwrap();
                let span = DeviceSpanMut::from_buffer(&mut buffer, 0, size);
                self.module()
                    .slice_set_seq(self.stream(), &prepared, span.descriptor(), false)
                    .unwrap();
            }
            InitType::Random => {
                let seed = rand::random();
                let prepared = self.module().prepare_slice_set_random(config).unwrap();
                let span = DeviceSpanMut::from_buffer(&mut buffer, 0, size);
                self.module()
                    .slice_set_random(self.stream(), &prepared, span.descriptor(), seed)
                    .unwrap();
            }
            InitType::Zero => {}
        }
        self.create_matrix(buffer, rows, cols)
    }

    pub(crate) fn new_uninit_matrix(&mut self, rows: usize, cols: usize) -> Matrix {
        let len = rows
            .checked_mul(cols)
            .expect("matrix element count overflow");
        let buffer = self.get_uninit_buffer(len);
        self.create_matrix(buffer, rows, cols)
    }

    pub(crate) fn create_matrix(
        &self,
        buffer: DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
    ) -> Matrix {
        Matrix { buffer, rows, cols }
    }

    pub fn clone_matrix(&mut self, matrix: &Matrix) -> Matrix {
        let buffer = self.clone_buffer(&matrix.buffer);
        self.create_matrix(buffer, matrix.rows, matrix.cols)
    }

    pub(crate) fn clone_matrix_on(&mut self, matrix: &Matrix, stream: &CudaStream) -> Matrix {
        let mut buffer = self.get_uninit_buffer(matrix.buffer.len());
        stream.join(self.stream()).unwrap();
        buffer
            .copy_from_device_async(&matrix.buffer, stream)
            .unwrap();
        self.create_matrix(buffer, matrix.rows, matrix.cols)
    }
}
