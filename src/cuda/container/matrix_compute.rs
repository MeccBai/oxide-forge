use cuda_core::LaunchConfig2D;

use crate::cuda::{BinaryOp, DEFAULT_BLOCK_SIZE, DeviceSpan, DeviceSpanMut, runtime::CudaRuntime};

use super::Matrix;

impl Matrix {
    pub fn scale(&mut self, value: f32, runtime: &CudaRuntime) {
        self.for_each(runtime, move |x| x * value);
    }

    pub fn add_val(&mut self, value: f32, runtime: &CudaRuntime) {
        self.for_each(runtime, move |x| x + value);
    }

    pub fn binary_assign(&mut self, rhs: &Matrix, op: BinaryOp, runtime: &CudaRuntime) {
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
            .slice_binary_assign(
                runtime.stream(),
                &prepared,
                target.descriptor(),
                rhs.descriptor(),
                op,
            )
            .unwrap();
    }
}

impl CudaRuntime {
    pub fn matrix_multiply(&self, mat1: &Matrix, mat2: &Matrix) -> Matrix {
        assert_eq!(mat1.cols, mat2.rows);

        let rows = mat1.rows;
        let cols = mat2.cols;
        let len = mat1.cols;
        assert_eq!(rows % 16, 0);
        assert_eq!(cols % 16, 0);
        assert_eq!(len % 16, 0);

        let mut result_buffer = self.get_uninit_buffer(rows * cols);

        const TILE_SIZE: usize = 32;
        const BLOCK_SIZE: u32 = 128;
        let block = (BLOCK_SIZE, 1);

        let grid = (
            cols.div_ceil(TILE_SIZE) as u32,
            rows.div_ceil(TILE_SIZE) as u32,
        );

        let config = LaunchConfig2D::new(grid, block, 0);
        let prepared = self.module().prepare_matrix_multiply(config).unwrap();

        let lhs = DeviceSpan::from_buffer(&mat1.buffer, 0, mat1.buffer.len());
        let rhs = DeviceSpan::from_buffer(&mat2.buffer, 0, mat2.buffer.len());
        let result = DeviceSpanMut::from_buffer(&mut result_buffer, 0, rows * cols);

        self.module()
            .matrix_multiply(
                self.stream(),
                &prepared,
                lhs.descriptor(),
                rhs.descriptor(),
                result.descriptor(),
                len,
                rows,
                cols,
            )
            .unwrap();
        Matrix {
            buffer: result_buffer,
            rows,
            cols,
        }
    }

    pub fn matrix_add(&self, mat1: &Matrix, mat2: &Matrix) -> Matrix {
        self.matrix_binary(mat1, mat2, BinaryOp::Add)
    }

    pub fn matrix_sub(&self, mat1: &Matrix, mat2: &Matrix) -> Matrix {
        self.matrix_binary(mat1, mat2, BinaryOp::Sub)
    }

    pub fn matrix_mul(&self, mat1: &Matrix, mat2: &Matrix) -> Matrix {
        self.matrix_binary(mat1, mat2, BinaryOp::Mul)
    }

    pub fn matrix_div(&self, mat1: &Matrix, mat2: &Matrix) -> Matrix {
        self.matrix_binary(mat1, mat2, BinaryOp::Div)
    }

    pub fn matrix_binary(&self, mat1: &Matrix, mat2: &Matrix, op: BinaryOp) -> Matrix {
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
        Matrix {
            buffer: result_buffer,
            rows,
            cols,
        }
    }

    pub fn matrix_transpose(&self, mat: &Matrix) -> Matrix {
        let rows = mat.cols;
        let cols = mat.rows;
        let mut result_buffer = self.get_uninit_buffer(rows * cols);

        if mat.buffer.is_empty() {
            return Matrix {
                buffer: result_buffer,
                rows,
                cols,
            };
        }

        const TILE_SIZE: usize = 16;
        let grid = (
            mat.cols.div_ceil(TILE_SIZE) as u32,
            mat.rows.div_ceil(TILE_SIZE) as u32,
        );
        let config = LaunchConfig2D::new(grid, (TILE_SIZE as u32, TILE_SIZE as u32), 0);
        let prepared = self.module().prepare_matrix_transpose(config).unwrap();

        self.module()
            .matrix_transpose(
                self.stream(),
                &prepared,
                &mat.buffer,
                cuda_host::RowWidth::new(&mut result_buffer, cols as u32),
                mat.rows,
                mat.cols,
            )
            .unwrap();
        Matrix {
            buffer: result_buffer,
            rows,
            cols,
        }
    }
}
