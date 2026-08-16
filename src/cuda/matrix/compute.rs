use cuda_core::{DeviceBuffer, LaunchConfig2D};

use crate::cuda::{
    DEFAULT_BLOCK_SIZE, DeviceSpan, DeviceSpanMut, runtime::CudaRuntime, runtime::InitType,
};

use super::Matrix;

impl Matrix {
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

    pub fn add_val(&mut self, value: f32, runtime: &CudaRuntime) {
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

        let block = (16u32, 16u32);

        let grid = (cols.div_ceil(16) as u32, rows.div_ceil(16) as u32);

        let config = LaunchConfig2D::new(grid, block, 0);
        let prepared = self.module().prepare_matrix_multiply(config).unwrap();

        self.module()
            .matrix_multiply(
                self.stream(),
                &prepared,
                &mat1.buffer,
                &mat2.buffer,
                cuda_host::RowWidth::new(&mut result_buffer, cols as u32),
                len,
                rows,
                cols,
            )
            .unwrap();
        self.sync();

        Matrix {
            buffer: result_buffer,
            rows,
            cols,
        }
    }

    pub fn matrix_add(&self, mat1: &Matrix, mat2: &Matrix) -> Matrix {
        assert_eq!(mat1.rows, mat2.rows);
        assert_eq!(mat1.cols, mat2.cols);

        let rows = mat1.rows;
        let cols = mat1.cols;

        let mut result_buffer = self.get_uninit_buffer(rows * cols);

        let config = self.get_launch_config(mat1.buffer.len(), DEFAULT_BLOCK_SIZE);
        let prepared = self.module().prepare_vector_add(config).unwrap();

        self.module()
            .vector_add(
                self.stream(),
                &prepared,
                &mat1.buffer,
                &mat2.buffer,
                &mut result_buffer,
            )
            .unwrap();
        self.sync();

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
        self.sync();

        Matrix {
            buffer: result_buffer,
            rows,
            cols,
        }
    }
}
