use cuda_core::{CudaStream, LaunchConfig2D};

use crate::cuda::{DeviceSpan, DeviceSpanMut, runtime::CudaRuntime};

use super::{Matrix, Vector, VectorView};

impl Matrix {
    pub fn row_views(&mut self) -> Vec<VectorView<'_>> {
        DeviceSpanMut::chunks(&mut self.buffer, self.cols)
            .into_iter()
            .map(VectorView::new)
            .collect()
    }
}

impl CudaRuntime {
    pub fn matrix_transpose(&mut self, mat: &Matrix) -> Matrix {
        let rows = mat.cols;
        let cols = mat.rows;
        let mut result_buffer = self.get_uninit_buffer(rows * cols);
        self.matrix_transpose_into_on(self.stream(), mat, &mut result_buffer);
        self.create_matrix(result_buffer, rows, cols)
    }

    pub(crate) fn matrix_transpose_on(&mut self, mat: &Matrix, stream: &CudaStream) -> Matrix {
        let rows = mat.cols;
        let cols = mat.rows;
        let mut result_buffer = self.get_uninit_buffer(rows * cols);
        stream.join(self.stream()).unwrap();
        self.matrix_transpose_into_on(stream, mat, &mut result_buffer);
        self.create_matrix(result_buffer, rows, cols)
    }

    fn matrix_transpose_into_on(
        &self,
        stream: &CudaStream,
        mat: &Matrix,
        result_buffer: &mut cuda_core::DeviceBuffer<f32>,
    ) {
        if mat.buffer.is_empty() {
            return;
        }

        const TILE_SIZE: usize = 32;
        const BLOCK_ROWS: usize = 8;
        let grid = (
            mat.cols.div_ceil(TILE_SIZE) as u32,
            mat.rows.div_ceil(TILE_SIZE) as u32,
        );
        let config = LaunchConfig2D::new(grid, (TILE_SIZE as u32, BLOCK_ROWS as u32), 0);
        let prepared = self.module().prepare_matrix_transpose(config).unwrap();

        self.module()
            .matrix_transpose(
                stream,
                &prepared,
                &mat.buffer,
                cuda_host::RowWidth::new(result_buffer, mat.rows as u32),
                mat.rows,
                mat.cols,
            )
            .unwrap();
    }

    pub fn vector_zip(&mut self, vecs: &[Vector]) -> Matrix {
        let spans = vecs.iter().map(|v| v.as_span()).collect::<Vec<_>>();
        let rows = spans.len();
        let cols = spans[0].len();
        let buffer = self.concat_buffers_from_span(&spans);

        self.create_matrix(buffer, rows, cols)
    }

    pub fn matrix_split(&mut self, matrix: &Matrix) -> Vec<Vector> {
        let spans = DeviceSpan::chunks(&matrix.buffer, matrix.cols);
        let mut vectors = Vec::with_capacity(spans.len());
        for span in spans {
            let buffer = span.to_buffer(self);
            vectors.push(self.create_vector(buffer));
        }
        vectors
    }

    pub fn broadcast(&mut self, vector: &Vector, copies: usize) -> Matrix {
        let spans = vec![vector.as_span(); copies];
        let buffer = self.concat_buffers_from_span(&spans);
        self.create_matrix(buffer, copies, vector.len())
    }

    pub fn extract_vector(&mut self, matrix: Matrix) -> Vector {
        assert!(
            matrix.rows > 0,
            "cannot extract a vector from an empty matrix"
        );

        if matrix.rows == 1 {
            return self.create_vector(matrix.buffer);
        }

        let span = DeviceSpan::from_buffer(&matrix.buffer, 0, matrix.cols);
        let buffer = span.to_buffer(self);
        self.create_vector(buffer)
    }

    pub fn matrix_slice(&mut self, matrix: &Matrix, cols: usize, rows: usize) -> Vec<Matrix> {
        assert!(cols > 0, "matrix slice cols must be non-zero");
        assert!(rows > 0, "matrix slice rows must be non-zero");
        assert_eq!(
            matrix.cols % cols,
            0,
            "matrix cols must be divisible by slice cols"
        );
        assert_eq!(
            matrix.rows % rows,
            0,
            "matrix rows must be divisible by slice rows"
        );

        // Each span is one contiguous row segment of an output tile.
        let spans = DeviceSpan::chunks(&matrix.buffer, cols);
        let tiles_per_row = matrix.cols / cols;
        let tile_row_count = matrix.rows / rows;
        let mut result = Vec::with_capacity(tiles_per_row * tile_row_count);

        for tile_row in 0..tile_row_count {
            for tile_col in 0..tiles_per_row {
                let mut tile_spans = Vec::with_capacity(rows);

                for local_row in 0..rows {
                    let matrix_row = tile_row * rows + local_row;
                    let span_index = matrix_row * tiles_per_row + tile_col;
                    tile_spans.push(spans[span_index].clone());
                }

                let buffer = self.concat_buffers_from_span(&tile_spans);
                result.push(self.create_matrix(buffer, rows, cols));
            }
        }

        result
    }

    pub fn matrix_into_vector(&self, matrix: Matrix) -> Vector {
        self.create_vector(matrix.buffer)
    }

    pub fn vector_into_matrix(&self, vector: Vector) -> Matrix {
        let rows = vector.buffer.len();
        self.create_matrix(vector.buffer, rows, 1)
    }
}
