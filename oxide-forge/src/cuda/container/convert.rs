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
    /// Concatenates equally wide matrices by appending complete rows.
    pub fn matrix_concat_rows(
        &mut self,
        matrices: &[&Matrix],
        stream: Option<&CudaStream>,
    ) -> Matrix {
        assert!(
            !matrices.is_empty(),
            "matrix concat requires at least one input"
        );
        let cols = matrices[0].cols;
        let rows = matrices
            .iter()
            .map(|matrix| {
                assert_eq!(matrix.cols, cols, "row concat column mismatch");
                matrix.rows
            })
            .try_fold(0usize, usize::checked_add)
            .expect("row concat size overflow");
        let buffers = matrices
            .iter()
            .map(|matrix| &matrix.buffer)
            .collect::<Vec<_>>();
        let buffer = self.concat_buffers_on(&buffers, stream);
        self.create_matrix(buffer, rows, cols)
    }

    /// Concatenates equally tall matrices by physically interleaving each
    /// row's contiguous segments into a new row-major allocation.
    pub fn matrix_concat_cols(
        &mut self,
        matrices: &[&Matrix],
        stream: Option<&CudaStream>,
    ) -> Matrix {
        assert!(
            !matrices.is_empty(),
            "matrix concat requires at least one input"
        );
        let rows = matrices[0].rows;
        let cols = matrices
            .iter()
            .map(|matrix| {
                assert_eq!(matrix.rows, rows, "column concat row mismatch");
                matrix.cols
            })
            .try_fold(0usize, usize::checked_add)
            .expect("column concat size overflow");

        let mut spans = Vec::with_capacity(rows.saturating_mul(matrices.len()));
        for row in 0..rows {
            for matrix in matrices {
                spans.push(DeviceSpan::from_buffer(
                    &matrix.buffer,
                    row.checked_mul(matrix.cols).expect("row offset overflow"),
                    matrix.cols,
                ));
            }
        }
        let buffer = self.concat_buffers_from_span_on(&spans, stream);
        self.create_matrix(buffer, rows, cols)
    }

    pub fn matrix_split_rows(
        &mut self,
        matrix: &Matrix,
        row_sizes: &[usize],
        stream: Option<&CudaStream>,
    ) -> Vec<Matrix> {
        validate_partition(row_sizes, matrix.rows, "row");
        let mut offset = 0usize;
        row_sizes
            .iter()
            .map(|&rows| {
                let len = rows
                    .checked_mul(matrix.cols)
                    .expect("row split size overflow");
                let span = DeviceSpan::from_buffer(&matrix.buffer, offset, len);
                offset += len;
                let buffer = if let Some(stream) = stream {
                    span.to_buffer_on(self, stream)
                } else {
                    span.to_buffer(self)
                };
                self.create_matrix(buffer, rows, matrix.cols)
            })
            .collect()
    }

    pub fn matrix_split_cols(
        &mut self,
        matrix: &Matrix,
        col_sizes: &[usize],
        stream: Option<&CudaStream>,
    ) -> Vec<Matrix> {
        validate_partition(col_sizes, matrix.cols, "column");
        let mut column_offset = 0usize;
        col_sizes
            .iter()
            .map(|&cols| {
                let spans = (0..matrix.rows)
                    .map(|row| {
                        DeviceSpan::from_buffer(
                            &matrix.buffer,
                            row.checked_mul(matrix.cols)
                                .and_then(|offset| offset.checked_add(column_offset))
                                .expect("column split offset overflow"),
                            cols,
                        )
                    })
                    .collect::<Vec<_>>();
                column_offset += cols;
                let buffer = self.concat_buffers_from_span_on(&spans, stream);
                self.create_matrix(buffer, matrix.rows, cols)
            })
            .collect()
    }

    pub fn matrix_transpose(&mut self, mat: &Matrix, stream: Option<&CudaStream>) -> Matrix {
        let rows = mat.cols;
        let cols = mat.rows;
        let mut result_buffer = self.get_uninit_buffer(rows * cols);
        let stream = self.execution_stream(stream);
        self.matrix_transpose_into_on(stream, mat, &mut result_buffer);
        self.create_matrix(result_buffer, rows, cols)
    }

    pub(crate) fn matrix_transpose_batches(
        &mut self,
        matrix: &Matrix,
        batch_count: usize,
        rows: usize,
        cols: usize,
        stream: Option<&CudaStream>,
    ) -> Matrix {
        assert!(batch_count > 0 && rows > 0 && cols > 0);
        assert_eq!(matrix.buffer.len(), batch_count * rows * cols);
        let mut result = self.new_uninit_matrix(batch_count * cols, rows);
        let grid = (
            cols.div_ceil(32) as u32,
            (batch_count * rows.div_ceil(32)) as u32,
        );
        let config = LaunchConfig2D::new(grid, (32, 8), 0);
        let prepared = self
            .module()
            .prepare_matrix_transpose_batches(config)
            .unwrap();
        let input = DeviceSpan::from_buffer(&matrix.buffer, 0, matrix.buffer.len());
        let output_len = result.buffer.len();
        let output = DeviceSpanMut::from_buffer(&mut result.buffer, 0, output_len);
        let stream = self.execution_stream(stream);
        self.module()
            .matrix_transpose_batches(
                stream,
                &prepared,
                input.descriptor(),
                output.descriptor(),
                rows,
                cols,
                batch_count,
            )
            .unwrap();
        result
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

    pub fn vector_zip(&mut self, vecs: &[Vector], stream: Option<&CudaStream>) -> Matrix {
        let spans = vecs.iter().map(|v| v.as_span()).collect::<Vec<_>>();
        let rows = spans.len();
        let cols = spans[0].len();
        let buffer = self.concat_buffers_from_span_on(&spans, stream);

        self.create_matrix(buffer, rows, cols)
    }

    pub fn matrix_split(&mut self, matrix: &Matrix, stream: Option<&CudaStream>) -> Vec<Vector> {
        let spans = DeviceSpan::chunks(&matrix.buffer, matrix.cols);
        let mut vectors = Vec::with_capacity(spans.len());
        for span in spans {
            let buffer = if let Some(stream) = stream {
                span.to_buffer_on(self, stream)
            } else {
                span.to_buffer(self)
            };
            vectors.push(self.create_vector(buffer));
        }
        vectors
    }

    pub fn broadcast(
        &mut self,
        vector: &Vector,
        copies: usize,
        stream: Option<&CudaStream>,
    ) -> Matrix {
        let spans = vec![vector.as_span(); copies];
        let buffer = self.concat_buffers_from_span_on(&spans, stream);
        self.create_matrix(buffer, copies, vector.len())
    }

    pub fn extract_vector(&mut self, matrix: Matrix, stream: Option<&CudaStream>) -> Vector {
        assert!(
            matrix.rows > 0,
            "cannot extract a vector from an empty matrix"
        );

        if matrix.rows == 1 {
            return self.create_vector(matrix.buffer);
        }

        let span = DeviceSpan::from_buffer(&matrix.buffer, 0, matrix.cols);
        let buffer = if let Some(stream) = stream {
            span.to_buffer_on(self, stream)
        } else {
            span.to_buffer(self)
        };
        self.create_vector(buffer)
    }

    pub fn matrix_slice(
        &mut self,
        matrix: &Matrix,
        cols: usize,
        rows: usize,
        stream: Option<&CudaStream>,
    ) -> Vec<Matrix> {
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

                let buffer = self.concat_buffers_from_span_on(&tile_spans, stream);
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

fn validate_partition(sizes: &[usize], expected: usize, axis: &str) {
    assert!(
        !sizes.is_empty(),
        "matrix split requires at least one output"
    );
    assert!(
        sizes.iter().all(|&size| size > 0),
        "matrix split sizes must be non-zero"
    );
    let total = sizes
        .iter()
        .copied()
        .try_fold(0usize, usize::checked_add)
        .expect("matrix split size overflow");
    assert_eq!(total, expected, "{axis} split sizes do not cover input");
}
