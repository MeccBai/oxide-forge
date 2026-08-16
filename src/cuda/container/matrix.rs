use cuda_core::DeviceBuffer;

use crate::cuda::{
    DEFAULT_BLOCK_SIZE, DeviceSpan, DeviceSpanMut, runtime::CudaRuntime, runtime::InitType,
};

use crate::cuda::container::{Vector, VectorView};

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

    pub fn softmax_rows(&mut self, runtime: &CudaRuntime) {
        let mut rows = runtime.split_view(self);
        for row in &mut rows {
            row.softmax(runtime);
        }
        runtime.sync();
    }

    pub fn sum_rows(&mut self, runtime: &CudaRuntime) -> Vector {
        let mut rows = runtime
            .split_view(self)
            .into_iter()
            .map(|row| row.sum(runtime))
            .collect::<Vec<f32>>();

        runtime
            .create_vector(DeviceBuffer::from_host(runtime.stream(), rows.as_mut_slice()).unwrap())
    }

    pub fn layer_norm(&mut self, runtime: &CudaRuntime) {
        let mut rows = runtime.split_view(self);
        for row in &mut rows {
            let mean = row.sum(runtime) / row.len() as f32;
            let variance = row.map_sum(runtime, move |x| {
                let diff = x - mean;
                diff * diff
            }) / row.len() as f32;
            let std = (variance + 1e-5).sqrt();
            row.for_each(runtime, move |x| (x - mean) / std);
        }
        runtime.sync();
    }

    pub fn for_each<F>(&mut self, runtime: &CudaRuntime, f: F)
    where
        F: Fn(f32) -> f32 + Copy,
    {
        if self.buffer.is_empty() {
            return;
        }

        let len = self.buffer.len();
        let mut span = DeviceSpanMut::from_buffer(&mut self.buffer, 0, len);
        span.for_each(runtime, f);
    }

    pub fn add_by_rows(&mut self, vec: &Vector, runtime: &CudaRuntime) {
        let mut rows = runtime.split_view(self);
        let streams = runtime.create_extra_streams(rows.len());
        let rhs = DeviceSpan::from_buffer(&vec.buffer, 0, vec.buffer.len());

        rows.iter_mut()
            .zip(streams.iter())
            .for_each(|(row, stream)| {
                let config = runtime.get_launch_config(row.len(), DEFAULT_BLOCK_SIZE);
                let prepared = runtime.module().prepare_slice_add_assign(config).unwrap();
                runtime
                    .module()
                    .slice_add_assign(stream, &prepared, row.span.descriptor(), rhs.descriptor())
                    .unwrap();
            });
        runtime.sync_streams(&streams);
    }
}

impl CudaRuntime {
    pub fn new_matrix(&self, init_type: InitType, rows: usize, cols: usize) -> Matrix {
        let size = rows * cols;
        if init_type.is_zero() {
            return Matrix {
                buffer: self.get_zerod_buffer(size),
                rows,
                cols,
            };
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
        self.sync();
        Matrix { buffer, rows, cols }
    }

    fn create_matrix(&self, buffer: DeviceBuffer<f32>, rows: usize, cols: usize) -> Matrix {
        Matrix { buffer, rows, cols }
    }

    pub fn vector_zip(&self, vecs: &[Vector]) -> Matrix {
        let spans = vecs.iter().map(|v| v.as_span()).collect::<Vec<_>>();
        let row = spans.len();
        let col = spans[0].len();
        let buffer = self.concat_buffers_from_span(&spans);

        self.create_matrix(buffer, row, col)
    }

    pub fn split_view<'a>(&self, matrix: &'a mut Matrix) -> Vec<VectorView<'a>> {
        DeviceSpanMut::chunks(&mut matrix.buffer, matrix.cols)
            .into_iter()
            .map(VectorView::new)
            .collect()
    }

    pub fn matrix_split(&self, matrix: &mut Matrix) -> Vec<Vector> {
        DeviceSpanMut::chunks(&mut matrix.buffer, matrix.cols)
            .into_iter()
            .map(|span| self.create_vector(span.to_buffer(self)))
            .collect()
    }

    pub fn broadcast(&self, vector: &Vector, copies: usize) -> Matrix {
        let spans = vec![vector.as_span(); copies];
        let buffer = self.concat_buffers_from_span(&spans);
        self.create_matrix(buffer, copies, vector.len())
    }

    pub fn extract_vector(&self, matrix: Matrix) -> Vector {
        assert!(
            matrix.rows > 0,
            "cannot extract a vector from an empty matrix"
        );

        if matrix.rows == 1 {
            return self.create_vector(matrix.buffer);
        }

        let span = DeviceSpan::from_buffer(&matrix.buffer, 0, matrix.cols);
        self.create_vector(span.to_buffer(self))
    }

    pub fn matrix_slice(&self, matrix: &Matrix, cols: usize, rows: usize) -> Vec<Matrix> {
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

    pub fn to_vector(&self, matrix: Matrix) -> Vector {
        self.create_vector(matrix.buffer)
    }
}
