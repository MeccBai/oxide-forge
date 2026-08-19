use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};

use crate::cuda::{
    BinaryOp, DEFAULT_BLOCK_SIZE, DeviceSpan, DeviceSpanMut, runtime::CudaRuntime,
    runtime::InitType,
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
        if self.rows == 0 {
            return;
        }
        assert!(self.cols > 0 && self.cols <= DEFAULT_BLOCK_SIZE);
        let config = LaunchConfig1D::new(self.rows as u32, DEFAULT_BLOCK_SIZE as u32, 0);
        let prepared = runtime
            .module()
            .prepare_matrix_softmax_rows(config)
            .unwrap();
        let len = self.buffer.len();
        let matrix = DeviceSpanMut::from_buffer(&mut self.buffer, 0, len);
        runtime
            .module()
            .matrix_softmax_rows(runtime.stream(), &prepared, matrix.descriptor(), self.cols)
            .unwrap();
    }

    pub fn layer_norm(&mut self, runtime: &CudaRuntime) {
        if self.rows == 0 {
            return;
        }
        assert!(self.cols > 0 && self.cols <= DEFAULT_BLOCK_SIZE);
        let config = LaunchConfig1D::new(self.rows as u32, DEFAULT_BLOCK_SIZE as u32, 0);
        let prepared = runtime
            .module()
            .prepare_matrix_layer_norm_rows(config)
            .unwrap();
        let len = self.buffer.len();
        let matrix = DeviceSpanMut::from_buffer(&mut self.buffer, 0, len);
        runtime
            .module()
            .matrix_layer_norm_rows(
                runtime.stream(),
                &prepared,
                matrix.descriptor(),
                self.cols,
                1e-5,
            )
            .unwrap();
    }

    pub fn for_each<F>(&mut self, runtime: &CudaRuntime, f: F)
    where
        F: Fn(f32) -> f32 + Copy,
    {
        self.for_each_on(runtime, runtime.stream(), f);
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
    /// Consumes a matrix and returns its allocation to the runtime pool.
    pub fn recycle_matrix(&mut self, matrix: Matrix) {
        self.recycle_buffer(matrix.buffer);
    }

    pub fn matrix_sum_rows(&mut self, matrix: &Matrix) -> Vector {
        if matrix.rows == 0 {
            let buffer = self.get_uninit_buffer(0);
            return self.create_vector(buffer);
        }
        assert!(matrix.cols > 0 && matrix.cols <= DEFAULT_BLOCK_SIZE);
        let mut buffer = self.get_uninit_buffer(matrix.rows);
        let config = LaunchConfig1D::new(matrix.rows as u32, DEFAULT_BLOCK_SIZE as u32, 0);
        let prepared = self.module().prepare_matrix_sum_rows(config).unwrap();
        let input = DeviceSpan::from_buffer(&matrix.buffer, 0, matrix.buffer.len());
        let result_len = buffer.len();
        let result = DeviceSpanMut::from_buffer(&mut buffer, 0, result_len);
        self.module()
            .matrix_sum_rows(
                self.stream(),
                &prepared,
                input.descriptor(),
                result.descriptor(),
                matrix.cols,
            )
            .unwrap();
        self.create_vector(buffer)
    }

    pub fn softmax_rows_backward(
        &mut self,
        probabilities: &Matrix,
        output_gradient: &Matrix,
    ) -> Matrix {
        assert_eq!(probabilities.rows, output_gradient.rows);
        assert_eq!(probabilities.cols, output_gradient.cols);
        assert!(probabilities.cols <= DEFAULT_BLOCK_SIZE);
        let mut buffer = self.get_uninit_buffer(probabilities.rows * probabilities.cols);
        let config = LaunchConfig1D::new(probabilities.rows as u32, DEFAULT_BLOCK_SIZE as u32, 0);
        let prepared = self.module().prepare_softmax_rows_backward(config).unwrap();
        let probabilities_span =
            DeviceSpan::from_buffer(&probabilities.buffer, 0, probabilities.buffer.len());
        let output_gradient_span =
            DeviceSpan::from_buffer(&output_gradient.buffer, 0, output_gradient.buffer.len());
        let len = buffer.len();
        let result = DeviceSpanMut::from_buffer(&mut buffer, 0, len);
        self.module()
            .softmax_rows_backward(
                self.stream(),
                &prepared,
                probabilities_span.descriptor(),
                output_gradient_span.descriptor(),
                result.descriptor(),
                probabilities.cols,
            )
            .unwrap();
        self.create_matrix(buffer, probabilities.rows, probabilities.cols)
    }

    pub fn layer_norm_backward(&mut self, input: &Matrix, output_gradient: &Matrix) -> Matrix {
        assert_eq!(input.rows, output_gradient.rows);
        assert_eq!(input.cols, output_gradient.cols);

        assert!(input.cols <= DEFAULT_BLOCK_SIZE);
        let mut buffer = self.get_uninit_buffer(input.rows * input.cols);
        let config = LaunchConfig1D::new(input.rows as u32, DEFAULT_BLOCK_SIZE as u32, 0);
        let prepared = self.module().prepare_layer_norm_backward(config).unwrap();
        let input_span = DeviceSpan::from_buffer(&input.buffer, 0, input.buffer.len());
        let output_gradient_span =
            DeviceSpan::from_buffer(&output_gradient.buffer, 0, output_gradient.buffer.len());
        let len = buffer.len();
        let result = DeviceSpanMut::from_buffer(&mut buffer, 0, len);
        self.module()
            .layer_norm_backward(
                self.stream(),
                &prepared,
                input_span.descriptor(),
                output_gradient_span.descriptor(),
                result.descriptor(),
                input.cols,
                1e-5,
            )
            .unwrap();
        self.create_matrix(buffer, input.rows, input.cols)
    }

    pub fn new_matrix(&mut self, init_type: InitType, rows: usize, cols: usize) -> Matrix {
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

    pub fn vector_zip(&mut self, vecs: &[Vector]) -> Matrix {
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

    pub fn matrix_split(&mut self, matrix: &mut Matrix) -> Vec<Vector> {
        let spans = DeviceSpanMut::chunks(&mut matrix.buffer, matrix.cols);
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

    pub fn to_vector(&self, matrix: Matrix) -> Vector {
        self.create_vector(matrix.buffer)
    }

    pub fn matrix_copy(&mut self, matrix: &Matrix) -> Matrix {
        let buffer = self.clone_buffer(&matrix.buffer);
        self.create_matrix(buffer, matrix.rows, matrix.cols)
    }
}
