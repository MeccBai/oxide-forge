use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig1D};

use crate::cuda::{
    DEFAULT_BLOCK_SIZE, DeviceSpan, DeviceSpanMut,
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

    pub fn sum(&self, runtime: &mut CudaRuntime) -> f32 {
        self.map_reduce(runtime, 0.0, move |value| value, move |lhs, rhs| lhs + rhs)
    }

    pub fn max(&self, runtime: &mut CudaRuntime) -> f32 {
        self.map_reduce(
            runtime,
            f32::NEG_INFINITY,
            move |value| value,
            move |lhs, rhs| lhs.max(rhs),
        )
    }

    pub fn map_sum<F>(&self, runtime: &mut CudaRuntime, map: F) -> f32
    where
        F: Fn(f32) -> f32 + Copy,
    {
        self.map_reduce(runtime, 0.0, map, move |lhs, rhs| lhs + rhs)
    }

    pub fn map_reduce<FM, FR>(
        &self,
        runtime: &mut CudaRuntime,
        identity: f32,
        map: FM,
        reduce: FR,
    ) -> f32
    where
        FM: Fn(f32) -> f32 + Copy,
        FR: Fn(f32, f32) -> f32 + Copy,
    {
        DeviceSpan::from_buffer(&self.buffer, 0, self.buffer.len())
            .map_reduce(runtime, identity, map, reduce)
    }

    pub fn zip_map_reduce<FM, FR>(
        &self,
        rhs: &Matrix,
        runtime: &mut CudaRuntime,
        identity: f32,
        map: FM,
        reduce: FR,
    ) -> f32
    where
        FM: Fn(f32, f32) -> f32 + Copy,
        FR: Fn(f32, f32) -> f32 + Copy,
    {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        let lhs = DeviceSpan::from_buffer(&self.buffer, 0, self.buffer.len());
        let rhs = DeviceSpan::from_buffer(&rhs.buffer, 0, rhs.buffer.len());
        lhs.zip_map_reduce(&rhs, runtime, identity, map, reduce)
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

    pub fn threshold(&mut self, threshold: f32, runtime: &CudaRuntime) {
        assert!(threshold.is_finite() && (0.0..1.0).contains(&threshold));
        self.for_each(
            runtime,
            move |value| {
                if value >= threshold { 1.0 } else { 0.0 }
            },
        );
    }

    /// Applies the numerically stable logistic sigmoid in place.
    pub fn sigmoid(&mut self, runtime: &CudaRuntime) {
        self.sigmoid_on(runtime, runtime.stream());
    }

    pub(crate) fn sigmoid_on(&mut self, runtime: &CudaRuntime, stream: &CudaStream) {
        self.for_each_on(runtime, stream, crate::cuda::sigmoid_f32);
    }

    pub fn binary_assign(
        &mut self,
        rhs: &Matrix,
        f: impl Fn(f32, f32) -> f32 + Copy,
        runtime: &CudaRuntime,
    ) {
        self.binary_assign_on(rhs, f, runtime, runtime.stream());
    }

    pub(crate) fn binary_assign_on(
        &mut self,
        rhs: &Matrix,
        f: impl Fn(f32, f32) -> f32 + Copy,
        runtime: &CudaRuntime,
        stream: &CudaStream,
    ) {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        let len = self.buffer.len();
        let (config, elements_per_thread) =
            runtime.get_elementwise_launch_config(len, DEFAULT_BLOCK_SIZE);
        let prepared = runtime
            .module()
            .prepare_slice_binary_assign(config)
            .unwrap();
        let target = DeviceSpanMut::from_buffer(&mut self.buffer, 0, len);
        let rhs = DeviceSpan::from_buffer(&rhs.buffer, 0, len);
        runtime
            .module()
            .slice_binary_assign(
                stream,
                &prepared,
                target.descriptor(),
                rhs.descriptor(),
                elements_per_thread,
                f,
            )
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
            .matrix_causal_mask(
                runtime.stream(),
                &prepared,
                matrix.descriptor(),
                self.cols,
                self.rows,
            )
            .unwrap();
    }

    pub(crate) fn causal_mask_heads(&mut self, head_rows: usize, runtime: &CudaRuntime) {
        assert!(head_rows > 0 && self.rows % head_rows == 0);
        assert_eq!(self.cols, head_rows);
        let config = LaunchConfig1D::new(self.rows as u32, self.cols as u32, 0);
        let prepared = runtime.module().prepare_matrix_causal_mask(config).unwrap();
        let len = self.buffer.len();
        let matrix = DeviceSpanMut::from_buffer(&mut self.buffer, 0, len);
        runtime
            .module()
            .matrix_causal_mask(
                runtime.stream(),
                &prepared,
                matrix.descriptor(),
                self.cols,
                head_rows,
            )
            .unwrap();
    }

    pub fn rope_encoding(&mut self, runtime: &CudaRuntime) {
        if self.rows == 0 {
            return;
        }
        assert!(self.cols > 0 && self.cols % 2 == 0);
        let config = LaunchConfig1D::new(self.rows as u32, self.cols as u32 / 2, 0);
        let prepared = runtime
            .module()
            .prepare_matrix_rope_encoding(config)
            .unwrap();
        let len = self.buffer.len();
        let matrix = DeviceSpanMut::from_buffer(&mut self.buffer, 0, len);
        runtime
            .module()
            .matrix_rope_encoding(runtime.stream(), &prepared, matrix.descriptor(), self.cols)
            .unwrap();
    }
}

impl CudaRuntime {
    pub fn matrix_from_host(
        &self,
        values: &[f32],
        rows: usize,
        cols: usize,
        stream: Option<&CudaStream>,
    ) -> Result<Matrix, DriverError> {
        let len = rows
            .checked_mul(cols)
            .expect("matrix element count overflow");
        assert_eq!(values.len(), len);
        let stream = self.execution_stream(stream);
        Ok(Matrix {
            buffer: DeviceBuffer::from_host(stream, values)?,
            rows,
            cols,
        })
    }

    pub fn matrix_multiply(
        &mut self,
        mat1: &Matrix,
        mat2: &Matrix,
        stream: Option<&CudaStream>,
    ) -> Matrix {
        let mut result = self.new_uninit_matrix(mat1.rows, mat2.cols);
        let stream = self.execution_stream(stream);
        self.matrix_multiply_into_on(stream, mat1, mat2, &mut result);
        result
    }

    pub fn matrix_multiply_into(
        &self,
        mat1: &Matrix,
        mat2: &Matrix,
        result: &mut Matrix,
        stream: Option<&CudaStream>,
    ) {
        let stream = self.execution_stream(stream);
        self.matrix_multiply_into_on(stream, mat1, mat2, result);
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

        let block_rows = rows.div_ceil(32);
        let block_cols = cols.div_ceil(32);
        let config = LaunchConfig1D::new((block_rows * block_cols) as u32, 128, 0);
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn matrix_multiply_batched_strided_into(
        &self,
        mat1: &Matrix,
        mat2: &Matrix,
        result: &mut Matrix,
        inner: usize,
        rows: usize,
        cols: usize,
        batch_count: usize,
        a_offset: usize,
        a_row_stride: usize,
        a_batch_stride: usize,
        b_offset: usize,
        b_row_stride: usize,
        b_batch_stride: usize,
        result_offset: usize,
        result_row_stride: usize,
        result_batch_stride: usize,
    ) {
        assert!(rows > 0 && cols > 0 && inner > 0 && batch_count > 0);
        assert_eq!(rows % 16, 0);
        assert_eq!(cols % 16, 0);
        assert_eq!(inner % 16, 0);
        let last_batch = batch_count - 1;
        assert!(
            a_offset + last_batch * a_batch_stride + (rows - 1) * a_row_stride + inner
                <= mat1.buffer.len()
        );
        assert!(
            b_offset + last_batch * b_batch_stride + (inner - 1) * b_row_stride + cols
                <= mat2.buffer.len()
        );
        assert!(
            result_offset
                + last_batch * result_batch_stride
                + (rows - 1) * result_row_stride
                + cols
                <= result.buffer.len()
        );

        let blocks_per_batch = rows.div_ceil(32) * cols.div_ceil(32);
        let config = LaunchConfig1D::new((batch_count * blocks_per_batch) as u32, 128, 0);
        let prepared = self
            .module()
            .prepare_matrix_multiply_batched_strided(config)
            .unwrap();
        let lhs = DeviceSpan::from_buffer(&mat1.buffer, 0, mat1.buffer.len());
        let rhs = DeviceSpan::from_buffer(&mat2.buffer, 0, mat2.buffer.len());
        let result_len = result.buffer.len();
        let output = DeviceSpanMut::from_buffer(&mut result.buffer, 0, result_len);

        self.module()
            .matrix_multiply_batched_strided(
                self.stream(),
                &prepared,
                lhs.descriptor(),
                rhs.descriptor(),
                output.descriptor(),
                inner,
                rows,
                cols,
                batch_count,
                a_offset,
                a_row_stride,
                a_batch_stride,
                b_offset,
                b_row_stride,
                b_batch_stride,
                result_offset,
                result_row_stride,
                result_batch_stride,
            )
            .unwrap();
    }

    pub fn matrix_add(
        &mut self,
        mat1: &Matrix,
        mat2: &Matrix,
        stream: Option<&CudaStream>,
    ) -> Matrix {
        self.matrix_binary(mat1, mat2, move |lhs, rhs| lhs + rhs, stream)
    }

    pub fn matrix_sub(
        &mut self,
        mat1: &Matrix,
        mat2: &Matrix,
        stream: Option<&CudaStream>,
    ) -> Matrix {
        self.matrix_binary(mat1, mat2, move |lhs, rhs| lhs - rhs, stream)
    }

    pub fn matrix_mul(
        &mut self,
        mat1: &Matrix,
        mat2: &Matrix,
        stream: Option<&CudaStream>,
    ) -> Matrix {
        self.matrix_binary(mat1, mat2, move |lhs, rhs| lhs * rhs, stream)
    }

    pub fn matrix_div(
        &mut self,
        mat1: &Matrix,
        mat2: &Matrix,
        stream: Option<&CudaStream>,
    ) -> Matrix {
        self.matrix_binary(mat1, mat2, move |lhs, rhs| lhs / rhs, stream)
    }

    pub fn matrix_binary(
        &mut self,
        mat1: &Matrix,
        mat2: &Matrix,
        f: impl Fn(f32, f32) -> f32 + Copy,
        stream: Option<&CudaStream>,
    ) -> Matrix {
        assert_eq!(mat1.rows, mat2.rows);
        assert_eq!(mat1.cols, mat2.cols);

        let rows = mat1.rows;
        let cols = mat1.cols;
        let mut result_buffer = self.get_uninit_buffer(rows * cols);

        let (config, elements_per_thread) =
            self.get_elementwise_launch_config(mat1.buffer.len(), DEFAULT_BLOCK_SIZE);
        let prepared = self.module().prepare_slice_binary(config).unwrap();
        let lhs = DeviceSpan::from_buffer(&mat1.buffer, 0, mat1.buffer.len());
        let rhs = DeviceSpan::from_buffer(&mat2.buffer, 0, mat2.buffer.len());
        let output = DeviceSpanMut::from_buffer(&mut result_buffer, 0, rows * cols);
        let stream = self.execution_stream(stream);

        self.module()
            .slice_binary(
                stream,
                &prepared,
                lhs.descriptor(),
                rhs.descriptor(),
                output.descriptor(),
                elements_per_thread,
                f,
            )
            .unwrap();
        self.create_matrix(result_buffer, rows, cols)
    }

    /// Consumes a matrix and returns its allocation to the runtime pool.
    pub fn recycle_matrix(&mut self, matrix: Matrix) {
        self.recycle_buffer(matrix.buffer);
    }

    pub fn new_matrix(
        &mut self,
        init_type: InitType,
        rows: usize,
        cols: usize,
        stream: Option<&CudaStream>,
    ) -> Matrix {
        let size = rows * cols;
        if init_type.is_zero() {
            let mut buffer = self.get_uninit_buffer(size);
            let stream = self.execution_stream(stream);
            buffer.zero_async(stream).unwrap();
            return self.create_matrix(buffer, rows, cols);
        }
        let mut buffer = self.get_uninit_buffer(size);
        let (config, elements_per_thread) =
            self.get_elementwise_launch_config(buffer.len(), DEFAULT_BLOCK_SIZE);
        let stream = self.execution_stream(stream);
        match init_type {
            InitType::Sequence => {
                let prepared = self.module().prepare_slice_set_seq(config).unwrap();
                let span = DeviceSpanMut::from_buffer(&mut buffer, 0, size);
                self.module()
                    .slice_set_seq(
                        stream,
                        &prepared,
                        span.descriptor(),
                        elements_per_thread,
                        true,
                        0.0,
                        1.0,
                    )
                    .unwrap();
            }
            InitType::Reserve => {
                let prepared = self.module().prepare_slice_set_seq(config).unwrap();
                let span = DeviceSpanMut::from_buffer(&mut buffer, 0, size);
                self.module()
                    .slice_set_seq(
                        stream,
                        &prepared,
                        span.descriptor(),
                        elements_per_thread,
                        false,
                        0.0,
                        1.0,
                    )
                    .unwrap();
            }
            InitType::Random => {
                let seed = rand::random();
                let prepared = self.module().prepare_slice_set_random(config).unwrap();
                let span = DeviceSpanMut::from_buffer(&mut buffer, 0, size);
                self.module()
                    .slice_set_random(
                        stream,
                        &prepared,
                        span.descriptor(),
                        elements_per_thread,
                        seed,
                    )
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

    pub fn clone_matrix(&mut self, matrix: &Matrix, stream: Option<&CudaStream>) -> Matrix {
        let mut buffer = self.get_uninit_buffer(matrix.buffer.len());
        let stream = self.execution_stream(stream);
        buffer
            .copy_from_device_async(&matrix.buffer, stream)
            .unwrap();
        self.create_matrix(buffer, matrix.rows, matrix.cols)
    }
}
