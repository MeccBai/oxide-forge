use cuda_core::DeviceBuffer;

use crate::cuda::{
    DEFAULT_BLOCK_SIZE, DeviceSpan, DeviceSpanMut, runtime::CudaRuntime, runtime::InitType,
};

use crate::cuda::vector::{Vector, VectorView};

pub struct Matrix {
    buffer: DeviceBuffer<f32>,
    rows: usize,
    cols: usize,
}

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
        let prepared = self.module().prepare_vec_set_seq(config).unwrap();
        match init_type {
            InitType::Sequence => {
                self.module()
                    .vec_set_seq(self.stream(), &prepared, &mut buffer, true)
                    .unwrap();
            }
            InitType::Reserve => {
                self.module()
                    .vec_set_seq(self.stream(), &prepared, &mut buffer, false)
                    .unwrap();
            }
            InitType::Random => {
                let seed = rand::random();
                let prepared = self.module().prepare_vector_set_random(config).unwrap();
                self.module()
                    .vector_set_random(self.stream(), &prepared, &mut buffer, seed)
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

    pub fn split(&self, matrix: &mut Matrix) -> Vec<Vector> {
        DeviceSpanMut::chunks(&mut matrix.buffer, matrix.cols)
            .into_iter()
            .map(|span| self.create_vector(span.to_buffer(self)))
            .collect()
    }

    pub fn broadcast(&self, vector: &Vector, copies: usize) -> Matrix {
        let spans = vec![vector.as_span(); copies];
        let buffer = self.concat_buffers_from_span(&spans);
        self.create_matrix(buffer, copies, vector.buffer_len())
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
}
