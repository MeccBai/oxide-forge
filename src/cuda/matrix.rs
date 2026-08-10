use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

use crate::cuda::{CudaRuntime, DEFAULT_BLOCK_SIZE, DeviceSpanMut, InitType};

use crate::cuda::vector::{Vector, VectorView};

pub struct Matrix {
    buffer: DeviceBuffer<f32>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    pub fn to_host(&self, runtime: &CudaRuntime) -> Vec<f32> {
        self.buffer.to_host_vec(&runtime.stream).unwrap()
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
        let prepared = self.module.prepare_vec_set_seq(config).unwrap();
        match init_type {
            InitType::Sequence => {
                self.module
                    .vec_set_seq(&self.stream, &prepared, &mut buffer, true)
                    .unwrap();
            }
            InitType::Reserve => {
                self.module
                    .vec_set_seq(&self.stream, &prepared, &mut buffer, false)
                    .unwrap();
            }
            InitType::Random => {
                let seed = rand::random();
                let prepared = self.module.prepare_vector_set_random(config).unwrap();
                self.module
                    .vector_set_random(&self.stream, &prepared, &mut buffer, seed)
                    .unwrap();
            }
            InitType::Zero => {}
        }
        self.sync();
        Matrix { buffer, rows, cols }
    }

    pub fn vector_zip(&self, vecs: &[Vector], by_col: bool) {
        let vector_len = vecs[0].buffer_len();

        let buffer = self.get_uninit_buffer(vecs.len() * vector_len);
    }

    pub fn split_view<'a>(&self, matrix: &'a mut Matrix) -> Vec<VectorView<'a>> {
        DeviceSpanMut::split_contiguous(&mut matrix.buffer, matrix.rows, matrix.cols)
            .into_iter()
            .map(VectorView::new)
            .collect()
    }

    /* pub fn split(&self, matrix: &Matrix) -> Vec<Vector> {
        let mut vecs = Vec::new();
        let len = matrix.cols;
        for i in 0..len {
            let offset = i * matrix.rows;
            let vec = Vector {
                buffer: self.clone_buffer_slice(&matrix.buffer, offset, matrix.rows),
            };
            vecs.push(vec);
        }
        vecs
    } */
}
