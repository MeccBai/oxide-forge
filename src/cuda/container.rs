mod matrix;
mod matrix_compute;
mod vector;
mod vector_compute;
mod vector_view;

use cuda_core::DeviceBuffer;

use crate::cuda::DeviceSpanMut;

pub struct Vector {
    buffer: DeviceBuffer<f32>,
}

pub struct VectorView<'a> {
    span: DeviceSpanMut<'a, f32>,
}

pub struct Matrix {
    buffer: DeviceBuffer<f32>,
    rows: usize,
    cols: usize,
}
