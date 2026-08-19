mod convert;
mod matrix;
mod norm;
mod rows;
mod vector;
mod vector_runtime;
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
