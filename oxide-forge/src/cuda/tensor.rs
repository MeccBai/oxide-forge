use cuda_core::DeviceBuffer;

use crate::cuda::{
    DEFAULT_BLOCK_SIZE, DeviceSpan, DeviceSpanMut, runtime::CudaRuntime, runtime::InitType,
};

use super::vector::{Vector, VectorView};

use super::matrix::Matrix;

struct Tensor3D {
    buffer: DeviceBuffer<f32>,
    dim1: usize,
    dim2: usize,
    dim3: usize,
}