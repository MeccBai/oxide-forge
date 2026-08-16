use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D, PreparedLaunch};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, shared, thread, warp};
use cuda_host::cuda_module;

const DEFAULT_BLOCK_SIZE: usize = 1024;

mod device;
pub mod matrix;
pub mod runtime;
pub mod vector;
//mod tensor;

mod span;

pub use runtime::CudaRuntime;
pub use runtime::InitType;
pub(crate) use span::{DeviceSpan, DeviceSpanMut};

#[cuda_module]
mod kernels {
    const DEFAULT_BLOCK_SIZE_U32: u32 = 1024;

    use super::*;

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1)]
    pub fn vec_set(mut buffer: DisjointSlice<f32>, value: f32) {
        let idx = thread::index_1d();
        if let Some(elem) = buffer.get_mut(idx) {
            *elem = value;
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn vec_set_seq(mut buffer: DisjointSlice<f32>, dir: bool) {
        let idx = thread::index_1d();
        let max = buffer.len();
        let val = if dir {
            idx.get() as f32
        } else {
            (max - idx.get()) as f32
        };
        if let Some(elem) = buffer.get_mut(idx) {
            *elem = val;
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn vector_add(buffer1: &[f32], buffer2: &[f32], mut result: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let index = idx.get();
        if let Some(elem) = result.get_mut(idx) {
            *elem = buffer1[index] + buffer2[index];
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn vector_sum(buffer: &[f32], mut result: DisjointSlice<f32>) {
        let value = device::vector_add_device(buffer);
        if thread::threadIdx_x() == 0 {
            let block_id = thread::blockIdx_x() as usize;
            if block_id < result.len() {
                unsafe {
                    *result.get_unchecked_mut(block_id) = value;
                }
            }
        }
    }

    #[kernel]
    #[launch_bounds(1024)]
    #[launch_contract(domain = 1)]
    pub fn vector_for_each<F>(mut buffer: DisjointSlice<f32>, f: F)
    where
        F: Fn(f32) -> f32 + Copy,
    {
        let idx = thread::index_1d();

        if let Some(element) = buffer.get_mut(idx) {
            *element = f(*element);
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn span_for_each<F>(span: span::DeviceSpanDescriptor<f32>, f: F)
    where
        F: Fn(f32) -> f32 + Copy,
    {
        let index = thread::index_1d().get();
        if index < span.len {
            let element = unsafe { span.ptr.add(index) };
            unsafe {
                element.write(f(element.read()));
            }
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub unsafe fn span_sum(span: span::DeviceSpanDescriptor<f32>, mut result: DisjointSlice<f32>) {
        static mut SHARED: shared::SharedArray<f32, 32> = shared::SharedArray::UNINIT;

        let index = thread::index_1d().get();
        let lane = thread::threadIdx_x() as usize % 32;
        let warp_id = thread::threadIdx_x() as usize / 32;
        let mut value = if index < span.len {
            unsafe { span.ptr.add(index).read() }
        } else {
            0.0
        };

        for delta in [1, 2, 4, 8, 16] {
            value += warp::shuffle_down_f32(value, delta);
        }
        if lane == 0 {
            unsafe { SHARED[warp_id] = value };
        }
        thread::sync_threads();

        if warp_id == 0 {
            value = unsafe { SHARED[lane] };
            for delta in [1, 2, 4, 8, 16] {
                value += warp::shuffle_down_f32(value, delta);
            }
        }

        if thread::threadIdx_x() == 0 {
            let block_id = thread::blockIdx_x() as usize;
            if block_id < result.len() {
                unsafe { *result.get_unchecked_mut(block_id) = value };
            }
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn span_map_sum<F>(
        span: span::DeviceSpanDescriptor<f32>,
        mut result: DisjointSlice<f32>,
        f: F,
    ) where
        F: Fn(f32) -> f32 + Copy,
    {
        static mut SHARED: shared::SharedArray<f32, 32> = shared::SharedArray::UNINIT;

        let index = thread::index_1d().get();
        let lane = thread::threadIdx_x() as usize % 32;
        let warp_id = thread::threadIdx_x() as usize / 32;

        let mut value = if index < span.len {
            let x = unsafe { span.ptr.add(index).read() };
            f(x)
        } else {
            0.0
        };

        for delta in [1, 2, 4, 8, 16] {
            value += warp::shuffle_down_f32(value, delta);
        }

        if lane == 0 {
            unsafe {
                SHARED[warp_id] = value;
            }
        }

        thread::sync_threads();

        if warp_id == 0 {
            value = unsafe { SHARED[lane] };

            for delta in [1, 2, 4, 8, 16] {
                value += warp::shuffle_down_f32(value, delta);
            }
        }

        if thread::threadIdx_x() == 0 {
            let block_id = thread::blockIdx_x() as usize;

            if block_id < result.len() {
                unsafe {
                    *result.get_unchecked_mut(block_id) = value;
                }
            }
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub unsafe fn span_max(span: span::DeviceSpanDescriptor<f32>, mut result: DisjointSlice<f32>) {
        static mut SHARED: shared::SharedArray<f32, 32> = shared::SharedArray::UNINIT;

        let index = thread::index_1d().get();
        let lane = thread::threadIdx_x() as usize % 32;
        let warp_id = thread::threadIdx_x() as usize / 32;
        let mut value = if index < span.len {
            unsafe { span.ptr.add(index).read() }
        } else {
            f32::MIN
        };

        for delta in [1, 2, 4, 8, 16] {
            value = value.max(warp::shuffle_down_f32(value, delta));
        }
        if lane == 0 {
            unsafe { SHARED[warp_id] = value };
        }
        thread::sync_threads();

        if warp_id == 0 {
            value = unsafe { SHARED[lane] };
            for delta in [1, 2, 4, 8, 16] {
                value = value.max(warp::shuffle_down_f32(value, delta));
            }
        }

        if thread::threadIdx_x() == 0 {
            let block_id = thread::blockIdx_x() as usize;
            if block_id < result.len() {
                unsafe { *result.get_unchecked_mut(block_id) = value };
            }
        }
    }

    #[kernel]
    #[launch_bounds(1024)]
    #[launch_contract(domain = 1)]
    pub fn vector_set_random(mut buffer: DisjointSlice<f32>, seed: u32) {
        let idx = thread::index_1d();
        let val = idx.get() as u32;
        if let Some(elem) = buffer.get_mut(idx) {
            let rand = device::random(seed + val as u32);
            *elem = (rand as f32) / (u32::MAX as f32);
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn vector_max(buffer: &[f32], mut result: DisjointSlice<f32>) {
        let value = device::vector_max_device(buffer);
        if thread::threadIdx_x() == 0 {
            let block_id = thread::blockIdx_x() as usize;
            if block_id < result.len() {
                unsafe {
                    *result.get_unchecked_mut(block_id) = value;
                }
            }
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn copy_slice(buffer: &[f32], mut result: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let index = idx.get();
        if let Some(elem) = result.get_mut(idx) {
            *elem = buffer[index];
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn vector_pre_dot_product(
        buffer1: &[f32],
        buffer2: &[f32],
        mut result: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let index = idx.get();
        if let Some(elem) = result.get_mut(idx) {
            *elem = buffer1[index] * buffer2[index];
        }
    }

    const TILE_SIZE: usize = 16;
    const SHARED_SIZE: usize = TILE_SIZE * TILE_SIZE;
    const TRANSPOSE_STRIDE: usize = TILE_SIZE + 1;
    const TRANSPOSE_SHARED_SIZE: usize = TILE_SIZE * TRANSPOSE_STRIDE;

    #[kernel]
    #[launch_bounds(256)] // 16 × 16 = 256 threads
    #[launch_contract(domain = 2, block = (16, 16, 1))]
    pub fn matrix_multiply(
        matrix1: &[f32],
        matrix2: &[f32],
        mut result: DisjointSlice<f32, thread::Runtime2DIndex>,
        len: usize,
        rows: usize,
        cols: usize,
    ) {
        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;

        let c_row = thread::blockIdx_y() as usize * TILE_SIZE + ty;
        let c_col = thread::blockIdx_x() as usize * TILE_SIZE + tx;

        let mut shared_matrix1 = shared::SharedArray::<f32, SHARED_SIZE>::UNINIT;
        let mut shared_matrix2 = shared::SharedArray::<f32, SHARED_SIZE>::UNINIT;

        let mut sum = 0.0;

        for t in 0..len / TILE_SIZE {
            shared_matrix1[ty * TILE_SIZE + tx] = matrix1[c_row * len + t * TILE_SIZE + tx];
            shared_matrix2[ty * TILE_SIZE + tx] = matrix2[(t * TILE_SIZE + ty) * cols + c_col];

            thread::sync_threads();

            for k in 0..TILE_SIZE {
                sum += shared_matrix1[ty * TILE_SIZE + k] * shared_matrix2[k * TILE_SIZE + tx]
            }

            thread::sync_threads();
        }

        if c_row < rows {
            if let Some(c_index) = thread::index_2d_runtime(&result) {
                if let Some(element) = result.get_mut(c_index) {
                    *element = sum;
                }
            }
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, block = (16, 16, 1))]
    pub fn matrix_transpose(
        matrix: &[f32],
        mut result: DisjointSlice<f32, thread::Runtime2DIndex>,
        input_rows: usize,
        input_cols: usize,
    ) {
        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let input_row = thread::blockIdx_y() as usize * TILE_SIZE + ty;
        let input_col = thread::blockIdx_x() as usize * TILE_SIZE + tx;

        static mut TILE: shared::SharedArray<f32, TRANSPOSE_SHARED_SIZE> =
            shared::SharedArray::UNINIT;
        let shared_index = ty * TRANSPOSE_STRIDE + tx;

        unsafe {
            TILE[shared_index] = if input_row < input_rows && input_col < input_cols {
                matrix[input_row * input_cols + input_col]
            } else {
                0.0
            };
        }

        thread::sync_threads();

        let output_row = thread::blockIdx_x() as usize * TILE_SIZE + ty;
        let output_col = thread::blockIdx_y() as usize * TILE_SIZE + tx;

        if output_row < input_cols && output_col < input_rows {
            let output_index = output_row * input_rows + output_col;

            // SAFETY: the bounds above prove the linear index is valid. The
            // mapping from (block, thread) to output_index is one-to-one.
            unsafe {
                *result.get_unchecked_mut(output_index) = TILE[tx * TRANSPOSE_STRIDE + ty];
            }
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn matrix_slice(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        input_cols: usize,
        tile_rows: usize,
        tile_cols: usize,
        tiles_per_row: usize,
    ) {
        let index = thread::index_1d();
        let linear = index.get();

        if let Some(output_element) = output.get_mut(index) {
            let tile_size = tile_rows * tile_cols;

            let tile_index = linear / tile_size;
            let tile_local = linear % tile_size;

            let tile_y = tile_index / tiles_per_row;
            let tile_x = tile_index % tiles_per_row;

            let local_row = tile_local / tile_cols;
            let local_col = tile_local % tile_cols;

            let input_row = tile_y * tile_rows + local_row;
            let input_col = tile_x * tile_cols + local_col;

            *output_element = input[input_row * input_cols + input_col];
        }
    }
}
