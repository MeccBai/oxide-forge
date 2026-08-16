use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, shared, thread, warp};
use cuda_host::cuda_module;

const DEFAULT_BLOCK_SIZE: usize = 1024;

pub mod container;
mod device;
pub mod runtime;
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
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn slice_set(target: span::DeviceSliceMutDescriptor<f32>, value: f32) {
        let index = thread::index_1d().get();
        if index < target.len {
            unsafe { target.ptr.add(index).write(value) };
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn slice_set_seq(target: span::DeviceSliceMutDescriptor<f32>, dir: bool) {
        let index = thread::index_1d().get();
        if index < target.len {
            let value = if dir {
                index as f32
            } else {
                (target.len - index) as f32
            };
            unsafe { target.ptr.add(index).write(value) };
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn slice_set_random(target: span::DeviceSliceMutDescriptor<f32>, seed: u32) {
        let index = thread::index_1d().get();
        if index < target.len {
            let rand = device::random(seed + index as u32);
            unsafe {
                target
                    .ptr
                    .add(index)
                    .write((rand as f32) / (u32::MAX as f32));
            }
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn slice_add(
        lhs: span::DeviceSliceDescriptor<f32>,
        rhs: span::DeviceSliceDescriptor<f32>,
        output: span::DeviceSliceMutDescriptor<f32>,
    ) {
        let index = thread::index_1d().get();
        if index < output.len && index < lhs.len && index < rhs.len {
            unsafe {
                output
                    .ptr
                    .add(index)
                    .write(lhs.ptr.add(index).read() + rhs.ptr.add(index).read());
            }
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn slice_add_assign(
        target: span::DeviceSliceMutDescriptor<f32>,
        rhs: span::DeviceSliceDescriptor<f32>,
    ) {
        let index = thread::index_1d().get();
        if index < target.len && index < rhs.len {
            unsafe {
                let element = target.ptr.add(index);
                element.write(element.read() + rhs.ptr.add(index).read());
            }
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn slice_mul(
        lhs: span::DeviceSliceDescriptor<f32>,
        rhs: span::DeviceSliceDescriptor<f32>,
        output: span::DeviceSliceMutDescriptor<f32>,
    ) {
        let index = thread::index_1d().get();
        if index < output.len && index < lhs.len && index < rhs.len {
            unsafe {
                output
                    .ptr
                    .add(index)
                    .write(lhs.ptr.add(index).read() * rhs.ptr.add(index).read());
            }
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn slice_for_each<F>(span: span::DeviceSliceMutDescriptor<f32>, f: F)
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
    pub unsafe fn slice_sum(
        span: span::DeviceSliceDescriptor<f32>,
        result: span::DeviceSliceMutDescriptor<f32>,
    ) {
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
            if block_id < result.len {
                unsafe { result.ptr.add(block_id).write(value) };
            }
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn slice_map_sum<F>(
        span: span::DeviceSliceDescriptor<f32>,
        result: span::DeviceSliceMutDescriptor<f32>,
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

            if block_id < result.len {
                unsafe { result.ptr.add(block_id).write(value) };
            }
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub unsafe fn slice_max(
        span: span::DeviceSliceDescriptor<f32>,
        result: span::DeviceSliceMutDescriptor<f32>,
    ) {
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
            if block_id < result.len {
                unsafe { result.ptr.add(block_id).write(value) };
            }
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
