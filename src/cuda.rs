use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, shared, thread, warp};
use cuda_host::cuda_module;

const DEFAULT_BLOCK_SIZE: usize = 1024;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

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

    #[inline(always)]
    fn apply_binary(lhs: f32, rhs: f32, op: BinaryOp) -> f32 {
        match op {
            BinaryOp::Add => lhs + rhs,
            BinaryOp::Sub => lhs - rhs,
            BinaryOp::Mul => lhs * rhs,
            BinaryOp::Div => lhs / rhs,
        }
    }

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
    pub fn slice_binary(
        lhs: span::DeviceSliceDescriptor<f32>,
        rhs: span::DeviceSliceDescriptor<f32>,
        output: span::DeviceSliceMutDescriptor<f32>,
        op: BinaryOp,
    ) {
        let index = thread::index_1d().get();
        if index < output.len && index < lhs.len && index < rhs.len {
            unsafe {
                output.ptr.add(index).write(apply_binary(
                    lhs.ptr.add(index).read(),
                    rhs.ptr.add(index).read(),
                    op,
                ));
            }
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn slice_binary_assign(
        target: span::DeviceSliceMutDescriptor<f32>,
        rhs: span::DeviceSliceDescriptor<f32>,
        op: BinaryOp,
    ) {
        let index = thread::index_1d().get();
        if index < target.len && index < rhs.len {
            unsafe {
                let element = target.ptr.add(index);
                element.write(apply_binary(element.read(), rhs.ptr.add(index).read(), op));
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

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn matrix_sum_rows(
        matrix: span::DeviceSliceDescriptor<f32>,
        result: span::DeviceSliceMutDescriptor<f32>,
        cols: usize,
    ) {
        static mut SHARED: shared::SharedArray<f32, 32> = shared::SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let warp_id = tid / 32;
        let row = thread::blockIdx_x() as usize;
        let index = row * cols + tid;
        let mut value = if tid < cols && index < matrix.len {
            unsafe { matrix.ptr.add(index).read() }
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

        if tid == 0 && row < result.len {
            unsafe { result.ptr.add(row).write(value) };
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn matrix_softmax_rows(matrix: span::DeviceSliceMutDescriptor<f32>, cols: usize) {
        static mut SHARED: shared::SharedArray<f32, 32> = shared::SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let warp_id = tid / 32;
        let index = thread::blockIdx_x() as usize * cols + tid;
        let active = tid < cols && index < matrix.len;
        let x = if active {
            unsafe { matrix.ptr.add(index).read() }
        } else {
            f32::MIN
        };

        let mut value = x;
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
            if lane == 0 {
                unsafe { SHARED[0] = value };
            }
        }
        thread::sync_threads();
        let row_max = unsafe { SHARED[0] };
        thread::sync_threads();

        value = if active { (x - row_max).exp() } else { 0.0 };
        let exp_value = value;
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
            if lane == 0 {
                unsafe { SHARED[0] = value };
            }
        }
        thread::sync_threads();

        if active {
            let row_sum = unsafe { SHARED[0] };
            unsafe { matrix.ptr.add(index).write(exp_value / row_sum) };
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn matrix_layer_norm_rows(
        matrix: span::DeviceSliceMutDescriptor<f32>,
        cols: usize,
        epsilon: f32,
    ) {
        static mut SHARED: shared::SharedArray<f32, 32> = shared::SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let warp_id = tid / 32;
        let index = thread::blockIdx_x() as usize * cols + tid;
        let active = tid < cols && index < matrix.len;
        let x = if active {
            unsafe { matrix.ptr.add(index).read() }
        } else {
            0.0
        };

        let mut value = x;
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
            if lane == 0 {
                unsafe { SHARED[0] = value / cols as f32 };
            }
        }
        thread::sync_threads();
        let mean = unsafe { SHARED[0] };
        thread::sync_threads();

        let diff = x - mean;
        value = if active { diff * diff } else { 0.0 };
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
            if lane == 0 {
                unsafe { SHARED[0] = 1.0 / (value / cols as f32 + epsilon).sqrt() };
            }
        }
        thread::sync_threads();

        if active {
            let inverse_std = unsafe { SHARED[0] };
            unsafe { matrix.ptr.add(index).write(diff * inverse_std) };
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn matrix_binary_assign_by_rows(
        matrix: span::DeviceSliceMutDescriptor<f32>,
        row: span::DeviceSliceDescriptor<f32>,
        cols: usize,
        op: BinaryOp,
    ) {
        let index = thread::index_1d().get();
        if index < matrix.len {
            let element = unsafe { matrix.ptr.add(index) };
            let rhs = unsafe { row.ptr.add(index % cols).read() };
            unsafe { element.write(apply_binary(element.read(), rhs, op)) };
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn softmax_rows_backward(
        probabilities: span::DeviceSliceDescriptor<f32>,
        output_gradient: span::DeviceSliceDescriptor<f32>,
        result: span::DeviceSliceMutDescriptor<f32>,
        cols: usize,
    ) {
        static mut SHARED: shared::SharedArray<f32, 32> = shared::SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let warp_id = tid / 32;
        let index = thread::blockIdx_x() as usize * cols + tid;
        let active = tid < cols && index < result.len;
        let probability = if active {
            unsafe { probabilities.ptr.add(index).read() }
        } else {
            0.0
        };
        let gradient = if active {
            unsafe { output_gradient.ptr.add(index).read() }
        } else {
            0.0
        };
        let mut value = probability * gradient;
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
            if lane == 0 {
                unsafe { SHARED[0] = value };
            }
        }
        thread::sync_threads();
        if active {
            unsafe {
                result
                    .ptr
                    .add(index)
                    .write(probability * (gradient - SHARED[0]))
            };
        }
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn layer_norm_backward(
        input: span::DeviceSliceDescriptor<f32>,
        output_gradient: span::DeviceSliceDescriptor<f32>,
        result: span::DeviceSliceMutDescriptor<f32>,
        cols: usize,
        epsilon: f32,
    ) {
        static mut SHARED: shared::SharedArray<f32, 32> = shared::SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let warp_id = tid / 32;
        let index = thread::blockIdx_x() as usize * cols + tid;
        let active = tid < cols && index < result.len;
        let x = if active {
            unsafe { input.ptr.add(index).read() }
        } else {
            0.0
        };
        let dy = if active {
            unsafe { output_gradient.ptr.add(index).read() }
        } else {
            0.0
        };

        let mut value = x;
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
            if lane == 0 {
                unsafe { SHARED[0] = value / cols as f32 };
            }
        }
        thread::sync_threads();
        let mean = unsafe { SHARED[0] };
        let diff = x - mean;

        value = if active { diff * diff } else { 0.0 };
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
            if lane == 0 {
                unsafe { SHARED[0] = 1.0 / (value / cols as f32 + epsilon).sqrt() };
            }
        }
        thread::sync_threads();
        let inverse_std = unsafe { SHARED[0] };
        let normalized = diff * inverse_std;

        value = dy;
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
            if lane == 0 {
                unsafe { SHARED[0] = value };
            }
        }
        thread::sync_threads();
        let gradient_sum = unsafe { SHARED[0] };

        value = dy * normalized;
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
            if lane == 0 {
                unsafe { SHARED[0] = value };
            }
        }
        thread::sync_threads();
        let gradient_normalized_sum = unsafe { SHARED[0] };

        if active {
            let dx = inverse_std / cols as f32
                * (cols as f32 * dy - gradient_sum - normalized * gradient_normalized_sum);
            unsafe { result.ptr.add(index).write(dx) };
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

        static mut SHARED_MATRIX1: shared::SharedArray<f32, SHARED_SIZE> =
            shared::SharedArray::UNINIT;
        static mut SHARED_MATRIX2: shared::SharedArray<f32, SHARED_SIZE> =
            shared::SharedArray::UNINIT;

        let mut sum = 0.0;

        for t in 0..len / TILE_SIZE {
            unsafe {
                SHARED_MATRIX1[ty * TILE_SIZE + tx] = matrix1[c_row * len + t * TILE_SIZE + tx];
                SHARED_MATRIX2[ty * TILE_SIZE + tx] = matrix2[(t * TILE_SIZE + ty) * cols + c_col];
            }
            thread::sync_threads();

            for k in 0..TILE_SIZE {
                unsafe {
                    sum += SHARED_MATRIX1[ty * TILE_SIZE + k] * SHARED_MATRIX2[k * TILE_SIZE + tx];
                }
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
