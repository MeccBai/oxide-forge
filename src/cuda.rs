use cuda_device::{
    DisjointSlice,
    async_copy::{cp_async_ca_zfill_4, cp_async_commit_group, cp_async_wait_group},
    convert, kernel, launch_bounds, launch_contract, shared, thread, warp, wmma,
};
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

    const MATMUL_TILE_SIZE: usize = 32;
    const MATMUL_THREAD_TILE_SIZE: usize = 16;
    const MATMUL_SHARED_SIZE: usize = MATMUL_TILE_SIZE * MATMUL_TILE_SIZE;
    const TENSOR_K_TILE_SIZE: usize = 16;
    const TENSOR_SHARED_STRIDE: usize = 20;
    const TENSOR_SHARED_STAGE_SIZE: usize = MATMUL_TILE_SIZE * TENSOR_SHARED_STRIDE;
    const TENSOR_SHARED_SIZE: usize = TENSOR_SHARED_STAGE_SIZE * 2;
    const TRANSPOSE_TILE_SIZE: usize = 16;
    const TRANSPOSE_STRIDE: usize = TRANSPOSE_TILE_SIZE + 1;
    const TRANSPOSE_SHARED_SIZE: usize = TRANSPOSE_TILE_SIZE * TRANSPOSE_STRIDE;

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, block = (16, 16, 1))]
    pub fn matrix_multiply_fp32(
        matrix1: span::DeviceSliceDescriptor<f32>,
        matrix2: span::DeviceSliceDescriptor<f32>,
        result: span::DeviceSliceMutDescriptor<f32>,
        len: usize,
        rows: usize,
        cols: usize,
    ) {
        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;

        let row0 = thread::blockIdx_y() as usize * MATMUL_TILE_SIZE + ty;
        let row1 = row0 + MATMUL_THREAD_TILE_SIZE;
        let col0 = thread::blockIdx_x() as usize * MATMUL_TILE_SIZE + tx;
        let col1 = col0 + MATMUL_THREAD_TILE_SIZE;

        static mut SHARED_MATRIX1: shared::SharedArray<f32, MATMUL_SHARED_SIZE> =
            shared::SharedArray::UNINIT;
        static mut SHARED_MATRIX2: shared::SharedArray<f32, MATMUL_SHARED_SIZE> =
            shared::SharedArray::UNINIT;

        let mut sum00 = 0.0;
        let mut sum01 = 0.0;
        let mut sum10 = 0.0;
        let mut sum11 = 0.0;

        for tile in 0..len.div_ceil(MATMUL_TILE_SIZE) {
            let k0 = tile * MATMUL_TILE_SIZE + tx;
            let k1 = k0 + MATMUL_THREAD_TILE_SIZE;
            let b_row0 = tile * MATMUL_TILE_SIZE + ty;
            let b_row1 = b_row0 + MATMUL_THREAD_TILE_SIZE;

            unsafe {
                SHARED_MATRIX1[ty * MATMUL_TILE_SIZE + tx] = if row0 < rows && k0 < len {
                    matrix1.ptr.add(row0 * len + k0).read()
                } else {
                    0.0
                };
                SHARED_MATRIX1[ty * MATMUL_TILE_SIZE + tx + MATMUL_THREAD_TILE_SIZE] =
                    if row0 < rows && k1 < len {
                        matrix1.ptr.add(row0 * len + k1).read()
                    } else {
                        0.0
                    };
                SHARED_MATRIX1[(ty + MATMUL_THREAD_TILE_SIZE) * MATMUL_TILE_SIZE + tx] =
                    if row1 < rows && k0 < len {
                        matrix1.ptr.add(row1 * len + k0).read()
                    } else {
                        0.0
                    };
                SHARED_MATRIX1[(ty + MATMUL_THREAD_TILE_SIZE) * MATMUL_TILE_SIZE
                    + tx
                    + MATMUL_THREAD_TILE_SIZE] = if row1 < rows && k1 < len {
                    matrix1.ptr.add(row1 * len + k1).read()
                } else {
                    0.0
                };

                SHARED_MATRIX2[ty * MATMUL_TILE_SIZE + tx] = if b_row0 < len && col0 < cols {
                    matrix2.ptr.add(b_row0 * cols + col0).read()
                } else {
                    0.0
                };
                SHARED_MATRIX2[ty * MATMUL_TILE_SIZE + tx + MATMUL_THREAD_TILE_SIZE] =
                    if b_row0 < len && col1 < cols {
                        matrix2.ptr.add(b_row0 * cols + col1).read()
                    } else {
                        0.0
                    };
                SHARED_MATRIX2[(ty + MATMUL_THREAD_TILE_SIZE) * MATMUL_TILE_SIZE + tx] =
                    if b_row1 < len && col0 < cols {
                        matrix2.ptr.add(b_row1 * cols + col0).read()
                    } else {
                        0.0
                    };
                SHARED_MATRIX2[(ty + MATMUL_THREAD_TILE_SIZE) * MATMUL_TILE_SIZE
                    + tx
                    + MATMUL_THREAD_TILE_SIZE] = if b_row1 < len && col1 < cols {
                    matrix2.ptr.add(b_row1 * cols + col1).read()
                } else {
                    0.0
                };
            }
            thread::sync_threads();

            for k in 0..MATMUL_TILE_SIZE {
                unsafe {
                    let a0 = SHARED_MATRIX1[ty * MATMUL_TILE_SIZE + k];
                    let a1 = SHARED_MATRIX1[(ty + MATMUL_THREAD_TILE_SIZE) * MATMUL_TILE_SIZE + k];
                    let b0 = SHARED_MATRIX2[k * MATMUL_TILE_SIZE + tx];
                    let b1 = SHARED_MATRIX2[k * MATMUL_TILE_SIZE + tx + MATMUL_THREAD_TILE_SIZE];
                    sum00 += a0 * b0;
                    sum01 += a0 * b1;
                    sum10 += a1 * b0;
                    sum11 += a1 * b1;
                }
            }

            thread::sync_threads();
        }

        unsafe {
            if row0 < rows && col0 < cols {
                result.ptr.add(row0 * cols + col0).write(sum00);
            }
            if row0 < rows && col1 < cols {
                result.ptr.add(row0 * cols + col1).write(sum01);
            }
            if row1 < rows && col0 < cols {
                result.ptr.add(row1 * cols + col0).write(sum10);
            }
            if row1 < rows && col1 < cols {
                result.ptr.add(row1 * cols + col1).write(sum11);
            }
        }
    }

    #[inline(always)]
    unsafe fn prefetch_tensor_tile(
        matrix1: span::DeviceSliceDescriptor<f32>,
        matrix2: span::DeviceSliceDescriptor<f32>,
        shared_matrix1: *mut f32,
        shared_matrix2: *mut f32,
        shared_base: usize,
        tile_k: usize,
        block_row: usize,
        block_col: usize,
        tid: usize,
        len: usize,
        rows: usize,
        cols: usize,
    ) {
        for load in 0..4 {
            let logical_index = tid + load * 128;

            let local_row = logical_index / TENSOR_K_TILE_SIZE;
            let local_k = logical_index % TENSOR_K_TILE_SIZE;
            let global_row = block_row + local_row;
            let a_valid = global_row < rows;
            let a_source = if a_valid {
                unsafe {
                    matrix1
                        .ptr
                        .add(global_row * len + tile_k + local_k)
                        .cast::<u8>()
                }
            } else {
                matrix1.ptr.cast::<u8>()
            };
            let a_destination = unsafe {
                shared_matrix1.add(shared_base + local_row * TENSOR_SHARED_STRIDE + local_k)
            };
            unsafe {
                cp_async_ca_zfill_4(
                    a_destination.cast::<u32>(),
                    a_source,
                    if a_valid { 4 } else { 0 },
                );
            }

            // Each warp reads one complete global B row. The individual async
            // copies transpose that row into the column-major shared layout
            // consumed by mma.sync.
            let local_k = logical_index / MATMUL_TILE_SIZE;
            let local_col = logical_index % MATMUL_TILE_SIZE;
            let global_col = block_col + local_col;
            let b_valid = global_col < cols;
            let b_source = if b_valid {
                unsafe {
                    matrix2
                        .ptr
                        .add((tile_k + local_k) * cols + global_col)
                        .cast::<u8>()
                }
            } else {
                matrix2.ptr.cast::<u8>()
            };
            let b_destination = unsafe {
                shared_matrix2.add(shared_base + local_col * TENSOR_SHARED_STRIDE + local_k)
            };
            unsafe {
                cp_async_ca_zfill_4(
                    b_destination.cast::<u32>(),
                    b_source,
                    if b_valid { 4 } else { 0 },
                );
            }
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 2, block = (128, 1, 1))]
    pub fn matrix_multiply(
        matrix1: span::DeviceSliceDescriptor<f32>,
        matrix2: span::DeviceSliceDescriptor<f32>,
        result: span::DeviceSliceMutDescriptor<f32>,
        len: usize,
        rows: usize,
        cols: usize,
    ) {
        static mut SHARED_MATRIX1: shared::SharedArray<f32, TENSOR_SHARED_SIZE> =
            shared::SharedArray::UNINIT;
        static mut SHARED_MATRIX2: shared::SharedArray<f32, TENSOR_SHARED_SIZE> =
            shared::SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let warp_id = tid / 32;
        let lane = tid % 32;
        let group = lane / 4;
        let thread_in_group = lane % 4;
        let warp_row = warp_id / 2;
        let warp_col = warp_id % 2;

        let block_row = thread::blockIdx_y() as usize * MATMUL_TILE_SIZE;
        let block_col = thread::blockIdx_x() as usize * MATMUL_TILE_SIZE;

        let mut accumulator0 = [0.0f32; 4];
        let mut accumulator1 = [0.0f32; 4];

        let shared_matrix1 = core::ptr::addr_of_mut!(SHARED_MATRIX1) as *mut f32;
        let shared_matrix2 = core::ptr::addr_of_mut!(SHARED_MATRIX2) as *mut f32;
        let tile_count = len / TENSOR_K_TILE_SIZE;

        unsafe {
            prefetch_tensor_tile(
                matrix1,
                matrix2,
                shared_matrix1,
                shared_matrix2,
                0,
                0,
                block_row,
                block_col,
                tid,
                len,
                rows,
                cols,
            );
            cp_async_commit_group();
            cp_async_wait_group(0);
        }
        thread::sync_threads();

        for tile in 0..tile_count {
            let read_base = (tile % 2) * TENSOR_SHARED_STAGE_SIZE;

            if tile + 1 < tile_count {
                let next_tile = tile + 1;
                let write_base = (next_tile % 2) * TENSOR_SHARED_STAGE_SIZE;
                unsafe {
                    prefetch_tensor_tile(
                        matrix1,
                        matrix2,
                        shared_matrix1,
                        shared_matrix2,
                        write_base,
                        next_tile * TENSOR_K_TILE_SIZE,
                        block_row,
                        block_col,
                        tid,
                        len,
                        rows,
                        cols,
                    );
                    cp_async_commit_group();
                }
            }

            for k_half in 0..2 {
                let local_k = k_half * 8;
                let a_row_base = warp_row * 16;
                let b_col_base = warp_col * 16;

                let a = unsafe {
                    [
                        convert::cvt_rna_tf32_f32(
                            SHARED_MATRIX1[read_base
                                + (a_row_base + group) * TENSOR_SHARED_STRIDE
                                + local_k
                                + thread_in_group],
                        ),
                        convert::cvt_rna_tf32_f32(
                            SHARED_MATRIX1[read_base
                                + (a_row_base + group + 8) * TENSOR_SHARED_STRIDE
                                + local_k
                                + thread_in_group],
                        ),
                        convert::cvt_rna_tf32_f32(
                            SHARED_MATRIX1[read_base
                                + (a_row_base + group) * TENSOR_SHARED_STRIDE
                                + local_k
                                + thread_in_group
                                + 4],
                        ),
                        convert::cvt_rna_tf32_f32(
                            SHARED_MATRIX1[read_base
                                + (a_row_base + group + 8) * TENSOR_SHARED_STRIDE
                                + local_k
                                + thread_in_group
                                + 4],
                        ),
                    ]
                };

                let b0 = unsafe {
                    [
                        convert::cvt_rna_tf32_f32(
                            SHARED_MATRIX2[read_base
                                + (b_col_base + group) * TENSOR_SHARED_STRIDE
                                + local_k
                                + thread_in_group],
                        ),
                        convert::cvt_rna_tf32_f32(
                            SHARED_MATRIX2[read_base
                                + (b_col_base + group) * TENSOR_SHARED_STRIDE
                                + local_k
                                + thread_in_group
                                + 4],
                        ),
                    ]
                };
                let b1 = unsafe {
                    [
                        convert::cvt_rna_tf32_f32(
                            SHARED_MATRIX2[read_base
                                + (b_col_base + group + 8) * TENSOR_SHARED_STRIDE
                                + local_k
                                + thread_in_group],
                        ),
                        convert::cvt_rna_tf32_f32(
                            SHARED_MATRIX2[read_base
                                + (b_col_base + group + 8) * TENSOR_SHARED_STRIDE
                                + local_k
                                + thread_in_group
                                + 4],
                        ),
                    ]
                };

                unsafe {
                    accumulator0 = wmma::mma_m16n8k8_f32_tf32(accumulator0, a, b0);
                    accumulator1 = wmma::mma_m16n8k8_f32_tf32(accumulator1, a, b1);
                }
            }

            if tile + 1 < tile_count {
                unsafe { cp_async_wait_group(0) };
                thread::sync_threads();
            }
        }

        for register in 0..4 {
            let output_row = block_row + warp_row * 16 + group + if register >= 2 { 8 } else { 0 };
            let output_col = block_col + warp_col * 16 + thread_in_group * 2 + register % 2;

            unsafe {
                if output_row < rows && output_col < cols {
                    result
                        .ptr
                        .add(output_row * cols + output_col)
                        .write(accumulator0[register]);
                }
                if output_row < rows && output_col + 8 < cols {
                    result
                        .ptr
                        .add(output_row * cols + output_col + 8)
                        .write(accumulator1[register]);
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
        let input_row = thread::blockIdx_y() as usize * TRANSPOSE_TILE_SIZE + ty;
        let input_col = thread::blockIdx_x() as usize * TRANSPOSE_TILE_SIZE + tx;

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

        let output_row = thread::blockIdx_x() as usize * TRANSPOSE_TILE_SIZE + ty;
        let output_col = thread::blockIdx_y() as usize * TRANSPOSE_TILE_SIZE + tx;

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
