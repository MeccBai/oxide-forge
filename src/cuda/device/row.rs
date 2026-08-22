use super::elementwise::apply_binary;
use crate::cuda::{BinaryOp, span};
use cuda_device::{device, shared, thread, warp};

#[device]
pub(super) fn matrix_causal_mask_device(matrix: span::DeviceSliceMutDescriptor<f32>, cols: usize) {
    let row = thread::blockIdx_x() as usize;
    let col = thread::threadIdx_x() as usize;
    let index = row * cols + col;

    if col < cols && index < matrix.len() && col > row {
        matrix.write(index, f32::NEG_INFINITY);
    }
}

#[device]
pub(super) fn matrix_sum_rows_device(
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
    let mut value = if tid < cols && index < matrix.len() {
        matrix.read(index)
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

    if tid == 0 && row < result.len() {
        result.write(row, value);
    }
}

#[device]
pub(super) fn matrix_softmax_rows_device(matrix: span::DeviceSliceMutDescriptor<f32>, cols: usize) {
    static mut SHARED: shared::SharedArray<f32, 32> = shared::SharedArray::UNINIT;

    let tid = thread::threadIdx_x() as usize;
    let lane = tid % 32;
    let warp_id = tid / 32;
    let index = thread::blockIdx_x() as usize * cols + tid;
    let active = tid < cols && index < matrix.len();
    let x = if active { matrix.read(index) } else { f32::MIN };

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
        matrix.write(index, exp_value / row_sum);
    }
}

#[device]
pub(super) fn matrix_layer_norm_rows_device(
    matrix: span::DeviceSliceMutDescriptor<f32>,
    cols: usize,
    epsilon: f32,
) {
    static mut SHARED: shared::SharedArray<f32, 32> = shared::SharedArray::UNINIT;

    let tid = thread::threadIdx_x() as usize;
    let lane = tid % 32;
    let warp_id = tid / 32;
    let index = thread::blockIdx_x() as usize * cols + tid;
    let active = tid < cols && index < matrix.len();
    let x = if active { matrix.read(index) } else { 0.0 };

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
        matrix.write(index, diff * inverse_std);
    }
}

#[device]
pub(super) fn matrix_binary_assign_by_rows_device(
    matrix: span::DeviceSliceMutDescriptor<f32>,
    row: span::DeviceSliceDescriptor<f32>,
    cols: usize,
    op: BinaryOp,
) {
    let index = thread::index_1d().get();
    if index < matrix.len() {
        let rhs = row.read(index % cols);
        matrix.write(index, apply_binary(matrix.read(index), rhs, op));
    }
}

#[device]
pub(super) fn softmax_rows_backward_device(
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
    let active = tid < cols && index < result.len();
    let probability = if active {
        probabilities.read(index)
    } else {
        0.0
    };
    let gradient = if active {
        output_gradient.read(index)
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
        result.write(index, probability * (gradient - unsafe { SHARED[0] }));
    }
}

#[device]
pub(super) fn layer_norm_backward_device(
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
    let active = tid < cols && index < result.len();
    let x = if active { input.read(index) } else { 0.0 };
    let dy = if active {
        output_gradient.read(index)
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
        result.write(index, dx);
    }
}

#[device]
pub(super) fn rms_norm_assign_device(
    input: span::DeviceSliceMutDescriptor<f32>,
    cols: usize,
    epsilon: f32,
) {
    static mut SHARED: shared::SharedArray<f32, 32> = shared::SharedArray::UNINIT;
    let tid = thread::threadIdx_x() as usize;
    let block_id = thread::blockIdx_x() as usize;
    let warp_id = tid / 32;
    let lane = tid % 32;

    let index = block_id * cols + tid;
    let active = tid < cols && index < input.len();
    let original = if active { input.read(index) } else { 0.0 };
    let mut value = original * original;

    for delta in [1, 2, 4, 8, 16] {
        value += warp::shuffle_down_f32(value, delta);
    }

    if lane == 0 {
        unsafe { SHARED[warp_id] = value };
    }
    thread::sync_threads();

    if tid < 32 {
        value = unsafe { SHARED[tid] };
    }

    thread::sync_threads();

    if tid < 32 {
        for delta in [1, 2, 4, 8, 16] {
            value += warp::shuffle_down_f32(value, delta);
        }
        unsafe { SHARED[tid] = value };
    }

    thread::sync_threads();

    if active {
        input.write(
            index,
            original / (unsafe { SHARED[0] } / cols as f32 + epsilon).sqrt(),
        );
    }
}

#[device]
pub fn rms_norm_backward_device(
    input: span::DeviceSliceDescriptor<f32>,
    output_gradient: span::DeviceSliceDescriptor<f32>,
    result: span::DeviceSliceMutDescriptor<f32>,
    cols: usize,
    epsilon: f32,
) {
    static mut SUM_REDUCE: shared::SharedArray<f32, 32> = shared::SharedArray::UNINIT;
    static mut DOT_REDUCE: shared::SharedArray<f32, 32> = shared::SharedArray::UNINIT;

    let tid = thread::threadIdx_x() as usize;
    let block_id = thread::blockIdx_x() as usize;
    let warp_id = tid / 32;
    let lane = tid % 32;
    let index = block_id * cols + tid;
    let active =
        tid < cols && index < input.len() && index < output_gradient.len() && index < result.len();
    let (original, mut dot) = if active {
        let val = input.read(index);
        (val, val * output_gradient.read(index))
    } else {
        (0.0, 0.0)
    };
    let mut value = original * original;
    for delta in [1, 2, 4, 8, 16] {
        value += warp::shuffle_down_f32(value, delta);
        dot += warp::shuffle_down_f32(dot, delta);
    }
    if lane == 0 {
        unsafe {
            SUM_REDUCE[warp_id] = value;
            DOT_REDUCE[warp_id] = dot;
        };
    }
    thread::sync_threads();

    if tid < 32 {
        value = unsafe { SUM_REDUCE[tid] };
        dot = unsafe { DOT_REDUCE[tid] };
    }

    thread::sync_threads();

    if tid < 32 {
        for delta in [1, 2, 4, 8, 16] {
            value += warp::shuffle_down_f32(value, delta);
            dot += warp::shuffle_down_f32(dot, delta);
        }
        unsafe {
            SUM_REDUCE[tid] = value;
            DOT_REDUCE[tid] = dot;
        };
    }

    thread::sync_threads();
    let sum = unsafe { SUM_REDUCE[0] };
    let dot_reduce = unsafe { DOT_REDUCE[0] };
    thread::sync_threads();

    let rms = (sum / cols as f32 + epsilon).sqrt();

    let inv_rms = 1.0 / rms;
    let correction = dot_reduce / cols as f32 * inv_rms * inv_rms * inv_rms;

    if active {
        let gradient = output_gradient.read(index);
        let dx = gradient * inv_rms - original * correction;
        result.write(index, dx);
    }
}
