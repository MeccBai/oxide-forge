use super::elementwise::apply_binary;
use crate::cuda::{BinaryOp, span};
use cuda_device::{device, shared, thread, warp};

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

#[device]
pub(super) fn matrix_softmax_rows_device(matrix: span::DeviceSliceMutDescriptor<f32>, cols: usize) {
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

#[device]
pub(super) fn matrix_binary_assign_by_rows_device(
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
