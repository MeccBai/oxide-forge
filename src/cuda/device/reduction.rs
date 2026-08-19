use crate::cuda::span;
use cuda_device::{device, shared, thread, warp};

#[device]
pub(super) unsafe fn slice_sum_device(
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

#[device]
pub(super) fn slice_map_sum_device<F>(
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

#[device]
pub(super) unsafe fn slice_max_device(
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
