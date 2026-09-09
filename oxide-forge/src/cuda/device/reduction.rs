use crate::cuda::span;
use cuda_device::{device, shared, thread, warp};
use core::mem::size_of;

#[device]
pub(super) fn compare_vectors_device(
    lhs: span::DeviceSliceDescriptor<f32>,
    rhs: span::DeviceSliceDescriptor<f32>,
    elements_per_thread: usize,
    result: span::DeviceSliceMutDescriptor<u32>,
) {
    static mut SHARED: shared::SharedArray<u32, 32> = shared::SharedArray::UNINIT;
    let index = thread::index_1d().get();
    let lane = thread::threadIdx_x() as usize % 32;
    let warp_id = thread::threadIdx_x() as usize / 32;

    let mut temp_result = true;
    for element in 0..elements_per_thread {
        let i = index + element * thread::blockDim_x() as usize;
        if i < lhs.len() {
            temp_result &= lhs.read(i) == rhs.read(i);
        }
    }
    for delta in [1, 2, 4, 8, 16] {
        let other = warp::shuffle_down_u64(temp_result as u64, delta);
        if lane + (delta as usize) < 32 {
            temp_result &= other == 1 as u64;
        }
    }
    if lane == 0 {
        unsafe { SHARED[warp_id] = temp_result as u32 };
    }
    thread::sync_threads();
    if warp_id == 0 {
        temp_result = unsafe { SHARED[lane] == 1 };
        for delta in [1, 2, 4, 8, 16] {
            let other = warp::shuffle_down_u64(temp_result as u64, delta);
            if lane + (delta as usize) < 32 {
                temp_result &= other == 1 as u64;
            }
        }
    }

    if thread::threadIdx_x() == 0 {
        result.write(0, temp_result as u32);
    }
}

#[device]
pub(super) fn map_reduce_device<FM, FR>(
    source: span::DeviceSliceDescriptor<f32>,
    result: span::DeviceSliceMutDescriptor<f32>,
    elements_per_thread: usize,
    map: FM,
    reduce: FR,
    default: f32,
) where
    FM: Fn(f32) -> f32 + Copy,
    FR: Fn(f32, f32) -> f32 + Copy,
{
    let index = thread::index_1d().get();
    let lane = thread::threadIdx_x() as usize % 32;
    let warp_id = thread::threadIdx_x() as usize / 32;

    let mut value: f32 = default;

    for element in 0..elements_per_thread {
        let i = index + element * thread::blockDim_x() as usize;
        let temp = if i < source.len() {
            map(source.read(i))
        } else {
            default
        };
        value = reduce(value, temp);
    }

    value = reduce_warp(value, reduce);

    if lane == 0 {
        let slot = shared::DynamicSharedArray::<f32>::offset(warp_id * size_of::<f32>());
        unsafe { *slot = value };
    }

    thread::sync_threads();

    if warp_id == 0 {
        let slot = shared::DynamicSharedArray::<f32>::offset(lane * size_of::<f32>());
        value = unsafe { *slot };

        value = reduce_warp(value, reduce);
    }

    if thread::threadIdx_x() == 0 {
        let block_id = thread::blockIdx_x() as usize;

        if block_id < result.len() {
            result.write(block_id, value);
        }
    }
}

#[device]
fn reduce_warp<F>(mut value: f32, reduce: F) -> f32
where
    F: Fn(f32, f32) -> f32 + Copy,
{
    for i in [1, 2, 4, 8, 16] {
        value = reduce(value, warp::shuffle_down_f32(value, i));
    }

    value
}

#[device]
pub(super) fn zip_map_reduce_device<FM, FR>(
    source1: span::DeviceSliceDescriptor<f32>,
    source2: span::DeviceSliceDescriptor<f32>,
    result: span::DeviceSliceMutDescriptor<f32>,
    elements_per_thread: usize,
    map: FM,
    reduce: FR,
    default: f32,
) where
    FM: Fn(f32, f32) -> f32 + Copy,
    FR: Fn(f32, f32) -> f32 + Copy,
{
    let len = source1.len();

    let index = thread::index_1d().get();
    let lane = thread::threadIdx_x() as usize % 32;
    let warp_id = thread::threadIdx_x() as usize / 32;

    let mut value: f32 = default;

    for element in 0..elements_per_thread {
        let i = index + element * thread::blockDim_x() as usize;
        let temp = if i < len {
            let s1_value = source1.read(i);
            let s2_value = source2.read(i);
            map(s1_value, s2_value)
        } else {
            default
        };
        value = reduce(value, temp);
    }

    value = reduce_warp(value, reduce);

    if lane == 0 {
        let slot = shared::DynamicSharedArray::<f32>::offset(warp_id * size_of::<f32>());
        unsafe { *slot = value };
    }

    thread::sync_threads();

    if warp_id == 0 {
        let slot = shared::DynamicSharedArray::<f32>::offset(lane * size_of::<f32>());
        value = unsafe { *slot };
        value = reduce_warp(value, reduce);
    }

    if thread::threadIdx_x() == 0 {
        let block_id = thread::blockIdx_x() as usize;
        if block_id < result.len() {
            result.write(block_id, value);
        }
    }
}
