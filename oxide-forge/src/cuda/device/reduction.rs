use crate::cuda::span;
use cuda_device::{device, shared, thread, warp};

#[device]
pub(super) fn compare_vectors_device(
    lhs: span::DeviceSliceDescriptor<f32>,
    rhs: span::DeviceSliceDescriptor<f32>,
    result: span::DeviceSliceMutDescriptor<u32>,
) {
    static mut SHARED: shared::SharedArray<u32, 32> = shared::SharedArray::UNINIT;
    let index = thread::index_1d().get();
    let lane = thread::threadIdx_x() as usize % 32;
    let warp_id = thread::threadIdx_x() as usize / 32;

    let mut temp_result = if index < lhs.len() && index < rhs.len() {
        lhs.read(index) == rhs.read(index)
    } else {
        true
    };
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
        let block_id = thread::blockIdx_x() as usize;
        if block_id < result.len() {
            result.write(block_id, temp_result as u32);
        }
    }
}

#[device]
pub(super) fn map_reduce_device<FM, FR>(
    source: span::DeviceSliceDescriptor<f32>,
    result: span::DeviceSliceMutDescriptor<f32>,
    elements_per_thread: usize,
    apply_map: bool,
    map: FM,
    reduce: FR,
    default: f32,
) where
    FM: Fn(f32) -> f32 + Copy,
    FR: Fn(f32, f32) -> f32 + Copy,
{
    let shared = shared::DynamicSharedArray::<f32>::get();
    let index = thread::index_1d().get();
    let lane = thread::threadIdx_x() as usize % 32;
    let warp_id = thread::threadIdx_x() as usize / 32;

    let range = (
        index * elements_per_thread,
        (index + 1) * elements_per_thread,
    );

    let mut value: f32 = default;

    for i in range.0..range.1 {
        let temp = if i < source.len() {
            let source_value = source.read(i);
            if apply_map {
                map(source_value)
            } else {
                source_value
            }
        } else {
            default
        };
        value = reduce(value, temp);
    }

    for i in [1, 2, 4, 8, 16] {
        value = reduce(value, warp::shuffle_down_f32(value, i));
    }

    if lane == 0 {
        unsafe { shared.add(warp_id).write(value) };
    }

    thread::sync_threads();

    if warp_id == 0 {
        value = unsafe { shared.add(lane).read() };

        for delta in [1, 2, 4, 8, 16] {
            value = reduce(value, warp::shuffle_down_f32(value, delta));
        }
    }

    if thread::threadIdx_x() == 0 {
        let block_id = thread::blockIdx_x() as usize;

        if block_id < result.len() {
            result.write(block_id, value);
        }
    }
}
