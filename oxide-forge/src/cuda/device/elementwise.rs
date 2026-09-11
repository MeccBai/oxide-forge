use super::common::random;
use crate::cuda::span;
use cuda_device::{device, thread};


#[device]
pub(super) fn slice_set_device(
    target: span::DeviceSliceMutDescriptor<f32>,
    elements_per_thread: usize,
    value: f32,
) {
    let thread_index = thread::index_1d().get();
    let stride = thread::gridDim_x() as usize * thread::blockDim_x() as usize;

    for iteration in 0..elements_per_thread {
        let index = thread_index + stride * iteration;

        if index < target.len() {
            target.write(index, value);
        }
    }
}

#[device]
pub(super) fn slice_set_seq_device(
    target: span::DeviceSliceMutDescriptor<f32>,
    elements_per_thread: usize,
    dir: bool,
    start:f32,
    step:f32
) {
    let thread_index = thread::index_1d().get();
    let stride = thread::gridDim_x() as usize * thread::blockDim_x() as usize;
    for iteration in 0..elements_per_thread {
        let index = thread_index + stride * iteration;
        if index < target.len() {
            let value = if dir {
                index  as f32 * step + start
            } else {
                (target.len() - index) as f32 * step - start
            };
            
            target.write(index, value);
        }
    }
}

#[device]
pub(super) fn slice_set_random_device(
    target: span::DeviceSliceMutDescriptor<f32>,
    elements_per_thread: usize,
    seed: u32,
) {
    let thread_index = thread::index_1d().get();
    let stride = thread::gridDim_x() as usize * thread::blockDim_x() as usize;

    for iteration in 0..elements_per_thread {
        let index = thread_index + stride * iteration;
        if index < target.len() {
            let rand = random(seed + index as u32);
            target.write(index, (rand as f32) / (u32::MAX as f32));
        }
    }
}

#[device]
pub(super) fn slice_binary_device<F>(
    lhs: span::DeviceSliceDescriptor<f32>,
    rhs: span::DeviceSliceDescriptor<f32>,
    output: span::DeviceSliceMutDescriptor<f32>,
    elements_per_thread: usize,
    f: F,
) where
    F: Fn(f32, f32) -> f32 + Copy,
{
    let thread_index = thread::index_1d().get();
    let stride = thread::gridDim_x() as usize * thread::blockDim_x() as usize;

    let len = output.len();

    for iteration in 0..elements_per_thread {
        let index = thread_index + stride * iteration;

        if index < len {
            let val = f(lhs.read(index), rhs.read(index));
            output.write(index, val);
        }
    }
}

#[device]
pub(super) fn slice_binary_assign_device<F>(
    target: span::DeviceSliceMutDescriptor<f32>,
    rhs: span::DeviceSliceDescriptor<f32>,
    elements_per_thread: usize,
    f: F,
) where
    F: Fn(f32, f32) -> f32 + Copy,
{
    let thread_index = thread::index_1d().get();
    let stride = thread::gridDim_x() as usize * thread::blockDim_x() as usize;

    let len = target.len();

    for iteration in 0..elements_per_thread {
        let index = thread_index + stride * iteration;

        if index < len {
            let val = f(target.read(index), rhs.read(index));
            target.write(index, val);
        }
    }
}

#[device]
pub(super) fn slice_for_each_device<F>(
    span: span::DeviceSliceMutDescriptor<f32>,
    elements_per_thread: usize,
    f: F,
) where
    F: Fn(f32) -> f32 + Copy,
{
    let thread_index = thread::index_1d().get();
    let stride = thread::gridDim_x() as usize * thread::blockDim_x() as usize;

    for iteration in 0..elements_per_thread {
        let index = thread_index + stride * iteration;

        if index < span.len() {
            span.write(index, f(span.read(index)));
        }
    }
}
