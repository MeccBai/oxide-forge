use crate::cuda::span;
use cuda_device::{device, thread};

#[device]
pub(super) fn span_set_device<F>(
    target: span::DeviceSliceMutDescriptor<f32>,
    elements_per_thread: usize,
    f: F,
) where
    F: Fn(usize) -> f32 + Copy,
{
    let thread_index = thread::index_1d().get();
    let stride = thread::gridDim_x() as usize * thread::blockDim_x() as usize;

    for iteration in 0..elements_per_thread {
        let index = thread_index + stride * iteration;

        if index < target.len() {
            target.write(index, f(index));
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
