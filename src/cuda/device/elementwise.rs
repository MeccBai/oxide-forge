use super::common::random;
use crate::cuda::{BinaryOp, span};
use cuda_device::{device, thread};

#[inline(always)]
pub(super) fn apply_binary(lhs: f32, rhs: f32, op: BinaryOp) -> f32 {
    match op {
        BinaryOp::Add => lhs + rhs,
        BinaryOp::Sub => lhs - rhs,
        BinaryOp::Mul => lhs * rhs,
        BinaryOp::Div => lhs / rhs,
    }
}

#[device]
pub(super) fn slice_set_device(target: span::DeviceSliceMutDescriptor<f32>, value: f32) {
    let index = thread::index_1d().get();
    if index < target.len() {
        target.write(index, value);
    }
}

#[device]
pub(super) fn slice_set_seq_device(target: span::DeviceSliceMutDescriptor<f32>, dir: bool) {
    let index = thread::index_1d().get();
    if index < target.len() {
        let value = if dir {
            index as f32
        } else {
            (target.len() - index) as f32
        };
        target.write(index, value);
    }
}

#[device]
pub(super) fn slice_set_random_device(target: span::DeviceSliceMutDescriptor<f32>, seed: u32) {
    let index = thread::index_1d().get();
    if index < target.len() {
        let rand = random(seed + index as u32);
        target.write(index, (rand as f32) / (u32::MAX as f32));
    }
}

#[device]
pub(super) fn slice_binary_device(
    lhs: span::DeviceSliceDescriptor<f32>,
    rhs: span::DeviceSliceDescriptor<f32>,
    output: span::DeviceSliceMutDescriptor<f32>,
    op: BinaryOp,
) {
    let index = thread::index_1d().get();
    if index < output.len() && index < lhs.len() && index < rhs.len() {
        output.write(index, apply_binary(lhs.read(index), rhs.read(index), op));
    }
}

#[device]
pub(super) fn slice_binary_assign_device(
    target: span::DeviceSliceMutDescriptor<f32>,
    rhs: span::DeviceSliceDescriptor<f32>,
    op: BinaryOp,
) {
    let index = thread::index_1d().get();
    if index < target.len() && index < rhs.len() {
        target.write(index, apply_binary(target.read(index), rhs.read(index), op));
    }
}

#[device]
pub(super) fn slice_for_each_device<F>(span: span::DeviceSliceMutDescriptor<f32>, f: F)
where
    F: Fn(f32) -> f32 + Copy,
{
    let index = thread::index_1d().get();
    if index < span.len() {
        span.write(index, f(span.read(index)));
    }
}
