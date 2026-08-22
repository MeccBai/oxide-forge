use super::{elementwise, gemm, layout, reduction, row};
use crate::cuda::{BinaryOp, span};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub(in crate::cuda) mod kernels {
    const DEFAULT_BLOCK_SIZE_U32: u32 = 1024;

    use super::*;

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn slice_set(target: span::DeviceSliceMutDescriptor<f32>, value: f32) {
        elementwise::slice_set_device(target, value);
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn slice_set_seq(target: span::DeviceSliceMutDescriptor<f32>, dir: bool) {
        elementwise::slice_set_seq_device(target, dir);
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn slice_set_random(target: span::DeviceSliceMutDescriptor<f32>, seed: u32) {
        elementwise::slice_set_random_device(target, seed);
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
        elementwise::slice_binary_device(lhs, rhs, output, op);
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn slice_binary_assign(
        target: span::DeviceSliceMutDescriptor<f32>,
        rhs: span::DeviceSliceDescriptor<f32>,
        op: BinaryOp,
    ) {
        elementwise::slice_binary_assign_device(target, rhs, op);
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn slice_for_each<F>(span: span::DeviceSliceMutDescriptor<f32>, f: F)
    where
        F: Fn(f32) -> f32 + Copy,
    {
        elementwise::slice_for_each_device(span, f);
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn slice_sum(
        span: span::DeviceSliceDescriptor<f32>,
        result: span::DeviceSliceMutDescriptor<f32>,
    ) {
        reduction::slice_sum_device(span, result);
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
        reduction::slice_map_sum_device(span, result, f);
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn slice_max(
        span: span::DeviceSliceDescriptor<f32>,
        result: span::DeviceSliceMutDescriptor<f32>,
    ) {
        reduction::slice_max_device(span, result);
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn matrix_sum_rows(
        matrix: span::DeviceSliceDescriptor<f32>,
        result: span::DeviceSliceMutDescriptor<f32>,
        cols: usize,
    ) {
        row::matrix_sum_rows_device(matrix, result, cols);
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn matrix_softmax_rows(matrix: span::DeviceSliceMutDescriptor<f32>, cols: usize) {
        row::matrix_softmax_rows_device(matrix, cols);
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn matrix_layer_norm_rows(
        matrix: span::DeviceSliceMutDescriptor<f32>,
        cols: usize,
        epsilon: f32,
    ) {
        row::matrix_layer_norm_rows_device(matrix, cols, epsilon);
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn matrix_binary_assign_by_rows(
        matrix: span::DeviceSliceMutDescriptor<f32>,
        row_value: span::DeviceSliceDescriptor<f32>,
        cols: usize,
        op: BinaryOp,
    ) {
        row::matrix_binary_assign_by_rows_device(matrix, row_value, cols, op);
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
        row::softmax_rows_backward_device(probabilities, output_gradient, result, cols);
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
        row::layer_norm_backward_device(input, output_gradient, result, cols, epsilon);
    }

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
        gemm::matrix_multiply_fp32_device(matrix1, matrix2, result, len, rows, cols);
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
        gemm::matrix_multiply_device(matrix1, matrix2, result, len, rows, cols);
    }

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, block = (32, 8, 1))]
    pub fn matrix_transpose(
        matrix: &[f32],
        result: DisjointSlice<f32, thread::Runtime2DIndex>,
        input_rows: usize,
        input_cols: usize,
    ) {
        layout::matrix_transpose_device(matrix, result, input_rows, input_cols);
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn matrix_slice(
        input: &[f32],
        output: DisjointSlice<f32>,
        input_cols: usize,
        tile_rows: usize,
        tile_cols: usize,
        tiles_per_row: usize,
    ) {
        layout::matrix_slice_device(
            input,
            output,
            input_cols,
            tile_rows,
            tile_cols,
            tiles_per_row,
        );
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn rms_norm_assign(input: span::DeviceSliceMutDescriptor<f32>, cols: usize, epsilon: f32) {
        row::rms_norm_assign_device(input, cols, epsilon);
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn compare_vectors(
        lhs: span::DeviceSliceDescriptor<f32>,
        rhs: span::DeviceSliceDescriptor<f32>,
        result: span::DeviceSliceMutDescriptor<u32>,
    ) {
        reduction::compare_vectors_device(lhs, rhs, result);
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn matrix_causal_mask(matrix: span::DeviceSliceMutDescriptor<f32>, cols: usize) {
        row::matrix_causal_mask_device(matrix, cols);
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn matrix_rms_norm_backward(
        input: span::DeviceSliceDescriptor<f32>,
        output_gradient: span::DeviceSliceDescriptor<f32>,
        result: span::DeviceSliceMutDescriptor<f32>,
        cols: usize,
        epsilon: f32,
    ) {
        row::rms_norm_backward_device(input, output_gradient, result, cols, epsilon);
    }
}
