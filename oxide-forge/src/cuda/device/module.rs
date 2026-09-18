use super::{elementwise, gemm, layout, reduction, row};
use crate::cuda::span;
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub(in crate::cuda) mod kernels {
    const DEFAULT_BLOCK_SIZE_U32: u32 = 1024;

    use super::*;

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn slice_set(
        target: span::DeviceSliceMutDescriptor<f32>,
        elements_per_thread: usize,
        value: f32,
    ) {
        elementwise::slice_set_device(target, elements_per_thread, value);
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn slice_set_seq(
        target: span::DeviceSliceMutDescriptor<f32>,
        elements_per_thread: usize,
        dir: bool,
        start: f32,
        step: f32,
    ) {
        elementwise::slice_set_seq_device(target, elements_per_thread, dir, start, step);
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn slice_set_random(
        target: span::DeviceSliceMutDescriptor<f32>,
        elements_per_thread: usize,
        seed: u32,
    ) {
        elementwise::slice_set_random_device(target, elements_per_thread, seed);
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn slice_binary<F>(
        lhs: span::DeviceSliceDescriptor<f32>,
        rhs: span::DeviceSliceDescriptor<f32>,
        output: span::DeviceSliceMutDescriptor<f32>,
        elements_per_thread: usize,
        f: F,
    ) where
        F: Fn(f32, f32) -> f32 + Copy,
    {
        elementwise::slice_binary_device(lhs, rhs, output, elements_per_thread, f);
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn slice_binary_assign<F>(
        target: span::DeviceSliceMutDescriptor<f32>,
        rhs: span::DeviceSliceDescriptor<f32>,
        elements_per_thread: usize,
        f: F,
    ) where
        F: Fn(f32, f32) -> f32 + Copy,
    {
        elementwise::slice_binary_assign_device(target, rhs, elements_per_thread, f);
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn slice_for_each<F>(
        span: span::DeviceSliceMutDescriptor<f32>,
        elements_per_thread: usize,
        f: F,
    ) where
        F: Fn(f32) -> f32 + Copy,
    {
        elementwise::slice_for_each_device(span, elements_per_thread, f);
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1, dynamic_shared = 128)]
    pub fn matrix_sum_rows(
        matrix: span::DeviceSliceDescriptor<f32>,
        result: span::DeviceSliceMutDescriptor<f32>,
        cols: usize,
        elements_per_thread: usize,
    ) {
        let row = thread::blockIdx_x() as usize;
        let row = matrix.slice(row * cols, cols);
        reduction::map_reduce_device(
            row,
            result,
            elements_per_thread,
            move |value| value,
            move |lhs, rhs| lhs + rhs,
            0.0,
        );
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
    pub fn matrix_binary_assign_by_rows<F>(
        matrix: span::DeviceSliceMutDescriptor<f32>,
        row_value: span::DeviceSliceDescriptor<f32>,
        cols: usize,
        f: F,
    ) where
        F: Fn(f32, f32) -> f32 + Copy,
    {
        row::matrix_binary_assign_by_rows_device(matrix, row_value, cols, f);
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
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
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
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn matrix_multiply_batched_strided(
        matrix1: span::DeviceSliceDescriptor<f32>,
        matrix2: span::DeviceSliceDescriptor<f32>,
        result: span::DeviceSliceMutDescriptor<f32>,
        inner: usize,
        rows: usize,
        cols: usize,
        batch_count: usize,
        a_offset: usize,
        a_row_stride: usize,
        a_batch_stride: usize,
        b_offset: usize,
        b_row_stride: usize,
        b_batch_stride: usize,
        result_offset: usize,
        result_row_stride: usize,
        result_batch_stride: usize,
    ) {
        gemm::matrix_multiply_batched_strided_device(
            matrix1,
            matrix2,
            result,
            inner,
            rows,
            cols,
            batch_count,
            a_offset,
            a_row_stride,
            a_batch_stride,
            b_offset,
            b_row_stride,
            b_batch_stride,
            result_offset,
            result_row_stride,
            result_batch_stride,
        );
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
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, block = (32, 8, 1))]
    pub fn matrix_transpose_batches(
        input: span::DeviceSliceDescriptor<f32>,
        output: span::DeviceSliceMutDescriptor<f32>,
        rows: usize,
        cols: usize,
        batch_count: usize,
    ) {
        layout::matrix_transpose_batches_device(input, output, rows, cols, batch_count);
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
        elements_per_thread: usize,
        result: span::DeviceSliceMutDescriptor<u32>,
    ) {
        reduction::compare_vectors_device(lhs, rhs, elements_per_thread, result);
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1)]
    pub fn matrix_causal_mask(
        matrix: span::DeviceSliceMutDescriptor<f32>,
        cols: usize,
        row_period: usize,
    ) {
        row::matrix_causal_mask_device(matrix, cols, row_period);
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

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32 / 2)]
    #[launch_contract(domain = 1)]
    pub fn matrix_rope_encoding(mat: span::DeviceSliceMutDescriptor<f32>, cols: usize) {
        row::rope_encoding_device(mat, cols);
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1, dynamic_shared = 128)]
    pub fn map_reduce<FM, FR>(
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
        reduction::map_reduce_device(source, result, elements_per_thread, map, reduce, default);
    }

    #[kernel]
    #[launch_bounds(DEFAULT_BLOCK_SIZE_U32)]
    #[launch_contract(domain = 1, dynamic_shared = 128)]
    pub fn zip_map_reduce<FM, FR>(
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
        reduction::zip_map_reduce_device(
            source1,
            source2,
            result,
            elements_per_thread,
            map,
            reduce,
            default,
        );
    }
}
