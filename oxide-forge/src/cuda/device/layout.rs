use crate::cuda::span;
use cuda_device::{DisjointSlice, device, shared, thread};

const TRANSPOSE_TILE_SIZE: usize = 32;
const TRANSPOSE_BLOCK_ROWS: usize = 8;
const TRANSPOSE_STRIDE: usize = TRANSPOSE_TILE_SIZE + 1;
const TRANSPOSE_SHARED_SIZE: usize = TRANSPOSE_TILE_SIZE * TRANSPOSE_STRIDE;

#[device]
pub(super) fn matrix_transpose_device(
    matrix: &[f32],
    mut result: DisjointSlice<f32, thread::Runtime2DIndex>,
    input_rows: usize,
    input_cols: usize,
) {
    let tx = thread::threadIdx_x() as usize;
    let ty = thread::threadIdx_y() as usize;
    let input_col = thread::blockIdx_x() as usize * TRANSPOSE_TILE_SIZE + tx;

    static mut TILE: shared::SharedArray<f32, TRANSPOSE_SHARED_SIZE> = shared::SharedArray::UNINIT;

    // A warp loads one contiguous row on every iteration. Eight physical
    // thread rows cover a 32-row tile with four coalesced transactions.
    for row_offset in [0, TRANSPOSE_BLOCK_ROWS, 16, 24] {
        let local_row = ty + row_offset;
        let input_row = thread::blockIdx_y() as usize * TRANSPOSE_TILE_SIZE + local_row;
        unsafe {
            TILE[local_row * TRANSPOSE_STRIDE + tx] =
                if input_row < input_rows && input_col < input_cols {
                    matrix[input_row * input_cols + input_col]
                } else {
                    0.0
                };
        }
    }

    thread::sync_threads();

    let output_col = thread::blockIdx_y() as usize * TRANSPOSE_TILE_SIZE + tx;

    // Reading the padded tile in the opposite direction transposes it
    // without shared-memory bank conflicts. Stores remain warp-coalesced.
    for row_offset in [0, TRANSPOSE_BLOCK_ROWS, 16, 24] {
        let local_row = ty + row_offset;
        let output_row = thread::blockIdx_x() as usize * TRANSPOSE_TILE_SIZE + local_row;
        if output_row < input_cols && output_col < input_rows {
            let output_index = output_row * input_rows + output_col;

            // SAFETY: the bounds above prove the linear index is valid.
            // Each (block, thread, row_offset) tuple owns one output.
            unsafe {
                *result.get_unchecked_mut(output_index) = TILE[tx * TRANSPOSE_STRIDE + local_row];
            }
        }
    }
}

#[device]
pub(super) fn matrix_transpose_batches_device(
    input: span::DeviceSliceDescriptor<f32>,
    output: span::DeviceSliceMutDescriptor<f32>,
    rows: usize,
    cols: usize,
    batch_count: usize,
) {
    let tx = thread::threadIdx_x() as usize;
    let ty = thread::threadIdx_y() as usize;
    let tile_rows = rows.div_ceil(TRANSPOSE_TILE_SIZE);
    let batch = thread::blockIdx_y() as usize / tile_rows;
    let block_row = thread::blockIdx_y() as usize % tile_rows;
    if batch >= batch_count {
        return;
    }
    let input_col = thread::blockIdx_x() as usize * TRANSPOSE_TILE_SIZE + tx;
    let batch_offset = batch * rows * cols;

    static mut TILE: shared::SharedArray<f32, TRANSPOSE_SHARED_SIZE> = shared::SharedArray::UNINIT;

    for row_offset in [0, TRANSPOSE_BLOCK_ROWS, 16, 24] {
        let local_row = ty + row_offset;
        let input_row = block_row * TRANSPOSE_TILE_SIZE + local_row;
        unsafe {
            TILE[local_row * TRANSPOSE_STRIDE + tx] = if input_row < rows && input_col < cols {
                input.read(batch_offset + input_row * cols + input_col)
            } else {
                0.0
            };
        }
    }

    thread::sync_threads();
    let output_col = block_row * TRANSPOSE_TILE_SIZE + tx;
    for row_offset in [0, TRANSPOSE_BLOCK_ROWS, 16, 24] {
        let local_row = ty + row_offset;
        let output_row = thread::blockIdx_x() as usize * TRANSPOSE_TILE_SIZE + local_row;
        if output_row < cols && output_col < rows {
            output.write(batch_offset + output_row * rows + output_col, unsafe {
                TILE[tx * TRANSPOSE_STRIDE + local_row]
            });
        }
    }
}

#[device]
pub(super) fn matrix_slice_device(
    input: &[f32],
    mut output: DisjointSlice<f32>,
    input_cols: usize,
    tile_rows: usize,
    tile_cols: usize,
    tiles_per_row: usize,
) {
    let index = thread::index_1d();
    let linear = index.get();

    if let Some(output_element) = output.get_mut(index) {
        let tile_size = tile_rows * tile_cols;

        let tile_index = linear / tile_size;
        let tile_local = linear % tile_size;

        let tile_y = tile_index / tiles_per_row;
        let tile_x = tile_index % tiles_per_row;

        let local_row = tile_local / tile_cols;
        let local_col = tile_local % tile_cols;

        let input_row = tile_y * tile_rows + local_row;
        let input_col = tile_x * tile_cols + local_col;

        *output_element = input[input_row * input_cols + input_col];
    }
}
