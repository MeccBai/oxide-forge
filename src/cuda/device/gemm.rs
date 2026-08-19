use crate::cuda::span;
use cuda_device::{
    async_copy::{cp_async_ca_zfill_4, cp_async_commit_group, cp_async_wait_group},
    convert, device, shared, thread, wmma,
};

const MATMUL_TILE_SIZE: usize = 32;
const MATMUL_THREAD_TILE_SIZE: usize = 16;
const MATMUL_SHARED_SIZE: usize = MATMUL_TILE_SIZE * MATMUL_TILE_SIZE;
const TENSOR_K_TILE_SIZE: usize = 16;
const TENSOR_SHARED_STRIDE: usize = 20;
const TENSOR_SHARED_STAGE_SIZE: usize = MATMUL_TILE_SIZE * TENSOR_SHARED_STRIDE;
const TENSOR_SHARED_SIZE: usize = TENSOR_SHARED_STAGE_SIZE * 2;
#[device]
pub(super) fn matrix_multiply_fp32_device(
    matrix1: span::DeviceSliceDescriptor<f32>,
    matrix2: span::DeviceSliceDescriptor<f32>,
    result: span::DeviceSliceMutDescriptor<f32>,
    len: usize,
    rows: usize,
    cols: usize,
) {
    let tx = thread::threadIdx_x() as usize;
    let ty = thread::threadIdx_y() as usize;

    let row0 = thread::blockIdx_y() as usize * MATMUL_TILE_SIZE + ty;
    let row1 = row0 + MATMUL_THREAD_TILE_SIZE;
    let col0 = thread::blockIdx_x() as usize * MATMUL_TILE_SIZE + tx;
    let col1 = col0 + MATMUL_THREAD_TILE_SIZE;

    static mut SHARED_MATRIX1: shared::SharedArray<f32, MATMUL_SHARED_SIZE> =
        shared::SharedArray::UNINIT;
    static mut SHARED_MATRIX2: shared::SharedArray<f32, MATMUL_SHARED_SIZE> =
        shared::SharedArray::UNINIT;

    let mut sum00 = 0.0;
    let mut sum01 = 0.0;
    let mut sum10 = 0.0;
    let mut sum11 = 0.0;

    for tile in 0..len.div_ceil(MATMUL_TILE_SIZE) {
        let k0 = tile * MATMUL_TILE_SIZE + tx;
        let k1 = k0 + MATMUL_THREAD_TILE_SIZE;
        let b_row0 = tile * MATMUL_TILE_SIZE + ty;
        let b_row1 = b_row0 + MATMUL_THREAD_TILE_SIZE;

        unsafe {
            SHARED_MATRIX1[ty * MATMUL_TILE_SIZE + tx] = if row0 < rows && k0 < len {
                matrix1.read(row0 * len + k0)
            } else {
                0.0
            };
            SHARED_MATRIX1[ty * MATMUL_TILE_SIZE + tx + MATMUL_THREAD_TILE_SIZE] =
                if row0 < rows && k1 < len {
                    matrix1.read(row0 * len + k1)
                } else {
                    0.0
                };
            SHARED_MATRIX1[(ty + MATMUL_THREAD_TILE_SIZE) * MATMUL_TILE_SIZE + tx] =
                if row1 < rows && k0 < len {
                    matrix1.read(row1 * len + k0)
                } else {
                    0.0
                };
            SHARED_MATRIX1[(ty + MATMUL_THREAD_TILE_SIZE) * MATMUL_TILE_SIZE
                + tx
                + MATMUL_THREAD_TILE_SIZE] = if row1 < rows && k1 < len {
                matrix1.read(row1 * len + k1)
            } else {
                0.0
            };

            SHARED_MATRIX2[ty * MATMUL_TILE_SIZE + tx] = if b_row0 < len && col0 < cols {
                matrix2.read(b_row0 * cols + col0)
            } else {
                0.0
            };
            SHARED_MATRIX2[ty * MATMUL_TILE_SIZE + tx + MATMUL_THREAD_TILE_SIZE] =
                if b_row0 < len && col1 < cols {
                    matrix2.read(b_row0 * cols + col1)
                } else {
                    0.0
                };
            SHARED_MATRIX2[(ty + MATMUL_THREAD_TILE_SIZE) * MATMUL_TILE_SIZE + tx] =
                if b_row1 < len && col0 < cols {
                    matrix2.read(b_row1 * cols + col0)
                } else {
                    0.0
                };
            SHARED_MATRIX2[(ty + MATMUL_THREAD_TILE_SIZE) * MATMUL_TILE_SIZE
                + tx
                + MATMUL_THREAD_TILE_SIZE] = if b_row1 < len && col1 < cols {
                matrix2.read(b_row1 * cols + col1)
            } else {
                0.0
            };
        }
        thread::sync_threads();

        for k in 0..MATMUL_TILE_SIZE {
            unsafe {
                let a0 = SHARED_MATRIX1[ty * MATMUL_TILE_SIZE + k];
                let a1 = SHARED_MATRIX1[(ty + MATMUL_THREAD_TILE_SIZE) * MATMUL_TILE_SIZE + k];
                let b0 = SHARED_MATRIX2[k * MATMUL_TILE_SIZE + tx];
                let b1 = SHARED_MATRIX2[k * MATMUL_TILE_SIZE + tx + MATMUL_THREAD_TILE_SIZE];
                sum00 += a0 * b0;
                sum01 += a0 * b1;
                sum10 += a1 * b0;
                sum11 += a1 * b1;
            }
        }

        thread::sync_threads();
    }

    if row0 < rows && col0 < cols {
        result.write(row0 * cols + col0, sum00);
    }
    if row0 < rows && col1 < cols {
        result.write(row0 * cols + col1, sum01);
    }
    if row1 < rows && col0 < cols {
        result.write(row1 * cols + col0, sum10);
    }
    if row1 < rows && col1 < cols {
        result.write(row1 * cols + col1, sum11);
    }
}

#[inline(always)]
unsafe fn prefetch_tensor_tile(
    matrix1: span::DeviceSliceDescriptor<f32>,
    matrix2: span::DeviceSliceDescriptor<f32>,
    shared_matrix1: *mut f32,
    shared_matrix2: *mut f32,
    shared_base: usize,
    tile_k: usize,
    block_row: usize,
    block_col: usize,
    tid: usize,
    len: usize,
    rows: usize,
    cols: usize,
) {
    for load in 0..4 {
        let logical_index = tid + load * 128;

        let local_row = logical_index / TENSOR_K_TILE_SIZE;
        let local_k = logical_index % TENSOR_K_TILE_SIZE;
        let global_row = block_row + local_row;
        let a_valid = global_row < rows;
        let a_source = if a_valid {
            unsafe {
                matrix1
                    .as_ptr()
                    .add(global_row * len + tile_k + local_k)
                    .cast::<u8>()
            }
        } else {
            matrix1.as_ptr().cast::<u8>()
        };
        let a_destination =
            unsafe { shared_matrix1.add(shared_base + local_row * TENSOR_SHARED_STRIDE + local_k) };
        unsafe {
            cp_async_ca_zfill_4(
                a_destination.cast::<u32>(),
                a_source,
                if a_valid { 4 } else { 0 },
            );
        }

        // Each warp reads one complete global B row. The individual async
        // copies transpose that row into the column-major shared layout
        // consumed by mma.sync.
        let local_k = logical_index / MATMUL_TILE_SIZE;
        let local_col = logical_index % MATMUL_TILE_SIZE;
        let global_col = block_col + local_col;
        let b_valid = global_col < cols;
        let b_source = if b_valid {
            unsafe {
                matrix2
                    .as_ptr()
                    .add((tile_k + local_k) * cols + global_col)
                    .cast::<u8>()
            }
        } else {
            matrix2.as_ptr().cast::<u8>()
        };
        let b_destination =
            unsafe { shared_matrix2.add(shared_base + local_col * TENSOR_SHARED_STRIDE + local_k) };
        unsafe {
            cp_async_ca_zfill_4(
                b_destination.cast::<u32>(),
                b_source,
                if b_valid { 4 } else { 0 },
            );
        }
    }
}

#[device]
pub(super) fn matrix_multiply_device(
    matrix1: span::DeviceSliceDescriptor<f32>,
    matrix2: span::DeviceSliceDescriptor<f32>,
    result: span::DeviceSliceMutDescriptor<f32>,
    len: usize,
    rows: usize,
    cols: usize,
) {
    static mut SHARED_MATRIX1: shared::SharedArray<f32, TENSOR_SHARED_SIZE> =
        shared::SharedArray::UNINIT;
    static mut SHARED_MATRIX2: shared::SharedArray<f32, TENSOR_SHARED_SIZE> =
        shared::SharedArray::UNINIT;

    let tid = thread::threadIdx_x() as usize;
    let warp_id = tid / 32;
    let lane = tid % 32;
    let group = lane / 4;
    let thread_in_group = lane % 4;
    let warp_row = warp_id / 2;
    let warp_col = warp_id % 2;

    let block_row = thread::blockIdx_y() as usize * MATMUL_TILE_SIZE;
    let block_col = thread::blockIdx_x() as usize * MATMUL_TILE_SIZE;

    let mut accumulator0 = [0.0f32; 4];
    let mut accumulator1 = [0.0f32; 4];

    let shared_matrix1 = core::ptr::addr_of_mut!(SHARED_MATRIX1) as *mut f32;
    let shared_matrix2 = core::ptr::addr_of_mut!(SHARED_MATRIX2) as *mut f32;
    let tile_count = len / TENSOR_K_TILE_SIZE;

    unsafe {
        prefetch_tensor_tile(
            matrix1,
            matrix2,
            shared_matrix1,
            shared_matrix2,
            0,
            0,
            block_row,
            block_col,
            tid,
            len,
            rows,
            cols,
        );
        cp_async_commit_group();
        cp_async_wait_group(0);
    }
    thread::sync_threads();

    for tile in 0..tile_count {
        let read_base = (tile % 2) * TENSOR_SHARED_STAGE_SIZE;

        if tile + 1 < tile_count {
            let next_tile = tile + 1;
            let write_base = (next_tile % 2) * TENSOR_SHARED_STAGE_SIZE;
            unsafe {
                prefetch_tensor_tile(
                    matrix1,
                    matrix2,
                    shared_matrix1,
                    shared_matrix2,
                    write_base,
                    next_tile * TENSOR_K_TILE_SIZE,
                    block_row,
                    block_col,
                    tid,
                    len,
                    rows,
                    cols,
                );
                cp_async_commit_group();
            }
        }

        for k_half in 0..2 {
            let local_k = k_half * 8;
            let a_row_base = warp_row * 16;
            let b_col_base = warp_col * 16;

            let a = unsafe {
                [
                    convert::cvt_rna_tf32_f32(
                        SHARED_MATRIX1[read_base
                            + (a_row_base + group) * TENSOR_SHARED_STRIDE
                            + local_k
                            + thread_in_group],
                    ),
                    convert::cvt_rna_tf32_f32(
                        SHARED_MATRIX1[read_base
                            + (a_row_base + group + 8) * TENSOR_SHARED_STRIDE
                            + local_k
                            + thread_in_group],
                    ),
                    convert::cvt_rna_tf32_f32(
                        SHARED_MATRIX1[read_base
                            + (a_row_base + group) * TENSOR_SHARED_STRIDE
                            + local_k
                            + thread_in_group
                            + 4],
                    ),
                    convert::cvt_rna_tf32_f32(
                        SHARED_MATRIX1[read_base
                            + (a_row_base + group + 8) * TENSOR_SHARED_STRIDE
                            + local_k
                            + thread_in_group
                            + 4],
                    ),
                ]
            };

            let b0 = unsafe {
                [
                    convert::cvt_rna_tf32_f32(
                        SHARED_MATRIX2[read_base
                            + (b_col_base + group) * TENSOR_SHARED_STRIDE
                            + local_k
                            + thread_in_group],
                    ),
                    convert::cvt_rna_tf32_f32(
                        SHARED_MATRIX2[read_base
                            + (b_col_base + group) * TENSOR_SHARED_STRIDE
                            + local_k
                            + thread_in_group
                            + 4],
                    ),
                ]
            };
            let b1 = unsafe {
                [
                    convert::cvt_rna_tf32_f32(
                        SHARED_MATRIX2[read_base
                            + (b_col_base + group + 8) * TENSOR_SHARED_STRIDE
                            + local_k
                            + thread_in_group],
                    ),
                    convert::cvt_rna_tf32_f32(
                        SHARED_MATRIX2[read_base
                            + (b_col_base + group + 8) * TENSOR_SHARED_STRIDE
                            + local_k
                            + thread_in_group
                            + 4],
                    ),
                ]
            };

            unsafe {
                accumulator0 = wmma::mma_m16n8k8_f32_tf32(accumulator0, a, b0);
                accumulator1 = wmma::mma_m16n8k8_f32_tf32(accumulator1, a, b1);
            }
        }

        if tile + 1 < tile_count {
            unsafe { cp_async_wait_group(0) };
            thread::sync_threads();
        }
    }

    for register in 0..4 {
        let output_row = block_row + warp_row * 16 + group + if register >= 2 { 8 } else { 0 };
        let output_col = block_col + warp_col * 16 + thread_in_group * 2 + register % 2;

        if output_row < rows && output_col < cols {
            result.write(output_row * cols + output_col, accumulator0[register]);
        }
        if output_row < rows && output_col + 8 < cols {
            result.write(output_row * cols + output_col + 8, accumulator1[register]);
        }
    }
}
