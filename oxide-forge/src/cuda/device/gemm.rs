use crate::cuda::tools::DoubleBuffer;
use crate::cuda::{span, tools};
use cuda_device::{convert, device, shared, wmma};

pub(super) const TILE_SIZE: usize = 16;
const K_STEP: usize = 8;
pub(super) const BLOCK_TILE_AXIS: usize = 2;
pub(super) const STAGE_SIZE: usize = BLOCK_TILE_AXIS * TILE_SIZE * TILE_SIZE;

// Swizzle<B, M, S> uses the same bit roles as CuTe, expressed in f32 element
// offsets. M=2 preserves each 16-byte cp.async vector. CUTLASS's canonical
// 128-byte starting point is <3, 2, 3> in these units; NCU selects <3, 2, 2>
// for this kernel's manual TF32 fragment loads because it removes every shared
// load bank conflict. Keep these compile-time constants for cheap NCU sweeps.
const SWIZZLE_BITS: usize = 3;
const SWIZZLE_BASE: usize = 2;
const SWIZZLE_SHIFT: usize = 2;

#[inline(always)]
pub(super) fn swizzled_index(row: usize, col: usize) -> usize {
    let offset = row * TILE_SIZE + col;
    if SWIZZLE_BITS == 0 {
        return offset;
    }

    let bit_mask = (1usize << SWIZZLE_BITS) - 1;
    let source_mask = bit_mask << (SWIZZLE_BASE + SWIZZLE_SHIFT);
    offset ^ ((offset & source_mask) >> SWIZZLE_SHIFT)
}

pub(super) struct TileSources {
    pub(super) a: *const f32,
    pub(super) b: *const f32,
    pub(super) destination: usize,
    pub(super) a_valid: usize,
    pub(super) b_valid: usize,
}

#[device]
pub(super) fn tile_sources(
    matrix_a: *const f32,
    matrix_b: *const f32,
    block_row: usize,
    block_col: usize,
    k_tile: usize,
    thread_id: usize,
    inner: usize,
    rows: usize,
    cols: usize,
    a_row_stride: usize,
    b_row_stride: usize,
) -> TileSources {
    let subtile = thread_id / 64;
    let chunk = thread_id % 64;
    let local_row = chunk / 4;
    let local_col = (chunk % 4) * 4;
    let a_row = (block_row * BLOCK_TILE_AXIS + subtile) * TILE_SIZE + local_row;
    let a_col = k_tile * TILE_SIZE + local_col;
    let b_row = k_tile * TILE_SIZE + local_row;
    let b_col = (block_col * BLOCK_TILE_AXIS + subtile) * TILE_SIZE + local_col;

    let a_valid = if a_row < rows && a_col < inner {
        (inner - a_col).min(4)
    } else {
        0
    };
    let b_valid = if b_row < inner && b_col < cols {
        (cols - b_col).min(4)
    } else {
        0
    };

    let a = if a_valid == 0 {
        matrix_a
    } else {
        unsafe { matrix_a.add(a_row * a_row_stride + a_col) }
    };
    let b = if b_valid == 0 {
        matrix_b
    } else {
        unsafe { matrix_b.add(b_row * b_row_stride + b_col) }
    };

    TileSources {
        a,
        b,
        destination: subtile * TILE_SIZE * TILE_SIZE + swizzled_index(local_row, local_col),
        a_valid,
        b_valid,
    }
}

pub(super) struct Tf32Fragments {
    pub(super) a: [u32; 4],
    pub(super) b_left: [u32; 2],
    pub(super) b_right: [u32; 2],
}

#[inline(always)]
unsafe fn tf32_at(matrix: *const f32, row: usize, col: usize) -> u32 {
    unsafe { convert::cvt_rna_tf32_f32(*matrix.add(swizzled_index(row, col))) }
}

/// Loads one K=8 slice of swizzled 16x16 A/B shared tiles into the register
/// layout required by two `mma.m16n8k8.row.col.tf32` instructions.
#[device]
pub(super) unsafe fn prefetch_tf32_fragments(
    matrix_a: *const f32,
    matrix_b: *const f32,
    k_offset: usize,
    lane: usize,
) -> Tf32Fragments {
    let group = lane / 4;
    let thread = lane % 4;

    let a = unsafe {
        [
            tf32_at(matrix_a, group, k_offset + thread),
            tf32_at(matrix_a, group + 8, k_offset + thread),
            tf32_at(matrix_a, group, k_offset + thread + 4),
            tf32_at(matrix_a, group + 8, k_offset + thread + 4),
        ]
    };
    let b_left = unsafe {
        [
            tf32_at(matrix_b, k_offset + thread, group),
            tf32_at(matrix_b, k_offset + thread + 4, group),
        ]
    };
    let b_right = unsafe {
        [
            tf32_at(matrix_b, k_offset + thread, group + 8),
            tf32_at(matrix_b, k_offset + thread + 4, group + 8),
        ]
    };

    Tf32Fragments { a, b_left, b_right }
}

#[device]
pub(super) fn matrix_multiply_device(
    matrix_a: span::DeviceSliceDescriptor<f32>,
    matrix_b: span::DeviceSliceDescriptor<f32>,
    result: span::DeviceSliceMutDescriptor<f32>,
    inner: usize,
    rows: usize,
    cols: usize,
) {
    matrix_multiply_strided_device(
        matrix_a, matrix_b, result, inner, rows, cols, 0, inner, 0, cols, 0, cols,
    );
}

#[device]
pub(super) fn matrix_multiply_strided_device(
    matrix_a: span::DeviceSliceDescriptor<f32>,
    matrix_b: span::DeviceSliceDescriptor<f32>,
    result: span::DeviceSliceMutDescriptor<f32>,
    inner: usize,
    rows: usize,
    cols: usize,
    a_offset: usize,
    a_row_stride: usize,
    b_offset: usize,
    b_row_stride: usize,
    result_offset: usize,
    result_row_stride: usize,
) {
    matrix_multiply_batched_strided_device(
        matrix_a,
        matrix_b,
        result,
        inner,
        rows,
        cols,
        1,
        a_offset,
        a_row_stride,
        0,
        b_offset,
        b_row_stride,
        0,
        result_offset,
        result_row_stride,
        0,
    );
}

#[device]
pub(super) fn matrix_multiply_batched_strided_device(
    matrix_a: span::DeviceSliceDescriptor<f32>,
    matrix_b: span::DeviceSliceDescriptor<f32>,
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
    static mut MATA0: shared::SharedArray<f32, STAGE_SIZE> = shared::SharedArray::UNINIT;
    static mut MATA1: shared::SharedArray<f32, STAGE_SIZE> = shared::SharedArray::UNINIT;
    static mut MATB0: shared::SharedArray<f32, STAGE_SIZE> = shared::SharedArray::UNINIT;
    static mut MATB1: shared::SharedArray<f32, STAGE_SIZE> = shared::SharedArray::UNINIT;

    let local_warp = tools::index::get_warp_id();
    let global_block_id = tools::index::get_block_id_1d();
    let thread_id = tools::index::get_thread_id_1d();
    let lane = tools::index::get_lane_id();
    let tile_rows = rows.div_ceil(TILE_SIZE);
    let tile_cols = cols.div_ceil(TILE_SIZE);
    let block_tile_cols = tile_cols.div_ceil(BLOCK_TILE_AXIS);
    let block_tile_rows = tile_rows.div_ceil(BLOCK_TILE_AXIS);
    let blocks_per_batch = block_tile_rows * block_tile_cols;
    let batch = global_block_id / blocks_per_batch;
    if batch >= batch_count {
        return;
    }
    let block_id = global_block_id % blocks_per_batch;
    let block_row = block_id / block_tile_cols;
    let block_col = block_id % block_tile_cols;
    let warp_row = local_warp / BLOCK_TILE_AXIS;
    let warp_col = local_warp % BLOCK_TILE_AXIS;
    let tile_row = block_row * BLOCK_TILE_AXIS + warp_row;
    let tile_col = block_col * BLOCK_TILE_AXIS + warp_col;
    let active = tile_row < tile_rows && tile_col < tile_cols;

    let a_stage0 = (&raw mut MATA0).cast::<f32>();
    let a_stage1 = (&raw mut MATA1).cast::<f32>();
    let b_stage0 = (&raw mut MATB0).cast::<f32>();
    let b_stage1 = (&raw mut MATB1).cast::<f32>();
    let source_base = unsafe {
        (
            matrix_a.as_ptr().add(a_offset + batch * a_batch_stride),
            matrix_b.as_ptr().add(b_offset + batch * b_batch_stride),
        )
    };
    let initial = tile_sources(
        source_base.0,
        source_base.1,
        block_row,
        block_col,
        0,
        thread_id,
        inner,
        rows,
        cols,
        a_row_stride,
        b_row_stride,
    );

    let mut a_buffer = unsafe {
        DoubleBuffer::initialize(
            a_stage0,
            a_stage1,
            initial.destination,
            initial.a,
            initial.a_valid,
            4,
        )
    };
    let mut b_buffer = unsafe {
        DoubleBuffer::initialize(
            b_stage0,
            b_stage1,
            initial.destination,
            initial.b,
            initial.b_valid,
            4,
        )
    };
    unsafe { DoubleBuffer::<f32>::ready_blocking() };

    let mut accumulator_left = [0.0f32; 4];
    let mut accumulator_right = [0.0f32; 4];
    let k_tiles = inner.div_ceil(TILE_SIZE);

    for k_tile in 0..k_tiles {
        let has_next = k_tile + 1 < k_tiles;
        if has_next {
            let next = tile_sources(
                source_base.0,
                source_base.1,
                block_row,
                block_col,
                k_tile + 1,
                thread_id,
                inner,
                rows,
                cols,
                a_row_stride,
                b_row_stride,
            );
            unsafe {
                a_buffer.copy_async(next.destination, next.a, next.a_valid, 4);
                b_buffer.copy_async(next.destination, next.b, next.b_valid, 4);
                DoubleBuffer::<f32>::ready_async();
            }
        }

        let current_a = unsafe { a_buffer.current().add(warp_row * TILE_SIZE * TILE_SIZE) };
        let current_b = unsafe { b_buffer.current().add(warp_col * TILE_SIZE * TILE_SIZE) };
        let fragments0 = unsafe { prefetch_tf32_fragments(current_a, current_b, 0, lane) };
        unsafe {
            accumulator_left =
                wmma::mma_m16n8k8_f32_tf32(accumulator_left, fragments0.a, fragments0.b_left);
            accumulator_right =
                wmma::mma_m16n8k8_f32_tf32(accumulator_right, fragments0.a, fragments0.b_right);
        }

        let fragments1 = unsafe { prefetch_tf32_fragments(current_a, current_b, K_STEP, lane) };
        unsafe {
            accumulator_left =
                wmma::mma_m16n8k8_f32_tf32(accumulator_left, fragments1.a, fragments1.b_left);
            accumulator_right =
                wmma::mma_m16n8k8_f32_tf32(accumulator_right, fragments1.a, fragments1.b_right);
        }

        if has_next {
            unsafe { DoubleBuffer::<f32>::wait() };
            a_buffer.advance();
            b_buffer.advance();
        }
    }

    let group = lane / 4;
    let thread = lane % 4;
    for register in 0..4 {
        let output_row = tile_row * TILE_SIZE + group + if register >= 2 { 8 } else { 0 };
        let output_col = tile_col * TILE_SIZE + thread * 2 + register % 2;
        if active && output_row < rows && output_col < cols {
            result.write(
                result_offset
                    + batch * result_batch_stride
                    + output_row * result_row_stride
                    + output_col,
                accumulator_left[register],
            );
        }
        if active && output_row < rows && output_col + 8 < cols {
            result.write(
                result_offset
                    + batch * result_batch_stride
                    + output_row * result_row_stride
                    + output_col
                    + 8,
                accumulator_right[register],
            );
        }
    }
}
