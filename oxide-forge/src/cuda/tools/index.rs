use cuda_device::{device, thread, warp};

/// Returns the warp's linear index within its block. Thread coordinates are
/// first linearized across all three block dimensions.
#[device]
pub fn get_warp_id() -> usize {
    let thread_id = thread::threadIdx_x() as usize
        + thread::blockDim_x() as usize
            * (thread::threadIdx_y() as usize
                + thread::blockDim_y() as usize * thread::threadIdx_z() as usize);
    thread_id / 32
}

#[device]
pub fn get_lane_id() -> usize {
    warp::lane_id() as usize
}

/// Returns the thread's linear index within its block for a 1D, 2D, or 3D
/// block. The X coordinate is the fastest-moving dimension.
#[device]
pub fn get_thread_id() -> usize {
    thread::threadIdx_x() as usize
        + thread::blockDim_x() as usize
            * (thread::threadIdx_y() as usize
                + thread::blockDim_y() as usize * thread::threadIdx_z() as usize)
}

/// Returns the block's linear index within a 1D, 2D, or 3D grid. The X
/// coordinate is the fastest-moving dimension.
#[device]
pub fn get_block_id() -> usize {
    thread::blockIdx_x() as usize
        + thread::gridDim_x() as usize
            * (thread::blockIdx_y() as usize
                + thread::gridDim_y() as usize * thread::blockIdx_z() as usize)
}

/// Returns a unique linear thread index across the complete launch.
#[device]
pub fn get_global_thread_id() -> usize {
    let thread_id = thread::threadIdx_x() as usize
        + thread::blockDim_x() as usize
            * (thread::threadIdx_y() as usize
                + thread::blockDim_y() as usize * thread::threadIdx_z() as usize);
    let block_id = thread::blockIdx_x() as usize
        + thread::gridDim_x() as usize
            * (thread::blockIdx_y() as usize
                + thread::gridDim_y() as usize * thread::blockIdx_z() as usize);
    let threads_per_block = thread::blockDim_x() as usize
        * thread::blockDim_y() as usize
        * thread::blockDim_z() as usize;

    block_id * threads_per_block + thread_id
}

/// Returns a unique warp index across the complete launch. Partial tail warps
/// remain local to their block and are never merged with the following block.
#[device]
pub fn get_global_warp_id() -> usize {
    let thread_id = thread::threadIdx_x() as usize
        + thread::blockDim_x() as usize
            * (thread::threadIdx_y() as usize
                + thread::blockDim_y() as usize * thread::threadIdx_z() as usize);
    let block_id = thread::blockIdx_x() as usize
        + thread::gridDim_x() as usize
            * (thread::blockIdx_y() as usize
                + thread::gridDim_y() as usize * thread::blockIdx_z() as usize);
    let threads_per_block = thread::blockDim_x() as usize
        * thread::blockDim_y() as usize
        * thread::blockDim_z() as usize;
    let warps_per_block = (threads_per_block + 31) / 32;

    block_id * warps_per_block + thread_id / 32
}

#[device]
pub fn get_thread_id_1d() -> usize {
    thread::threadIdx_x() as usize
}

#[device]
pub fn get_block_id_1d() -> usize {
    thread::blockIdx_x() as usize
}

#[device]
pub fn get_block_size_1d() -> usize {
    thread::blockDim_x() as usize
}

#[device]
pub fn get_grid_size_1d() -> usize {
    thread::gridDim_x() as usize
}

#[device]
pub fn get_global_id_1d() -> usize {
    thread::blockIdx_x() as usize * thread::blockDim_x() as usize + thread::threadIdx_x() as usize
}

#[device]
pub fn get_grid_stride_1d() -> usize {
    thread::gridDim_x() as usize * thread::blockDim_x() as usize
}

#[device]
pub fn get_row_2d() -> usize {
    thread::blockIdx_y() as usize * thread::blockDim_y() as usize + thread::threadIdx_y() as usize
}

#[device]
pub fn get_col_2d() -> usize {
    thread::blockIdx_x() as usize * thread::blockDim_x() as usize + thread::threadIdx_x() as usize
}
