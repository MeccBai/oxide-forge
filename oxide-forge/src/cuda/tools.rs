use cuda_device::async_copy::{
    cp_async_ca_zfill_16, cp_async_ca_zfill_4, cp_async_ca_zfill_8, cp_async_commit_group,
    cp_async_wait_group,
};
pub mod index;
pub mod tensor;

pub(super) struct DoubleBuffer<S: Copy> {
    stages: [*mut S; 2],
    current: usize,
}

impl<S: Copy> DoubleBuffer<S> {
    /// Creates a double buffer and enqueues the first copy into stage 0.
    ///
    /// This does not commit or wait for the copy. Multiple buffers (for
    /// example, the A and B tiles of GEMM) can therefore be initialized before
    /// one shared `ready_blocking`, or `ready_async` followed by `wait`,
    /// completes the copy group.
    #[inline(always)]
    pub(super) unsafe fn initialize(
        stage0: *mut S,
        stage1: *mut S,
        destination_offset: usize,
        source: *const S,
        valid_elements: usize,
        count: usize,
    ) -> Self {
        let buffer = Self {
            stages: [stage0, stage1],
            current: 0,
        };
        unsafe { buffer.copy_current_async(destination_offset, source, valid_elements, count) };
        buffer
    }

    #[inline(always)]
    pub(super) fn current(&self) -> *mut S {
        self.stages[self.current]
    }

    #[inline(always)]
    pub(super) fn writable(&self) -> *mut S {
        self.stages[self.current ^ 1]
    }

    #[inline(always)]
    pub(super) unsafe fn copy_current_async(
        &self,
        destination_offset: usize,
        source: *const S,
        valid_elements: usize,
        count: usize,
    ) {
        let destination = unsafe { self.current().add(destination_offset) };
        unsafe { Self::copy_to(destination, source, valid_elements, count) };
    }

    /// Enqueues a copy into the non-current stage without committing it.
    #[inline(always)]
    pub(super) unsafe fn copy_async(
        &self,
        destination_offset: usize,
        source: *const S,
        valid_elements: usize,
        count: usize,
    ) {
        let destination = unsafe { self.writable().add(destination_offset) };
        unsafe { Self::copy_to(destination, source, valid_elements, count) };
    }

    /// Commits every async copy issued by this thread since the last commit and
    /// returns immediately. Call `wait` before consuming the copied stage.
    #[inline(always)]
    pub(super) unsafe fn ready_async() {
        unsafe { cp_async_commit_group() };
    }

    /// Commits the pending copies and waits until they are visible to every
    /// thread in the block.
    #[inline(always)]
    pub unsafe fn ready_blocking() {
        unsafe { Self::ready_async() };
        unsafe { Self::wait() };
    }

    /// Completes a group previously submitted by `ready_async`.
    #[inline(always)]
    pub unsafe fn wait() {
        unsafe { cp_async_wait_group(0) };
        cuda_device::thread::sync_threads();
    }

    /// Makes the completed writable stage current. Call this on every buffer
    /// participating in the same copy group after `ready_blocking` or `wait`.
    #[inline(always)]
    pub(super) fn advance(&mut self) {
        self.current ^= 1;
    }

    #[inline(always)]
    unsafe fn copy_to(destination: *mut S, source: *const S, valid_elements: usize, count: usize) {
        let element_size = core::mem::size_of::<S>();
        let total_bytes = count * element_size;
        let valid_bytes = valid_elements.min(count) * element_size;

        debug_assert!(element_size != 0);
        debug_assert!(total_bytes >= 4 && total_bytes % 4 == 0);

        let destination = destination.cast::<u8>();
        let source = source.cast::<u8>();
        let mut byte_offset = 0;

        while byte_offset < total_bytes {
            let remaining = total_bytes - byte_offset;
            let copy_bytes = if remaining >= 16 {
                16
            } else if remaining >= 8 {
                8
            } else {
                4
            };
            let readable_bytes = valid_bytes.saturating_sub(byte_offset).min(copy_bytes) as u32;

            let destination = unsafe { destination.add(byte_offset).cast::<u32>() };
            let source = if readable_bytes == 0 {
                source
            } else {
                unsafe { source.add(byte_offset) }
            };

            unsafe {
                match copy_bytes {
                    4 => cp_async_ca_zfill_4(destination, source, readable_bytes),
                    8 => cp_async_ca_zfill_8(destination, source, readable_bytes),
                    16 => cp_async_ca_zfill_16(destination, source, readable_bytes),
                    _ => core::hint::unreachable_unchecked(),
                }
            }

            byte_offset += copy_bytes;
        }
    }
}
