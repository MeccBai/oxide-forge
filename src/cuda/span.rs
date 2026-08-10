use core::marker::PhantomData;

use cuda_core::DeviceBuffer;

use super::{CudaRuntime, DEFAULT_BLOCK_SIZE};

/// Copyable device-side description of a possibly-strided mutable span.
///
/// This value never owns the allocation. Keep construction private to the
/// borrowing `DeviceSpanMut` wrapper below.
#[repr(C)]
pub(super) struct DeviceSpanDescriptor<T> {
    pub ptr: *mut T,
    pub len: usize,
    pub stride: usize,
}

impl<T> Copy for DeviceSpanDescriptor<T> {}

impl<T> Clone for DeviceSpanDescriptor<T> {
    fn clone(&self) -> Self {
        *self
    }
}

/// A non-owning, exclusively borrowed region of a `DeviceBuffer`.
///
/// Dropping this value does not free device memory. The `PhantomData` keeps the
/// mutable borrow of the owning allocation active for the span's lifetime.
pub struct DeviceSpanMut<'a, T> {
    descriptor: DeviceSpanDescriptor<T>,
    _borrow: PhantomData<&'a mut [T]>,
}

impl<'a, T> DeviceSpanMut<'a, T> {
    pub fn from_buffer(buffer: &'a mut DeviceBuffer<T>, offset: usize, len: usize) -> Self {
        Self::from_buffer_strided(buffer, offset, len, 1)
    }

    pub fn from_buffer_strided(
        buffer: &'a mut DeviceBuffer<T>,
        offset: usize,
        len: usize,
        stride: usize,
    ) -> Self {
        assert!(stride > 0, "device span stride must be non-zero");
        assert!(offset <= buffer.len(), "device span offset out of bounds");

        if len > 0 {
            let last = offset
                .checked_add((len - 1).checked_mul(stride).expect("device span overflow"))
                .expect("device span overflow");
            assert!(last < buffer.len(), "device span exceeds its buffer");
        }

        let byte_offset = offset
            .checked_mul(core::mem::size_of::<T>())
            .expect("device span byte offset overflow");
        let ptr = buffer
            .cu_deviceptr()
            .checked_add(byte_offset as u64)
            .expect("device span pointer overflow") as usize as *mut T;

        Self {
            descriptor: DeviceSpanDescriptor { ptr, len, stride },
            _borrow: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.descriptor.len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(super) fn descriptor(&self) -> DeviceSpanDescriptor<T> {
        self.descriptor
    }

    /// Splits one allocation into disjoint, contiguous mutable spans.
    pub(super) fn split_contiguous(
        buffer: &'a mut DeviceBuffer<T>,
        chunk_len: usize,
        chunks: usize,
    ) -> Vec<Self> {
        let covered = chunk_len
            .checked_mul(chunks)
            .expect("device span partition overflow");
        assert!(
            covered <= buffer.len(),
            "device span partition exceeds buffer"
        );

        let base = buffer.cu_deviceptr();
        let elem_size = core::mem::size_of::<T>();
        let mut spans = Vec::with_capacity(chunks);

        for index in 0..chunks {
            let offset = index
                .checked_mul(chunk_len)
                .expect("device span partition overflow");
            let byte_offset = offset
                .checked_mul(elem_size)
                .expect("device span partition overflow");
            let ptr = base
                .checked_add(byte_offset as u64)
                .expect("device span pointer overflow") as usize as *mut T;

            spans.push(Self {
                descriptor: DeviceSpanDescriptor {
                    ptr,
                    len: chunk_len,
                    stride: 1,
                },
                _borrow: PhantomData,
            });
        }

        spans
    }
}

impl DeviceSpanMut<'_, f32> {
    pub fn for_each<F>(&mut self, runtime: &CudaRuntime, f: F)
    where
        F: Fn(f32) -> f32 + Copy,
    {
        if self.is_empty() {
            return;
        }

        let config = runtime.get_launch_config(self.len(), DEFAULT_BLOCK_SIZE);
        let prepared = runtime.module.prepare_span_for_each::<F>(config).unwrap();

        unsafe {
            runtime
                .module
                .span_for_each::<F>(&runtime.stream, &prepared, self.descriptor(), f)
                .unwrap();
        }
        runtime.sync();
    }
}
