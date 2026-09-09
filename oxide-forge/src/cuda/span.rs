use core::marker::PhantomData;

use cuda_core::simt::memory;
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};

use super::{CudaRuntime, DEFAULT_BLOCK_SIZE};

#[repr(C)]
pub(super) struct DeviceSliceDescriptor<T> {
    ptr: *const T,
    len: usize,
}

impl<T> Copy for DeviceSliceDescriptor<T> {}

impl<T> Clone for DeviceSliceDescriptor<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Copy> DeviceSliceDescriptor<T> {
    /// Reads an element after the caller has established the device index.
    ///
    /// Descriptors are crate-private and may only be constructed from a
    /// validated span. Device code must keep `index < len()` true.
    #[inline(always)]
    pub(super) fn read(&self, index: usize) -> T {
        debug_assert!(index < self.len);
        unsafe { self.ptr.add(index).read() }
    }
}

impl<T> DeviceSliceDescriptor<T> {
    #[inline(always)]
    pub(super) fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub(super) fn as_ptr(&self) -> *const T {
        self.ptr
    }
}

#[repr(C)]
pub(super) struct DeviceSliceMutDescriptor<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> Copy for DeviceSliceMutDescriptor<T> {}

impl<T> Clone for DeviceSliceMutDescriptor<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Copy> DeviceSliceMutDescriptor<T> {
    /// Reads an element under the same index invariant as [`Self::write`].
    #[inline(always)]
    pub(super) fn read(&self, index: usize) -> T {
        debug_assert!(index < self.len);
        unsafe { self.ptr.add(index).read() }
    }
}

impl<T> DeviceSliceMutDescriptor<T> {
    #[inline(always)]
    pub(super) fn len(&self) -> usize {
        self.len
    }

    /// Writes an element after device code has established `index < len()`.
    #[inline(always)]
    pub(super) fn write(&self, index: usize, value: T) {
        debug_assert!(index < self.len);
        unsafe { self.ptr.add(index).write(value) };
    }
}

/// A non-owning shared borrow of a contiguous `DeviceBuffer` region.
pub(crate) struct DeviceSpan<'a, T> {
    ptr: *const T,
    len: usize,
    _borrow: PhantomData<&'a [T]>,
}

impl<'a, T> DeviceSpan<'a, T> {
    pub(super) fn from_buffer(buffer: &'a DeviceBuffer<T>, offset: usize, len: usize) -> Self {
        check_range(buffer.len(), offset, len);
        let ptr = offset_ptr::<T>(buffer.cu_deviceptr(), offset);

        Self {
            ptr: ptr as usize as *const T,
            len,
            _borrow: PhantomData,
        }
    }
    pub(super) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn descriptor(&self) -> DeviceSliceDescriptor<T> {
        DeviceSliceDescriptor {
            ptr: self.ptr,
            len: self.len,
        }
    }

    fn split(self, chunk_size: usize) -> Vec<Self> {
        split_spans::<T>(self.ptr as usize as u64, self.len, chunk_size)
            .into_iter()
            .map(|(ptr, len)| Self {
                ptr: ptr as usize as *const T,
                len,
                _borrow: PhantomData,
            })
            .collect()
    }

    pub(super) fn chunks(buffer: &'a DeviceBuffer<T>, chunk_size: usize) -> Vec<Self> {
        Self::from_buffer(buffer, 0, buffer.len()).split(chunk_size)
    }

    pub fn to_buffer_async(&self, runtime: &mut CudaRuntime) -> DeviceBuffer<f32> {
        copy_to_buffer_async(self.ptr as usize as u64, self.len, runtime)
    }
}

impl DeviceSpan<'_, f32> {
    pub fn to_buffer(&self, runtime: &mut CudaRuntime) -> DeviceBuffer<f32> {
        copy_to_buffer(self.ptr as usize as u64, self.len, runtime)
    }

    pub fn sum(&self, runtime: &mut CudaRuntime) -> f32 {
        self.map_reduce(runtime, 0.0, move |value| value, move |lhs, rhs| lhs + rhs)
    }

    pub fn max(&self, runtime: &mut CudaRuntime) -> f32 {
        self.map_reduce(
            runtime,
            f32::NEG_INFINITY,
            move |value| value,
            move |lhs, rhs| lhs.max(rhs),
        )
    }

    pub fn map_sum<F>(&self, runtime: &mut CudaRuntime, f: F) -> f32
    where
        F: Fn(f32) -> f32 + Copy,
    {
        self.map_reduce(runtime, 0.0, f, move |lhs, rhs| lhs + rhs)
    }

    pub fn map_reduce<FM, FR>(
        &self,
        runtime: &mut CudaRuntime,
        identity: f32,
        map: FM,
        reduce: FR,
    ) -> f32
    where
        FM: Fn(f32) -> f32 + Copy,
        FR: Fn(f32, f32) -> f32 + Copy,
    {
        map_reduce_descriptor(self.descriptor(), runtime, identity, map, reduce)
    }

    pub fn zip_map_reduce<FM, FR>(
        &self,
        rhs: &DeviceSpan<'_, f32>,
        runtime: &mut CudaRuntime,
        identity: f32,
        map: FM,
        reduce: FR,
    ) -> f32
    where
        FM: Fn(f32, f32) -> f32 + Copy,
        FR: Fn(f32, f32) -> f32 + Copy,
    {
        zip_map_reduce_descriptors(
            self.descriptor(),
            rhs.descriptor(),
            runtime,
            identity,
            map,
            reduce,
        )
    }
}

impl<'a> Clone for DeviceSpan<'a, f32> {
    fn clone(&self) -> Self {
        DeviceSpan {
            ptr: self.ptr.clone(),
            len: self.len,
            _borrow: self._borrow.clone(),
        }
    }
}

/// A non-owning, exclusively borrowed region of a `DeviceBuffer`.
///
/// Dropping this value does not free device memory. The `PhantomData` keeps the
/// mutable borrow of the owning allocation active for the span's lifetime.
pub(crate) struct DeviceSpanMut<'a, T> {
    descriptor: DeviceSliceMutDescriptor<T>,
    _borrow: PhantomData<&'a mut [T]>,
}

impl<'a, T> DeviceSpanMut<'a, T> {
    pub(super) fn from_buffer(buffer: &'a mut DeviceBuffer<T>, offset: usize, len: usize) -> Self {
        check_range(buffer.len(), offset, len);
        let ptr = offset_ptr::<T>(buffer.cu_deviceptr(), offset) as usize as *mut T;

        Self {
            descriptor: DeviceSliceMutDescriptor { ptr, len },
            _borrow: PhantomData,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.descriptor.len
    }

    pub(crate) fn into_span(self) -> DeviceSpan<'a, T> {
        DeviceSpan {
            ptr: self.descriptor.ptr,
            len: self.descriptor.len,
            _borrow: PhantomData,
        }
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(super) fn descriptor(&self) -> DeviceSliceMutDescriptor<T> {
        self.descriptor
    }

    pub(super) fn read_descriptor(&self) -> DeviceSliceDescriptor<T> {
        DeviceSliceDescriptor {
            ptr: self.descriptor.ptr,
            len: self.descriptor.len,
        }
    }

    /// Consumes this span and partitions it into disjoint contiguous chunks.
    /// The last chunk may be shorter than `chunk_size`.
    fn split(self, chunk_size: usize) -> Vec<Self> {
        split_spans::<T>(
            self.descriptor.ptr as usize as u64,
            self.descriptor.len,
            chunk_size,
        )
        .into_iter()
        .map(|(ptr, len)| Self {
            descriptor: DeviceSliceMutDescriptor {
                ptr: ptr as usize as *mut T,
                len,
            },
            _borrow: PhantomData,
        })
        .collect()
    }

    /// Borrows an entire buffer and partitions it into mutable chunks.
    /// The last chunk may be shorter than `chunk_size`.
    pub(super) fn chunks(buffer: &'a mut DeviceBuffer<T>, chunk_size: usize) -> Vec<Self> {
        let len = buffer.len();
        Self::from_buffer(buffer, 0, len).split(chunk_size)
    }
}

impl DeviceSpanMut<'_, f32> {
    /// Copies this span into a new independently-owned device buffer.
    pub fn to_buffer(&self, runtime: &mut CudaRuntime) -> DeviceBuffer<f32> {
        copy_to_buffer(
            self.descriptor.ptr as usize as u64,
            self.descriptor.len,
            runtime,
        )
    }

    pub fn to_buffer_async(&self, runtime: &mut CudaRuntime) -> DeviceBuffer<f32> {
        copy_to_buffer_async(
            self.descriptor.ptr as usize as u64,
            self.descriptor.len,
            runtime,
        )
    }

    pub fn for_each<F>(&mut self, runtime: &CudaRuntime, f: F)
    where
        F: Fn(f32) -> f32 + Copy,
    {
        self.for_each_on(runtime, runtime.stream(), f);
    }

    pub(crate) fn for_each_on<F>(&mut self, runtime: &CudaRuntime, stream: &CudaStream, f: F)
    where
        F: Fn(f32) -> f32 + Copy,
    {
        if self.is_empty() {
            return;
        }

        let config = runtime.get_launch_config(self.len(), DEFAULT_BLOCK_SIZE);
        let prepared = runtime
            .module()
            .prepare_slice_for_each::<F>(config)
            .unwrap();

        runtime
            .module()
            .slice_for_each::<F>(stream, &prepared, self.descriptor(), f)
            .unwrap();
    }

    pub fn scale(&mut self, value: f32, runtime: &CudaRuntime) {
        self.for_each(runtime, move |x| x * value);
    }

    pub fn sum(&self, runtime: &mut CudaRuntime) -> f32 {
        self.map_reduce(runtime, 0.0, move |value| value, move |lhs, rhs| lhs + rhs)
    }

    pub fn max(&self, runtime: &mut CudaRuntime) -> f32 {
        self.map_reduce(
            runtime,
            f32::NEG_INFINITY,
            move |value| value,
            move |lhs, rhs| lhs.max(rhs),
        )
    }

    pub fn map_sum<F>(&self, runtime: &mut CudaRuntime, f: F) -> f32
    where
        F: Fn(f32) -> f32 + Copy,
    {
        self.map_reduce(runtime, 0.0, f, move |lhs, rhs| lhs + rhs)
    }

    pub fn map_reduce<FM, FR>(
        &self,
        runtime: &mut CudaRuntime,
        identity: f32,
        map: FM,
        reduce: FR,
    ) -> f32
    where
        FM: Fn(f32) -> f32 + Copy,
        FR: Fn(f32, f32) -> f32 + Copy,
    {
        map_reduce_descriptor(self.read_descriptor(), runtime, identity, map, reduce)
    }

    pub fn zip_map_reduce<FM, FR>(
        &self,
        rhs: &DeviceSpanMut<'_, f32>,
        runtime: &mut CudaRuntime,
        identity: f32,
        map: FM,
        reduce: FR,
    ) -> f32
    where
        FM: Fn(f32, f32) -> f32 + Copy,
        FR: Fn(f32, f32) -> f32 + Copy,
    {
        zip_map_reduce_descriptors(
            self.read_descriptor(),
            rhs.read_descriptor(),
            runtime,
            identity,
            map,
            reduce,
        )
    }
}

fn zip_map_reduce_descriptors<FM, FR>(
    lhs: DeviceSliceDescriptor<f32>,
    rhs: DeviceSliceDescriptor<f32>,
    runtime: &mut CudaRuntime,
    identity: f32,
    map: FM,
    reduce: FR,
) -> f32
where
    FM: Fn(f32, f32) -> f32 + Copy,
    FR: Fn(f32, f32) -> f32 + Copy,
{
    assert_eq!(lhs.len(), rhs.len(), "zip-map-reduce length mismatch");
    if lhs.len() == 0 {
        return identity;
    }

    let input = launch_zip_map_reduce(lhs, rhs, runtime, identity, map, reduce);
    let result = input.to_host_vec(runtime.stream()).unwrap()[0];
    runtime.recycle_buffer(input);
    result
}

fn map_reduce_descriptor<FM, FR>(
    source: DeviceSliceDescriptor<f32>,
    runtime: &mut CudaRuntime,
    identity: f32,
    map: FM,
    reduce: FR,
) -> f32
where
    FM: Fn(f32) -> f32 + Copy,
    FR: Fn(f32, f32) -> f32 + Copy,
{
    if source.len() == 0 {
        return identity;
    }

    let input = launch_map_reduce(source, source.len(), runtime, identity, map, reduce);
    let result = input.to_host_vec(runtime.stream()).unwrap()[0];
    runtime.recycle_buffer(input);
    result
}

fn launch_map_reduce<FM, FR>(
    source: DeviceSliceDescriptor<f32>,
    source_len: usize,
    runtime: &mut CudaRuntime,
    identity: f32,
    map: FM,
    reduce: FR,
) -> DeviceBuffer<f32>
where
    FM: Fn(f32) -> f32 + Copy,
    FR: Fn(f32, f32) -> f32 + Copy,
{
    let elements_per_thread = source_len.div_ceil(DEFAULT_BLOCK_SIZE);
    let shared_mem_bytes = 32 * size_of::<f32>() as u32;
    let config = LaunchConfig1D::new(1, DEFAULT_BLOCK_SIZE as u32, shared_mem_bytes);
    let mut output = runtime.get_uninit_buffer(1);
    let output_span = DeviceSpanMut::from_buffer(&mut output, 0, 1);
    let prepared = runtime
        .module()
        .prepare_map_reduce::<FM, FR>(config)
        .unwrap();
    runtime
        .module()
        .map_reduce::<FM, FR>(
            runtime.stream(),
            &prepared,
            source,
            output_span.descriptor(),
            elements_per_thread,
            map,
            reduce,
            identity,
        )
        .unwrap();
    output
}

fn launch_zip_map_reduce<FM, FR>(
    lhs: DeviceSliceDescriptor<f32>,
    rhs: DeviceSliceDescriptor<f32>,
    runtime: &mut CudaRuntime,
    identity: f32,
    map: FM,
    reduce: FR,
) -> DeviceBuffer<f32>
where
    FM: Fn(f32, f32) -> f32 + Copy,
    FR: Fn(f32, f32) -> f32 + Copy,
{
    let elements_per_thread = lhs.len().div_ceil(DEFAULT_BLOCK_SIZE);
    let shared_mem_bytes = 32 * size_of::<f32>() as u32;
    let config = LaunchConfig1D::new(1, DEFAULT_BLOCK_SIZE as u32, shared_mem_bytes);
    let mut output = runtime.get_uninit_buffer(1);
    let output_span = DeviceSpanMut::from_buffer(&mut output, 0, 1);
    let prepared = runtime
        .module()
        .prepare_zip_map_reduce::<FM, FR>(config)
        .unwrap();
    runtime
        .module()
        .zip_map_reduce::<FM, FR>(
            runtime.stream(),
            &prepared,
            lhs,
            rhs,
            output_span.descriptor(),
            elements_per_thread,
            map,
            reduce,
            identity,
        )
        .unwrap();
    output
}

fn check_range(buffer_len: usize, offset: usize, len: usize) {
    assert!(offset <= buffer_len, "device span offset out of bounds");
    let end = offset.checked_add(len).expect("device span overflow");
    assert!(end <= buffer_len, "device span exceeds its buffer");
}

fn offset_ptr<T>(base: u64, offset: usize) -> u64 {
    let byte_offset = offset
        .checked_mul(core::mem::size_of::<T>())
        .expect("device span byte offset overflow");
    base.checked_add(byte_offset as u64)
        .expect("device span pointer overflow")
}

fn split_spans<T>(base: u64, len: usize, chunk_size: usize) -> Vec<(u64, usize)> {
    assert!(chunk_size > 0, "device span chunk size must be non-zero");

    let chunk_count = len.div_ceil(chunk_size);
    let mut spans = Vec::with_capacity(chunk_count);
    for index in 0..chunk_count {
        let offset = index
            .checked_mul(chunk_size)
            .expect("device span partition overflow");
        spans.push((offset_ptr::<T>(base, offset), chunk_size.min(len - offset)));
    }
    spans
}

fn copy_to_buffer(src: u64, len: usize, runtime: &mut CudaRuntime) -> DeviceBuffer<f32> {
    let result = runtime.get_uninit_buffer(len);
    if len == 0 {
        return result;
    }

    let byte_len = len
        .checked_mul(core::mem::size_of::<f32>())
        .expect("device span copy size overflow");
    unsafe {
        memory::memcpy_dtod_async(
            result.cu_deviceptr(),
            src,
            byte_len,
            runtime.stream().cu_stream(),
        )
        .unwrap();
    }
    result
}

fn copy_to_buffer_async(src: u64, len: usize, runtime: &mut CudaRuntime) -> DeviceBuffer<f32> {
    let result = runtime.get_uninit_buffer(len);
    if len == 0 {
        return result;
    }

    let byte_len = len
        .checked_mul(core::mem::size_of::<f32>())
        .expect("device span copy size overflow");
    unsafe {
        memory::memcpy_dtod_async(
            result.cu_deviceptr(),
            src,
            byte_len,
            runtime.stream().cu_stream(),
        )
        .unwrap();
    }
    result
}

impl CudaRuntime {
    pub(crate) fn concat_buffers_from_span(
        &mut self,
        spans: &[DeviceSpan<'_, f32>],
    ) -> DeviceBuffer<f32> {
        let total_len = spans
            .iter()
            .try_fold(0usize, |total, span| total.checked_add(span.len))
            .expect("concatenated device span length overflow");
        let result = self.get_uninit_buffer(total_len);
        let mut destination_offset = 0usize;

        for span in spans {
            if span.len == 0 {
                continue;
            }

            let destination = offset_ptr::<f32>(result.cu_deviceptr(), destination_offset);
            let byte_len = span
                .len
                .checked_mul(core::mem::size_of::<f32>())
                .expect("device span copy size overflow");

            unsafe {
                memory::memcpy_dtod_async(
                    destination,
                    span.ptr as usize as u64,
                    byte_len,
                    self.stream().cu_stream(),
                )
                .unwrap();
            }

            destination_offset = destination_offset
                .checked_add(span.len)
                .expect("concatenated device span offset overflow");
        }

        result
    }
}
