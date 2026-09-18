use crate::cuda::{DeviceSpan, kernels};
use cuda_core::simt::memory;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use std::collections::HashMap;
use std::sync::Arc;

pub enum InitType {
    Sequence,
    Reserve,
    Random,
    Zero,
}

impl InitType {
    pub fn is_zero(&self) -> bool {
        match self {
            Self::Sequence => false,
            Self::Reserve => false,
            Self::Random => false,
            Self::Zero => true,
        }
    }
}

pub struct CudaRuntime {
    module: kernels::LoadedModule,

    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    buffer_pool: HashMap<usize, Vec<DeviceBuffer<f32>>>,
}

impl CudaRuntime {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let module = unsafe { kernels::load(&ctx)? };

        Ok(Self {
            module,
            ctx,
            stream,
            buffer_pool: HashMap::new(),
        })
    }

    pub fn get_uninit_buffer(&mut self, size: usize) -> DeviceBuffer<f32> {
        if size > 0
            && let Some(buffer) = self.buffer_pool.get_mut(&size).and_then(Vec::pop)
        {
            return buffer;
        }

        self.allocate_uninit_buffer(size)
    }

    pub fn get_u32_uninit_buffer(&mut self, size: usize) -> DeviceBuffer<u32> {
        let buffer = unsafe { DeviceBuffer::<u32>::uninitialized_async(&self.stream, size) };
        buffer.unwrap()
    }

    fn allocate_uninit_buffer(&self, size: usize) -> DeviceBuffer<f32> {
        let buffer = unsafe { DeviceBuffer::<f32>::uninitialized_async(&self.stream, size) };
        buffer.unwrap()
    }

    pub fn get_zerod_buffer(&mut self, size: usize) -> DeviceBuffer<f32> {
        let mut buffer = self.get_uninit_buffer(size);
        buffer.zero_async(&self.stream).unwrap();
        buffer
    }

    /// Ensures that at least `count` buffers of exactly `size` elements are
    /// immediately available from the runtime pool.
    pub fn reserve_buffers(&mut self, size: usize, count: usize) {
        if size == 0 {
            return;
        }

        let available = self.buffer_pool.get(&size).map_or(0, Vec::len);
        let missing = count.saturating_sub(available);
        if missing == 0 {
            return;
        }

        let mut buffers = Vec::with_capacity(missing);
        for _ in 0..missing {
            buffers.push(self.allocate_uninit_buffer(size));
        }
        self.buffer_pool.entry(size).or_default().extend(buffers);
    }

    /// Returns a buffer to the exact-size pool without invoking its `Drop`.
    ///
    /// All pending uses must be ordered on this runtime's stream. The current
    /// container API satisfies that contract because its kernels are launched
    /// on this stream.
    pub fn recycle_buffer(&mut self, buffer: DeviceBuffer<f32>) {
        if buffer.is_empty() {
            return;
        }
        assert!(
            Arc::ptr_eq(buffer.context(), &self.ctx),
            "cannot recycle a buffer owned by another CUDA context"
        );
        self.buffer_pool
            .entry(buffer.len())
            .or_default()
            .push(buffer);
    }

    pub fn sync(&self) {
        self.stream.synchronize().unwrap();
    }

    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    pub(crate) fn execution_stream<'a>(&'a self, stream: Option<&'a CudaStream>) -> &'a CudaStream {
        if let Some(stream) = stream {
            stream.join(&self.stream).unwrap();
            stream
        } else {
            &self.stream
        }
    }

    pub fn module(&self) -> &kernels::LoadedModule {
        &self.module
    }

    pub(crate) fn get_launch_config(&self, size: usize, block_size: usize) -> LaunchConfig1D {
        LaunchConfig1D::new(
            size.div_ceil(block_size).max(1) as u32,
            block_size as u32,
            0,
        )
    }

    pub(crate) fn get_elementwise_launch_config(
        &self,
        size: usize,
        block_size: usize,
    ) -> (LaunchConfig1D, usize) {
        const MAX_ELEMENTS_PER_THREAD: usize = 4;

        let elements_per_thread = size.div_ceil(block_size).clamp(1, MAX_ELEMENTS_PER_THREAD);
        let thread_count = size.div_ceil(elements_per_thread);
        let grid_size = thread_count.div_ceil(block_size).max(1);
        let grid_size = u32::try_from(grid_size).expect("elementwise grid exceeds CUDA limits");
        let block_size = u32::try_from(block_size).expect("block size exceeds u32");

        (
            LaunchConfig1D::new(grid_size, block_size, 0),
            elements_per_thread,
        )
    }

    pub fn concat_buffers(&mut self, buffers: &[&DeviceBuffer<f32>]) -> DeviceBuffer<f32> {
        self.concat_buffers_on(buffers, None)
    }

    pub(crate) fn concat_buffers_on(
        &mut self,
        buffers: &[&DeviceBuffer<f32>],
        stream: Option<&CudaStream>,
    ) -> DeviceBuffer<f32> {
        let total_len = buffers
            .iter()
            .try_fold(0usize, |total, buffer| total.checked_add(buffer.len()))
            .expect("concatenated buffer length overflow");

        let result = self.get_uninit_buffer(total_len);
        let stream = self.execution_stream(stream);
        let mut offset = 0usize;

        for buffer in buffers {
            if buffer.len() == 0 {
                continue;
            }

            let byte_offset = offset
                .checked_mul(size_of::<f32>())
                .expect("destination offset overflow");

            let byte_len = buffer
                .len()
                .checked_mul(size_of::<f32>())
                .expect("copy size overflow");

            let destination = result
                .cu_deviceptr()
                .checked_add(byte_offset as u64)
                .expect("destination pointer overflow");

            unsafe {
                memory::memcpy_dtod_async(
                    destination,
                    buffer.cu_deviceptr(),
                    byte_len,
                    stream.cu_stream(),
                )
                .unwrap();
            }

            offset += buffer.len();
        }

        result
    }

    pub fn clone_buffer(&mut self, buffer: &DeviceBuffer<f32>) -> DeviceBuffer<f32> {
        let mut new_buffer = self.get_uninit_buffer(buffer.len());
        new_buffer
            .copy_from_device_async(buffer, &self.stream())
            .unwrap();
        new_buffer
    }

    fn span_to_buffer_async(&mut self, span: &DeviceSpan<'_, f32>) -> DeviceBuffer<f32> {
        span.to_buffer_async(self)
    }

    pub fn create_extra_streams(&self, count: usize) -> Vec<Arc<CudaStream>> {
        (0..count)
            .map(|_| self.stream.fork().unwrap())
            .collect::<Vec<_>>()
    }

    /// Refreshes the fork point for reusable streams so their subsequent work
    /// waits for everything currently queued on the primary stream.
    pub fn fork_streams(&self, streams: &[Arc<CudaStream>]) {
        for stream in streams {
            stream.join(&self.stream).unwrap();
        }
    }

    pub fn join_streams(&self, streams: &[Arc<CudaStream>]) {
        for stream in streams {
            self.stream.join(stream).unwrap();
        }
    }

    pub fn sync_streams(&self, streams: &[Arc<CudaStream>]) {
        self.join_streams(streams);
        self.sync();
    }

    pub fn clear_buffers(&mut self) {
        self.buffer_pool.clear();
    }

    pub fn clear_buffers_by_size(&mut self, size: usize) {
        self.buffer_pool.remove(&size);
    }
}
