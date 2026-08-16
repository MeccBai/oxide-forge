use crate::cuda::{DeviceSpan, kernels};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D, memory};
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
        })
    }

    pub fn get_uninit_buffer(&self, size: usize) -> DeviceBuffer<f32> {
        let buffer = unsafe { DeviceBuffer::<f32>::uninitialized_async(&self.stream, size) };
        buffer.unwrap()
    }

    pub fn get_zerod_buffer(&self, size: usize) -> DeviceBuffer<f32> {
        let buffer = DeviceBuffer::<f32>::zeroed(&self.stream, size);
        buffer.unwrap()
    }

    pub fn sync(&self) {
        self.stream.synchronize().unwrap();
    }

    pub fn stream(&self) -> &CudaStream {
        &self.stream
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

    pub fn concat_buffers(&self, buffers: &[&DeviceBuffer<f32>]) -> DeviceBuffer<f32> {
        let total_len = buffers
            .iter()
            .try_fold(0usize, |total, buffer| total.checked_add(buffer.len()))
            .expect("concatenated buffer length overflow");

        let result = self.get_uninit_buffer(total_len);
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
                    self.stream().cu_stream(),
                )
                .unwrap();
            }

            offset += buffer.len();
        }

        self.sync();
        result
    }

    pub fn clone_buffer(&self, buffer: &DeviceBuffer<f32>) -> DeviceBuffer<f32> {
        let mut new_buffer = self.get_uninit_buffer(buffer.len());
        new_buffer
            .copy_from_device_async(buffer, &self.stream())
            .unwrap();
        self.sync();
        new_buffer
    }

    pub fn span_to_buffer_async(&self, span: &DeviceSpan<'_, f32>) -> DeviceBuffer<f32> {
        span.to_buffer_async(self)
    }

    pub fn create_extra_streams(&self, count: usize) -> Vec<Arc<CudaStream>> {
        (0..count)
            .map(|_| self.stream.fork().unwrap())
            .collect::<Vec<_>>()
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
}
