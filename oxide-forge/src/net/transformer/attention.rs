use crate::net::linear::Linear;
use cuda_core::CudaStream;
use std::sync::Arc;

pub(super) struct Qkv<T> {
    pub query: T,
    pub key: T,
    pub value: T,
}

pub(super) struct QkvProjector {
    layers: Qkv<Linear>,
    streams: Option<[Arc<CudaStream>; 3]>,
}

pub mod multi;
