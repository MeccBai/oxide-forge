use crate::cuda::{container::Matrix, runtime::CudaRuntime};

/// A non-parameterized positional encoding that creates a positioned Matrix.
///
/// The returned Matrix must have the same shape as the input. Training assumes
/// the encoding is additive with respect to the input, so its input derivative
/// is the identity.
pub type PositionEncoding = Box<dyn Fn(&Matrix, &mut CudaRuntime) -> Matrix>;
