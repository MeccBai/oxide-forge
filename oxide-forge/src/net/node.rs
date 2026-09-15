use crate::cuda::container::Matrix;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Mul,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SingleType {
    Softmax,
    RmsNorm,
    LayerNorm,
    Scale(f32),
}

/// Stateless multi-input operation used by inference paths.
pub struct BinaryNode {
    op: BinaryOp,
}

/// Stateless single-input operation used by inference paths.
pub struct SingleNode {
    op: SingleType,
}

enum BinaryCache {
    Inputs(Vec<Matrix>),
}

enum SingleCache {
    Input(Matrix),
    Output(Matrix),
}

/// Binary/multi-input node with the state required by one backward pass.
pub struct TrainingBinaryNode {
    node: BinaryNode,
    input_count: usize,
    forward_pending: bool,
    cache: Option<BinaryCache>,
}

/// Single-input node with the state required by one backward pass.
pub struct TrainingSingleNode {
    node: SingleNode,
    forward_pending: bool,
    cache: Option<SingleCache>,
}

mod binary;
mod single;
