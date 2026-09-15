use crate::cuda::container::Matrix;
use crate::net::linear::Activation;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SingleType {
    Activation(Activation),
    Softmax,
    RmsNorm,
    LayerNorm,
    Scale(f32),
    Transpose,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixAxis {
    Rows,
    Columns,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowReduction {
    Sum,
    Mean,
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

/// Concatenates matrices along rows or columns into contiguous storage.
pub struct ConcatNode {
    axis: MatrixAxis,
}

/// Concat node that retains only input shapes for its split backward pass.
pub struct TrainingConcatNode {
    node: ConcatNode,
    input_shapes: Option<Vec<(usize, usize)>>,
}

/// Physically splits a Matrix into independently owned contiguous matrices.
pub struct SplitNode {
    axis: MatrixAxis,
    sizes: Vec<usize>,
}

/// Split node that validates forward/backward pairing.
pub struct TrainingSplitNode {
    node: SplitNode,
    forward_pending: bool,
}

/// Reduces every Matrix row and returns a `[rows, 1]` Matrix.
pub struct RowReduceNode {
    reduction: RowReduction,
}

/// Row reduction with the input width required to broadcast its gradient.
pub struct TrainingRowReduceNode {
    node: RowReduceNode,
    input_shape: Option<(usize, usize)>,
}

mod binary;
mod reduction;
mod single;
mod structural;
