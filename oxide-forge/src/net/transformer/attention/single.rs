/// The single-head specialization uses the same implementation and scheduling
/// rules without exposing a second copy of the attention algorithm.
pub(crate) type Attention = super::multi::Attention<1>;
