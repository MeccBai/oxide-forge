mod attention;
mod inference;
pub mod position;

pub use inference::TransformerMetadata;
pub use position::{PositionEncoding, PositionEncodingMetadata};
pub type Encoder<const HEADS: usize = 1> = inference::Transformer<HEADS, false>;
pub type Decoder<const HEADS: usize = 1> = inference::Transformer<HEADS, true>;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormType {
    Rms,
    #[default]
    Layer,
}
