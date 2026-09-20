mod attention;
mod inference;
pub mod position;
mod training;

pub use inference::TransformerMetadata;
pub use position::{PositionEncoding, PositionEncodingMetadata};
pub type InferenceEncoder<const HEADS: usize = 1> = inference::InferenceTransformer<HEADS, false>;
pub type InferenceDecoder<const HEADS: usize = 1> = inference::InferenceTransformer<HEADS, true>;
pub type TrainingEncoder<const HEADS: usize = 1> = training::TrainingTransformer<HEADS, false>;
pub type TrainingDecoder<const HEADS: usize = 1> = training::TrainingTransformer<HEADS, true>;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormType {
    Rms,
    #[default]
    Layer,
}
