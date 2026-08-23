mod attention;
pub mod decoder;
pub mod encoder;
mod inference;
pub mod position;
pub use position::PositionEncoding;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormType {
    Rms,
    #[default]
    Layer,
}
