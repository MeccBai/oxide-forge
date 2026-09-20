mod io;
mod model;

pub use model::{dump_linear, dump_mlp, dump_transformer, load_linear, load_mlp, load_transformer};

pub type CheckpointResult<T> = Result<T, Box<dyn std::error::Error>>;
