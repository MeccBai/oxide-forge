mod io;
mod model;

pub use model::{
    dump_inference_mlp, dump_linear, dump_mlp, dump_training_mlp, dump_training_transformer,
    dump_transformer, load_inference_mlp, load_linear, load_mlp, load_training_mlp,
    load_training_transformer, load_transformer,
};

pub type CheckpointResult<T> = Result<T, Box<dyn std::error::Error>>;
