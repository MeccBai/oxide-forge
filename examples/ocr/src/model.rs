use std::error::Error;
use std::io::{Error as IoError, ErrorKind};
use std::path::{Path, PathBuf};

use oxide_forge::cuda::InitType::{Random, Zero};
use oxide_forge::cuda::container::Matrix;
use oxide_forge::cuda::runtime::CudaRuntime;
use oxide_forge::net::checkpoint;
use oxide_forge::net::linear::Activation::{Gelu, Identity};
use oxide_forge::net::linear::Linear;
use oxide_forge::net::mlp::{InferenceMLP, TrainingMlp};
use oxide_forge::net::transformer::NormType;
use oxide_forge::net::transformer::encoder::{InferenceTransformer, TrainingTransformer};

use crate::{INPUT_WIDTH, PATCH_COUNT, PATCHES_PER_SIDE, TARGET_WIDTH};

pub const VISION_MODEL_NAME: &str = "ocr_vit";
pub const MASK_MODEL_NAME: &str = "ocr_mask_mlp";

pub fn load_or_create_training(
    load_dir: impl AsRef<Path>,
    runtime: &mut CudaRuntime,
) -> Result<(TrainingTransformer, TrainingMlp), Box<dyn Error>> {
    let vision_model = model_path(load_dir.as_ref(), VISION_MODEL_NAME);
    let mask_model = model_path(load_dir.as_ref(), MASK_MODEL_NAME);

    let transformer = if checkpoint_exists(&vision_model)? {
        println!("Loading {}.toml", vision_model.display());
        checkpoint::load_training_transformer(&vision_model, position_encoding(runtime), runtime)?
    } else {
        println!(
            "No {} checkpoint; creating a random model",
            vision_model.display()
        );
        new_vision_transformer(runtime)
    };

    let mask_mlp = if checkpoint_exists(&mask_model)? {
        println!("Loading {}.toml", mask_model.display());
        checkpoint::load_training_mlp(&mask_model, runtime)?
    } else {
        println!(
            "No {} checkpoint; creating a random model",
            mask_model.display()
        );
        new_mask_mlp(runtime)
    };

    Ok((transformer, mask_mlp))
}

pub fn load_inference(
    load_dir: impl AsRef<Path>,
    runtime: &mut CudaRuntime,
) -> Result<(InferenceTransformer, InferenceMLP), Box<dyn Error>> {
    let vision_model = model_path(load_dir.as_ref(), VISION_MODEL_NAME);
    let mask_model = model_path(load_dir.as_ref(), MASK_MODEL_NAME);
    require_checkpoint(&vision_model)?;
    require_checkpoint(&mask_model)?;
    let transformer =
        checkpoint::load_transformer(&vision_model, position_encoding(runtime), runtime)?;
    let mask_mlp = checkpoint::load_inference_mlp(&mask_model, runtime)?;
    Ok((transformer, mask_mlp))
}

pub fn save_training(
    transformer: &TrainingTransformer,
    mask_mlp: &TrainingMlp,
    save_dir: impl AsRef<Path>,
    runtime: &CudaRuntime,
) -> Result<(), Box<dyn Error>> {
    let vision_model = model_path(save_dir.as_ref(), VISION_MODEL_NAME);
    let mask_model = model_path(save_dir.as_ref(), MASK_MODEL_NAME);
    checkpoint::dump_training_transformer(transformer, &vision_model, runtime)?;
    checkpoint::dump_training_mlp(mask_mlp, &mask_model, runtime)?;
    Ok(())
}

pub fn model_path(directory: &Path, name: &str) -> PathBuf {
    directory.join(name)
}

fn initialized_matrix(runtime: &mut CudaRuntime, rows: usize, cols: usize) -> Matrix {
    let mut matrix = runtime.new_matrix(Random, rows, cols);
    let limit = (6.0_f32 / (rows + cols) as f32).sqrt();
    matrix.for_each(runtime, move |value| (value * 2.0 - 1.0) * limit);
    matrix
}

fn sinusoidal_position(runtime: &CudaRuntime) -> Matrix {
    let mut values = vec![0.0; PATCH_COUNT * INPUT_WIDTH];
    let axis_width = INPUT_WIDTH / 2;
    for patch_y in 0..PATCHES_PER_SIDE {
        for patch_x in 0..PATCHES_PER_SIDE {
            let row = (patch_y * PATCHES_PER_SIDE + patch_x) * INPUT_WIDTH;
            for pair in 0..axis_width / 2 {
                let divisor = 10_000.0_f32.powf((2 * pair) as f32 / axis_width as f32);
                let y_angle = patch_y as f32 / divisor;
                let x_angle = patch_x as f32 / divisor;
                values[row + pair * 2] = y_angle.sin();
                values[row + pair * 2 + 1] = y_angle.cos();
                values[row + axis_width + pair * 2] = x_angle.sin();
                values[row + axis_width + pair * 2 + 1] = x_angle.cos();
            }
        }
    }
    runtime
        .matrix_from_host(&values, PATCH_COUNT, INPUT_WIDTH)
        .unwrap()
}

fn position_encoding(
    runtime: &CudaRuntime,
) -> impl Fn(&Matrix, &mut CudaRuntime) -> Matrix + 'static {
    let position = sinusoidal_position(runtime);
    move |input: &Matrix, runtime: &mut CudaRuntime| runtime.matrix_add(input, &position)
}

fn new_vision_transformer(runtime: &mut CudaRuntime) -> TrainingTransformer {
    let position_encoding = position_encoding(runtime);
    let query = Linear::new(
        initialized_matrix(runtime, INPUT_WIDTH, INPUT_WIDTH),
        None,
        Identity,
    );
    let key = Linear::new(
        initialized_matrix(runtime, INPUT_WIDTH, INPUT_WIDTH),
        None,
        Identity,
    );
    let value = Linear::new(
        initialized_matrix(runtime, INPUT_WIDTH, INPUT_WIDTH),
        None,
        Identity,
    );
    let feed_forward = TrainingMlp::new(
        vec![
            Linear::new(
                initialized_matrix(runtime, INPUT_WIDTH, INPUT_WIDTH * 4),
                Some(runtime.new_vector(Zero, INPUT_WIDTH * 4)),
                Gelu,
            ),
            Linear::new(
                initialized_matrix(runtime, INPUT_WIDTH * 4, INPUT_WIDTH),
                Some(runtime.new_vector(Zero, INPUT_WIDTH)),
                Identity,
            ),
        ],
        None,
    );
    let output = Linear::new(
        initialized_matrix(runtime, INPUT_WIDTH, INPUT_WIDTH),
        Some(runtime.new_vector(Zero, INPUT_WIDTH)),
        Identity,
    );
    TrainingTransformer::new(
        query,
        key,
        value,
        position_encoding,
        feed_forward,
        output,
        NormType::Rms,
    )
}

fn new_mask_mlp(runtime: &mut CudaRuntime) -> TrainingMlp {
    TrainingMlp::new(
        vec![
            Linear::new(
                initialized_matrix(runtime, INPUT_WIDTH, INPUT_WIDTH * 4),
                Some(runtime.new_vector(Zero, INPUT_WIDTH * 4)),
                Gelu,
            ),
            Linear::new(
                initialized_matrix(runtime, INPUT_WIDTH * 4, TARGET_WIDTH),
                Some(runtime.new_vector(Zero, TARGET_WIDTH)),
                Identity,
            ),
        ],
        None,
    )
}

fn checkpoint_exists(base: &Path) -> Result<bool, IoError> {
    let metadata_exists = base.with_extension("toml").is_file();
    let data_exists = base.with_extension("bin").is_file();
    match (metadata_exists, data_exists) {
        (true, true) => Ok(true),
        (false, false) => Ok(false),
        _ => Err(incomplete_checkpoint(base)),
    }
}

fn require_checkpoint(base: &Path) -> Result<(), IoError> {
    if checkpoint_exists(base)? {
        Ok(())
    } else {
        Err(IoError::new(
            ErrorKind::NotFound,
            format!("checkpoint {}.{{toml,bin}} does not exist", base.display()),
        ))
    }
}

fn incomplete_checkpoint(base: &Path) -> IoError {
    IoError::new(
        ErrorKind::InvalidData,
        format!(
            "incomplete checkpoint {}: .toml and .bin must both exist",
            base.display()
        ),
    )
}
