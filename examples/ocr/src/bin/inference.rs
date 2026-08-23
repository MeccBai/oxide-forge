use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use ocr::config::{Config, DEFAULT_CONFIG_PATH};
use oxide_forge::cuda;

fn main() -> Result<(), Box<dyn Error>> {
    let config_path = std::env::var("OCR_CONFIG").unwrap_or_else(|_| DEFAULT_CONFIG_PATH.into());
    let config = Config::load(config_path)?;
    let sample_index = std::env::var("OCR_SAMPLE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let output_path = std::env::var("OCR_OUTPUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            config
                .model
                .save_dir
                .join(format!("inference-{sample_index}.png"))
        });

    let dataset = ocr::Dataset::open(&config.dataset.source_dir, &config.dataset.target_dir)?;
    let sample = dataset.load_one(sample_index)?;
    let mut runtime = cuda::CudaRuntime::new()?;
    let (mut transformer, mask_mlp) =
        ocr::model::load_inference(&config.model.load_dir, &mut runtime)?;
    let sample = sample.upload(&runtime)?;

    let started = Instant::now();
    let vision = transformer.forward(&sample.input, &mut runtime);
    let output = mask_mlp.forward(&vision, &mut runtime);
    runtime.sync();
    ocr::save_mask(&output, &output_path, &runtime)?;

    runtime.recycle_matrix(vision);
    runtime.recycle_matrix(output);
    runtime.recycle_matrix(sample.input);
    runtime.recycle_matrix(sample.target);

    println!(
        "Inference for dataset #{sample_index} completed in {:?}: {}",
        started.elapsed(),
        output_path.display()
    );
    Ok(())
}
