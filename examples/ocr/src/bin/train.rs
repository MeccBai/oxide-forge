use std::error::Error;
use std::fs;
use std::time::Instant;

use ocr::config::{Config, DEFAULT_CONFIG_PATH};
use ocr::model::{MASK_MODEL_NAME, VISION_MODEL_NAME};
use oxide_forge::cuda;
use oxide_forge::net::linear::Linear;

fn main() -> Result<(), Box<dyn Error>> {
    let config_path = std::env::var("OCR_CONFIG").unwrap_or_else(|_| DEFAULT_CONFIG_PATH.into());
    let config = Config::load(&config_path)?;
    let dataset = ocr::Dataset::open(&config.dataset.source_dir, &config.dataset.target_dir)?;
    let load_started = Instant::now();
    let samples = dataset.load_all()?;
    println!(
        "Loaded and expanded {} samples with 16 workers in {:?}",
        samples.len(),
        load_started.elapsed()
    );

    let epochs = config.training.epochs;
    let sample_limit = dataset.len().min(samples.len());
    let learning_rate = config.training.learning_rate;

    fs::create_dir_all(&config.model.save_dir)?;
    let mut runtime = cuda::CudaRuntime::new()?;
    let (mut transformer, mut mask_mlp) =
        ocr::model::load_or_create_training(&config.model.load_dir, &mut runtime)?;

    println!(
        "OCR mask training: {sample_limit} samples, {epochs} epoch(s), learning rate \
         {learning_rate}, save every {} epoch(s)",
        config.training.save_every
    );

    for epoch in 0..epochs {
        let epoch_started = Instant::now();
        let mut epoch_loss = 0.0;

        for (step, host_sample) in samples.iter().take(sample_limit).enumerate() {
            let sample_started = Instant::now();
            let sample = host_sample.upload(&runtime)?;
            let vision = transformer.forward(&sample.input, &mut runtime);
            let output = mask_mlp.forward(vision, &mut runtime);

            let loss_rows = Linear::loss_rows(&output, &sample.target, &mut runtime, None);
            let loss =
                loss_rows.to_host(&runtime).into_iter().sum::<f32>() / ocr::PATCH_COUNT as f32;
            epoch_loss += loss;

            let mut output_gradient = runtime.matrix_sub(&output, &sample.target);
            output_gradient.scale(
                1.0 / (ocr::PATCH_COUNT * ocr::TARGET_WIDTH) as f32,
                &runtime,
            );
            let vision_gradient = mask_mlp.backward(&output_gradient, learning_rate, &mut runtime);
            let input_gradient =
                transformer.backward(&vision_gradient, learning_rate, &mut runtime);
            runtime.sync();

            runtime.recycle_vector(loss_rows);
            runtime.recycle_matrix(input_gradient);
            runtime.recycle_matrix(vision_gradient);
            runtime.recycle_matrix(output_gradient);
            runtime.recycle_matrix(output);
            runtime.recycle_matrix(sample.input);
            runtime.recycle_matrix(sample.target);

            println!(
                "epoch {}/{epochs}, sample {}/{sample_limit} (dataset #{}), loss {:.6}, {:?}",
                epoch + 1,
                step + 1,
                sample.index,
                loss,
                sample_started.elapsed()
            );
        }

        let mean_loss = if sample_limit == 0 {
            0.0
        } else {
            epoch_loss / sample_limit as f32
        };
        println!(
            "epoch {}/{epochs} complete: mean loss {:.6}, {:?}",
            epoch + 1,
            mean_loss,
            epoch_started.elapsed()
        );

        if (epoch + 1) % config.training.save_every == 0 {
            save_checkpoint(
                &transformer,
                &mask_mlp,
                &config.model.save_dir,
                &runtime,
                epoch + 1,
            )?;
        }
    }

    if epochs > 0 && epochs % config.training.save_every != 0 {
        save_checkpoint(
            &transformer,
            &mask_mlp,
            &config.model.save_dir,
            &runtime,
            epochs,
        )?;
    }

    Ok(())
}

fn save_checkpoint(
    transformer: &oxide_forge::net::transformer::encoder::TrainingTransformer,
    mask_mlp: &oxide_forge::net::mlp::TrainingMlp,
    save_dir: &std::path::Path,
    runtime: &cuda::CudaRuntime,
    epoch: usize,
) -> Result<(), Box<dyn Error>> {
    ocr::model::save_training(transformer, mask_mlp, save_dir, runtime)?;
    println!(
        "epoch {epoch}: saved {}/{}.{{toml,bin}} and {}/{}.{{toml,bin}}",
        save_dir.display(),
        VISION_MODEL_NAME,
        save_dir.display(),
        MASK_MODEL_NAME
    );
    Ok(())
}
