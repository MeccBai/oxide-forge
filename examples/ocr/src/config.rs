use std::error::Error;
use std::fs;
use std::io::{Error as IoError, ErrorKind};
use std::path::{Path, PathBuf};

use serde::Deserialize;

pub const DEFAULT_CONFIG_PATH: &str = "examples/ocr/config.toml";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub model: ModelConfig,
    pub dataset: DatasetConfig,
    pub training: TrainingConfig,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub load_dir: PathBuf,
    pub save_dir: PathBuf,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetConfig {
    pub source_dir: PathBuf,
    pub target_dir: PathBuf,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub learning_rate: f32,
    pub save_every: usize,
}

impl Config {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn Error>> {
        let path = path.as_ref();
        let content = fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content).map_err(|error| {
            IoError::new(
                ErrorKind::InvalidData,
                format!("invalid OCR config {}: {error}", path.display()),
            )
        })?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), IoError> {
        if !self.training.learning_rate.is_finite() || self.training.learning_rate <= 0.0 {
            return Err(IoError::new(
                ErrorKind::InvalidInput,
                "training.learning_rate must be finite and greater than zero",
            ));
        }
        if self.training.save_every == 0 {
            return Err(IoError::new(
                ErrorKind::InvalidInput,
                "training.save_every must be greater than zero",
            ));
        }
        Ok(())
    }
}
