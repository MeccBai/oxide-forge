pub mod config;
pub mod model;
mod patch;

use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::io::{Error as IoError, ErrorKind};
use std::path::{Path, PathBuf};
use std::thread;

use image::ImageReader;

use oxide_forge::cuda::container::Matrix;
use oxide_forge::cuda::runtime::CudaRuntime;

pub use patch::{INPUT_WIDTH, PATCH_COUNT, PATCHES_PER_SIDE, TARGET_WIDTH};

pub const SAMPLE_COUNT: usize = 256;

pub type OcrResult<T> = Result<T, OcrError>;

#[derive(Debug)]
pub struct OcrError(String);

impl Display for OcrError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for OcrError {}

impl From<IoError> for OcrError {
    fn from(error: IoError) -> Self {
        Self(error.to_string())
    }
}

impl From<image::ImageError> for OcrError {
    fn from(error: image::ImageError) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone)]
pub struct Dataset {
    source_dir: PathBuf,
    target_dir: PathBuf,
}

pub struct HostSample {
    pub index: usize,
    input: Vec<f32>,
    target: Vec<f32>,
}

pub struct DeviceSample {
    pub index: usize,
    pub input: Matrix,
    pub target: Matrix,
}

impl Dataset {
    pub fn open(source_dir: impl Into<PathBuf>, target_dir: impl Into<PathBuf>) -> OcrResult<Self> {
        let dataset = Self {
            source_dir: source_dir.into(),
            target_dir: target_dir.into(),
        };
        for index in 0..SAMPLE_COUNT {
            require_file(&dataset.image_path(index))?;
            require_file(&dataset.mask_path(index))?;
        }
        Ok(dataset)
    }

    pub fn len(&self) -> usize {
        SAMPLE_COUNT
    }

    pub fn load_all(&self) -> OcrResult<Vec<HostSample>> {
        const WORKERS: usize = 16;
        const SAMPLES_PER_WORKER: usize = SAMPLE_COUNT / WORKERS;

        let indices = (0..SAMPLE_COUNT).collect::<Vec<_>>();
        let mut workers = Vec::with_capacity(WORKERS);

        for chunk in indices.chunks(SAMPLES_PER_WORKER) {
            let dataset = self.clone();
            let chunk = chunk.to_vec();
            workers.push(thread::spawn(move || {
                chunk
                    .into_iter()
                    .map(|index| dataset.load_host(index))
                    .collect::<OcrResult<Vec<_>>>()
            }));
        }

        let mut samples = Vec::with_capacity(SAMPLE_COUNT);
        for worker in workers {
            let mut chunk = worker
                .join()
                .map_err(|_| OcrError("dataset worker panicked".to_owned()))??;
            samples.append(&mut chunk);
        }
        samples.sort_unstable_by_key(|sample| sample.index);
        Ok(samples)
    }

    pub fn load_one(&self, index: usize) -> OcrResult<HostSample> {
        self.load_host(index)
    }

    fn load_host(&self, index: usize) -> OcrResult<HostSample> {
        if index >= SAMPLE_COUNT {
            return Err(IoError::new(
                ErrorKind::InvalidInput,
                format!("sample index {index} is outside 0..{SAMPLE_COUNT}"),
            )
            .into());
        }

        let image = ImageReader::open(self.image_path(index))?
            .decode()?
            .into_rgb8();
        let mask = ImageReader::open(self.mask_path(index))?
            .decode()?
            .into_luma8();
        validate_dimensions(index, image.width(), image.height(), "image")?;
        validate_dimensions(index, mask.width(), mask.height(), "mask")?;

        let input = patch::encode_image(&image);
        let target = patch::encode_mask(&mask);
        Ok(HostSample {
            index,
            input,
            target,
        })
    }

    fn image_path(&self, index: usize) -> PathBuf {
        self.source_dir.join(format!("{index}.jpg"))
    }

    fn mask_path(&self, index: usize) -> PathBuf {
        self.target_dir.join(format!("{index}.png"))
    }
}

impl HostSample {
    pub fn upload(&self, runtime: &CudaRuntime) -> OcrResult<DeviceSample> {
        Ok(DeviceSample {
            index: self.index,
            input: runtime
                .matrix_from_host(&self.input, PATCH_COUNT, INPUT_WIDTH)
                .map_err(|error| OcrError(error.to_string()))?,
            target: runtime
                .matrix_from_host(&self.target, PATCH_COUNT, TARGET_WIDTH)
                .map_err(|error| OcrError(error.to_string()))?,
        })
    }
}

pub fn save_mask(matrix: &Matrix, path: impl AsRef<Path>, runtime: &CudaRuntime) -> OcrResult<()> {
    assert_eq!(matrix.rows(), PATCH_COUNT);
    assert_eq!(matrix.cols(), TARGET_WIDTH);
    patch::decode_mask(&matrix.to_host(runtime)).save(path)?;
    Ok(())
}

fn require_file(path: &Path) -> OcrResult<()> {
    if path.is_file() {
        return Ok(());
    }
    Err(IoError::new(
        ErrorKind::NotFound,
        format!("dataset file is missing: {}", path.display()),
    )
    .into())
}

fn validate_dimensions(index: usize, width: u32, height: u32, kind: &str) -> OcrResult<()> {
    if width as usize == patch::IMAGE_SIZE && height as usize == patch::IMAGE_SIZE {
        return Ok(());
    }
    Err(IoError::new(
        ErrorKind::InvalidData,
        format!(
            "sample {index} {kind} is {width}x{height}; expected {}x{}",
            patch::IMAGE_SIZE,
            patch::IMAGE_SIZE
        ),
    )
    .into())
}
