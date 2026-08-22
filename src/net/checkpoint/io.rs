use super::CheckpointResult;
use crate::net::metadata::HostData;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::fs::File;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub(super) struct CheckpointPaths {
    pub metadata: PathBuf,
    pub data: PathBuf,
}

pub(super) fn checkpoint_paths(path: &Path) -> CheckpointResult<CheckpointPaths> {
    let metadata = normalize_metadata_path(path)?;
    let data = metadata.with_extension("bin");
    Ok(CheckpointPaths { metadata, data })
}

pub(super) fn normalize_metadata_path(path: &Path) -> CheckpointResult<PathBuf> {
    match path.extension().and_then(|extension| extension.to_str()) {
        None => Ok(path.with_extension("toml")),
        Some("toml") => Ok(path.to_owned()),
        Some(extension) => Err(invalid_data(format!(
            "checkpoint metadata must use .toml, not .{extension}"
        ))
        .into()),
    }
}

pub(super) fn data_file_name(path: &Path) -> CheckpointResult<String> {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(str::to_owned)
        .ok_or_else(|| invalid_data("checkpoint data path has no UTF-8 file name").into())
}

pub(super) fn resolve_data_path(
    metadata_path: &Path,
    data_file: &str,
) -> CheckpointResult<PathBuf> {
    let data_file = Path::new(data_file);
    if data_file.file_name().is_none() || data_file.components().count() != 1 {
        return Err(invalid_data("checkpoint data_file must be a plain file name").into());
    }
    Ok(metadata_path
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .join(data_file))
}

pub(super) fn write_metadata<T: Serialize>(path: &Path, metadata: &T) -> CheckpointResult<()> {
    let text = toml::to_string_pretty(metadata)?;
    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(text.as_bytes())?;
    writer.flush()?;
    Ok(())
}

pub(super) fn read_metadata<T: DeserializeOwned>(path: &Path) -> CheckpointResult<T> {
    let mut text = String::new();
    File::open(path)?.read_to_string(&mut text)?;
    Ok(toml::from_str(&text)?)
}

pub(super) fn write_parameters(
    path: &Path,
    parameters: &[HostData],
    expected_bytes: u64,
) -> CheckpointResult<()> {
    let mut writer = ParameterWriter::create(path)?;
    for parameter in parameters {
        writer.write_f32(parameter.values())?;
    }
    let actual_bytes = writer.finish()?;
    if actual_bytes != expected_bytes {
        return Err(invalid_data(format!(
            "model exported {actual_bytes} parameter bytes, metadata declares {expected_bytes}"
        ))
        .into());
    }
    Ok(())
}

struct ParameterWriter {
    writer: BufWriter<File>,
    position: u64,
}

impl ParameterWriter {
    fn create(path: &Path) -> io::Result<Self> {
        Ok(Self {
            writer: BufWriter::new(File::create(path)?),
            position: 0,
        })
    }

    fn write_f32(&mut self, values: &[f32]) -> io::Result<()> {
        for chunk in values.chunks(16 * 1024) {
            let mut bytes = Vec::with_capacity(chunk.len() * size_of::<f32>());
            for value in chunk {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            self.writer.write_all(&bytes)?;
        }
        let byte_len = values
            .len()
            .checked_mul(size_of::<f32>())
            .and_then(|len| u64::try_from(len).ok())
            .ok_or_else(|| invalid_data("parameter byte length overflow"))?;
        self.position = self
            .position
            .checked_add(byte_len)
            .ok_or_else(|| invalid_data("checkpoint offset overflow"))?;
        Ok(())
    }

    fn finish(mut self) -> io::Result<u64> {
        self.writer.flush()?;
        Ok(self.position)
    }
}

pub(super) struct ParameterReader {
    file: File,
    file_len: u64,
}

impl ParameterReader {
    pub(super) fn open(path: &Path, expected_bytes: u64) -> CheckpointResult<Self> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        if file_len != expected_bytes {
            return Err(invalid_data(format!(
                "data file has {file_len} bytes, metadata declares {expected_bytes}"
            ))
            .into());
        }
        Ok(Self { file, file_len })
    }

    pub(super) fn read_f32(
        &mut self,
        byte_start: u64,
        byte_end: u64,
        expected_values: usize,
    ) -> CheckpointResult<Vec<f32>> {
        if byte_start > byte_end || byte_end > self.file_len {
            return Err(invalid_data(format!(
                "parameter range [{byte_start}, {byte_end}) is outside a {} byte data file",
                self.file_len
            ))
            .into());
        }
        let expected_bytes = expected_values
            .checked_mul(size_of::<f32>())
            .and_then(|len| u64::try_from(len).ok())
            .ok_or_else(|| invalid_data("parameter byte length overflow"))?;
        if byte_end - byte_start != expected_bytes {
            return Err(invalid_data(format!(
                "parameter range has {} bytes, expected {expected_bytes}",
                byte_end - byte_start
            ))
            .into());
        }

        let byte_len = usize::try_from(expected_bytes)
            .map_err(|_| invalid_data("parameter is too large for this host"))?;
        let mut bytes = vec![0_u8; byte_len];
        self.file.seek(SeekFrom::Start(byte_start))?;
        self.file.read_exact(&mut bytes)?;
        Ok(bytes
            .chunks_exact(size_of::<f32>())
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect())
    }
}

pub(super) fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}
