use crate::cuda::container::{Matrix, Vector};
use crate::cuda::runtime::CudaRuntime;
use crate::net::linear::{Activation, Linear};
use crate::net::mlp::{InferenceMLP, Loss, MlpExecutor, TrainingMlp};
use crate::net::transformer::{InferenceTransformer, NormType, TrainingTransformer};
use cuda_core::DeviceBuffer;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub(super) type CheckpointResult<T> = Result<T, Box<dyn Error>>;

const FORMAT_VERSION: u32 = 1;
const SCALAR_TYPE: &str = "f32_le";
const LINEAR_MODEL_TYPE: &str = "linear";
const MLP_MODEL_TYPE: &str = "mlp";
const TRANSFORMER_MODEL_TYPE: &str = "transformer";

#[derive(Serialize, Deserialize)]
struct LinearFileMetadata {
    format_version: u32,
    model_type: String,
    scalar_type: String,
    data_file: String,
    data_bytes: u64,
    linear: LinearMetadata,
}

#[derive(Serialize, Deserialize)]
struct MlpFileMetadata {
    format_version: u32,
    model_type: String,
    scalar_type: String,
    data_file: String,
    data_bytes: u64,
    loss: Loss,
    mlp: MlpMetadata,
}

#[derive(Serialize, Deserialize)]
struct TransformerFileMetadata {
    format_version: u32,
    model_type: String,
    scalar_type: String,
    data_file: String,
    data_bytes: u64,
    loss: Loss,
    transformer: TransformerMetadata,
}

#[derive(Serialize, Deserialize)]
struct MlpMetadata {
    layer_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    residual: Option<ResidualMetadata>,
    layers: Vec<LinearMetadata>,
}

#[derive(Serialize, Deserialize)]
struct ResidualMetadata {
    start: usize,
    end: usize,
}

#[derive(Serialize, Deserialize)]
struct LinearMetadata {
    input_neurons: usize,
    output_neurons: usize,
    activation: Activation,
    weights: MatrixMetadata,
    #[serde(skip_serializing_if = "Option::is_none")]
    bias: Option<VectorMetadata>,
}

#[derive(Serialize, Deserialize)]
struct MatrixMetadata {
    rows: usize,
    cols: usize,
    byte_start: u64,
    byte_end: u64,
}

#[derive(Serialize, Deserialize)]
struct VectorMetadata {
    len: usize,
    byte_start: u64,
    byte_end: u64,
}

#[derive(Serialize, Deserialize)]
struct TransformerMetadata {
    block_count: usize,
    attention_residual: bool,
    feed_forward_residual: bool,
    #[serde(default)]
    normalization: NormType,
    position: MatrixMetadata,
    query: LinearMetadata,
    key: LinearMetadata,
    value: LinearMetadata,
    feed_forward: MlpMetadata,
    output: LinearMetadata,
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

    fn write_f32(&mut self, values: &[f32]) -> io::Result<(u64, u64)> {
        let start = self.position;
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
        Ok((start, self.position))
    }

    fn finish(mut self) -> io::Result<u64> {
        self.writer.flush()?;
        Ok(self.position)
    }
}

struct ParameterReader {
    file: File,
    file_len: u64,
}

impl ParameterReader {
    fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        Ok(Self { file, file_len })
    }

    fn read_f32(
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

pub(crate) fn dump_mlp_file(
    model: &MlpExecutor,
    path: &Path,
    runtime: &CudaRuntime,
) -> CheckpointResult<()> {
    let metadata_path = normalize_metadata_path(path)?;
    let data_path = metadata_path.with_extension("bin");
    let mut writer = ParameterWriter::create(&data_path)?;
    let mlp = dump_mlp(model, &mut writer, runtime)?;
    let data_bytes = writer.finish()?;
    let metadata = MlpFileMetadata {
        format_version: FORMAT_VERSION,
        model_type: MLP_MODEL_TYPE.to_owned(),
        scalar_type: SCALAR_TYPE.to_owned(),
        data_file: data_file_name(&data_path)?,
        data_bytes,
        loss: model.loss,
        mlp,
    };
    write_metadata(&metadata_path, &metadata)
}

pub(crate) fn dump_linear_file(
    model: &Linear,
    path: &Path,
    runtime: &CudaRuntime,
) -> CheckpointResult<()> {
    let metadata_path = normalize_metadata_path(path)?;
    let data_path = metadata_path.with_extension("bin");
    let mut writer = ParameterWriter::create(&data_path)?;
    let linear = dump_linear(model, &mut writer, runtime)?;
    let data_bytes = writer.finish()?;
    let metadata = LinearFileMetadata {
        format_version: FORMAT_VERSION,
        model_type: LINEAR_MODEL_TYPE.to_owned(),
        scalar_type: SCALAR_TYPE.to_owned(),
        data_file: data_file_name(&data_path)?,
        data_bytes,
        linear,
    };
    write_metadata(&metadata_path, &metadata)
}

pub(crate) fn load_linear_file(path: &Path, runtime: &mut CudaRuntime) -> CheckpointResult<Linear> {
    let metadata_path = normalize_metadata_path(path)?;
    let metadata: LinearFileMetadata = read_metadata(&metadata_path)?;
    validate_header(
        metadata.format_version,
        &metadata.model_type,
        LINEAR_MODEL_TYPE,
        &metadata.scalar_type,
    )?;
    validate_linear(&metadata.linear)?;
    let data_path = resolve_data_path(&metadata_path, &metadata.data_file);
    let mut reader = ParameterReader::open(&data_path)?;
    validate_data_len(metadata.data_bytes, reader.file_len)?;
    load_linear(&metadata.linear, &mut reader, runtime)
}

pub(crate) fn load_mlp_file(
    path: &Path,
    runtime: &mut CudaRuntime,
) -> CheckpointResult<MlpExecutor> {
    let metadata_path = normalize_metadata_path(path)?;
    let metadata: MlpFileMetadata = read_metadata(&metadata_path)?;
    validate_header(
        metadata.format_version,
        &metadata.model_type,
        MLP_MODEL_TYPE,
        &metadata.scalar_type,
    )?;
    validate_mlp(&metadata.mlp)?;
    let data_path = resolve_data_path(&metadata_path, &metadata.data_file);
    let mut reader = ParameterReader::open(&data_path)?;
    validate_data_len(metadata.data_bytes, reader.file_len)?;
    load_mlp(&metadata.mlp, metadata.loss, &mut reader, runtime)
}

pub(crate) fn dump_inference_transformer_file(
    model: &InferenceTransformer,
    path: &Path,
    runtime: &CudaRuntime,
) -> CheckpointResult<()> {
    dump_transformer_file(
        &model.q_matrix,
        &model.k_matrix,
        &model.v_matrix,
        &model.position_matrix,
        &model.fcs.executor,
        &model.output_matrix,
        model.norm_type,
        path,
        runtime,
    )
}

pub(crate) fn dump_training_transformer_file(
    model: &TrainingTransformer,
    path: &Path,
    runtime: &CudaRuntime,
) -> CheckpointResult<()> {
    dump_transformer_file(
        &model.q_matrix,
        &model.k_matrix,
        &model.v_matrix,
        &model.position_matrix,
        &model.fcs.executor,
        &model.output_matrix,
        NormType::Layer,
        path,
        runtime,
    )
}

pub(crate) fn load_inference_transformer_file(
    path: &Path,
    runtime: &mut CudaRuntime,
) -> CheckpointResult<InferenceTransformer> {
    let (metadata, mut reader) = open_transformer_file(path)?;
    let q_matrix = load_linear(&metadata.transformer.query, &mut reader, runtime)?;
    let k_matrix = load_linear(&metadata.transformer.key, &mut reader, runtime)?;
    let v_matrix = load_linear(&metadata.transformer.value, &mut reader, runtime)?;
    let position_matrix = load_matrix(&metadata.transformer.position, &mut reader, runtime)?;
    let executor = load_mlp(
        &metadata.transformer.feed_forward,
        metadata.loss,
        &mut reader,
        runtime,
    )?;
    let output_matrix = load_linear(&metadata.transformer.output, &mut reader, runtime)?;

    Ok(InferenceTransformer::new(
        q_matrix,
        k_matrix,
        v_matrix,
        position_matrix,
        InferenceMLP { executor },
        output_matrix,
        None,
        metadata.transformer.normalization,
    ))
}

pub(crate) fn load_training_transformer_file(
    path: &Path,
    runtime: &mut CudaRuntime,
) -> CheckpointResult<TrainingTransformer> {
    let (metadata, mut reader) = open_transformer_file(path)?;
    if metadata.transformer.normalization != NormType::Layer {
        return Err(invalid_data("TrainingTransformer only supports LayerNorm backward").into());
    }
    let q_matrix = load_linear(&metadata.transformer.query, &mut reader, runtime)?;
    let k_matrix = load_linear(&metadata.transformer.key, &mut reader, runtime)?;
    let v_matrix = load_linear(&metadata.transformer.value, &mut reader, runtime)?;
    let position_matrix = load_matrix(&metadata.transformer.position, &mut reader, runtime)?;
    let executor = load_mlp(
        &metadata.transformer.feed_forward,
        metadata.loss,
        &mut reader,
        runtime,
    )?;
    let output_matrix = load_linear(&metadata.transformer.output, &mut reader, runtime)?;
    Ok(TrainingTransformer::new(
        q_matrix,
        k_matrix,
        v_matrix,
        position_matrix,
        TrainingMlp {
            layer_inputs: Vec::new(),
            executor,
        },
        output_matrix,
    ))
}

#[allow(clippy::too_many_arguments)]
fn dump_transformer_file(
    q_matrix: &Linear,
    k_matrix: &Linear,
    v_matrix: &Linear,
    position_matrix: &Matrix,
    fcs: &MlpExecutor,
    output_matrix: &Linear,
    normalization: NormType,
    path: &Path,
    runtime: &CudaRuntime,
) -> CheckpointResult<()> {
    let metadata_path = normalize_metadata_path(path)?;
    let data_path = metadata_path.with_extension("bin");
    let mut writer = ParameterWriter::create(&data_path)?;

    let transformer = TransformerMetadata {
        block_count: 1,
        attention_residual: true,
        feed_forward_residual: true,
        normalization,
        query: dump_linear(q_matrix, &mut writer, runtime)?,
        key: dump_linear(k_matrix, &mut writer, runtime)?,
        value: dump_linear(v_matrix, &mut writer, runtime)?,
        position: dump_matrix(position_matrix, &mut writer, runtime)?,
        feed_forward: dump_mlp(fcs, &mut writer, runtime)?,
        output: dump_linear(output_matrix, &mut writer, runtime)?,
    };
    let data_bytes = writer.finish()?;
    let metadata = TransformerFileMetadata {
        format_version: FORMAT_VERSION,
        model_type: TRANSFORMER_MODEL_TYPE.to_owned(),
        scalar_type: SCALAR_TYPE.to_owned(),
        data_file: data_file_name(&data_path)?,
        data_bytes,
        loss: fcs.loss,
        transformer,
    };
    write_metadata(&metadata_path, &metadata)
}

fn open_transformer_file(
    path: &Path,
) -> CheckpointResult<(TransformerFileMetadata, ParameterReader)> {
    let metadata_path = normalize_metadata_path(path)?;
    let metadata: TransformerFileMetadata = read_metadata(&metadata_path)?;
    validate_header(
        metadata.format_version,
        &metadata.model_type,
        TRANSFORMER_MODEL_TYPE,
        &metadata.scalar_type,
    )?;
    validate_transformer(&metadata.transformer)?;
    let data_path = resolve_data_path(&metadata_path, &metadata.data_file);
    let reader = ParameterReader::open(&data_path)?;
    validate_data_len(metadata.data_bytes, reader.file_len)?;
    Ok((metadata, reader))
}

fn dump_mlp(
    model: &MlpExecutor,
    writer: &mut ParameterWriter,
    runtime: &CudaRuntime,
) -> CheckpointResult<MlpMetadata> {
    let layers = model
        .layers
        .iter()
        .map(|layer| dump_linear(layer, writer, runtime))
        .collect::<CheckpointResult<Vec<_>>>()?;
    Ok(MlpMetadata {
        layer_count: layers.len(),
        residual: model
            .res_range
            .map(|(start, end)| ResidualMetadata { start, end }),
        layers,
    })
}

fn load_mlp(
    metadata: &MlpMetadata,
    loss: Loss,
    reader: &mut ParameterReader,
    runtime: &mut CudaRuntime,
) -> CheckpointResult<MlpExecutor> {
    let layers = metadata
        .layers
        .iter()
        .map(|layer| load_linear(layer, reader, runtime))
        .collect::<CheckpointResult<Vec<_>>>()?;
    let residual = metadata
        .residual
        .as_ref()
        .map(|residual| (residual.start, residual.end));
    Ok(MlpExecutor::with_loss(layers, residual, loss))
}

fn dump_linear(
    linear: &Linear,
    writer: &mut ParameterWriter,
    runtime: &CudaRuntime,
) -> CheckpointResult<LinearMetadata> {
    Ok(LinearMetadata {
        input_neurons: linear.weights.rows(),
        output_neurons: linear.weights.cols(),
        activation: linear.activation,
        weights: dump_matrix(&linear.weights, writer, runtime)?,
        bias: linear
            .bias
            .as_ref()
            .map(|bias| dump_vector(bias, writer, runtime))
            .transpose()?,
    })
}

fn load_linear(
    metadata: &LinearMetadata,
    reader: &mut ParameterReader,
    runtime: &mut CudaRuntime,
) -> CheckpointResult<Linear> {
    let weights = load_matrix(&metadata.weights, reader, runtime)?;
    let bias = metadata
        .bias
        .as_ref()
        .map(|bias| load_vector(bias, reader, runtime))
        .transpose()?;
    Ok(Linear::new(weights, bias, metadata.activation))
}

fn dump_matrix(
    matrix: &Matrix,
    writer: &mut ParameterWriter,
    runtime: &CudaRuntime,
) -> CheckpointResult<MatrixMetadata> {
    let values = matrix.to_host(runtime);
    let (byte_start, byte_end) = writer.write_f32(&values)?;
    Ok(MatrixMetadata {
        rows: matrix.rows(),
        cols: matrix.cols(),
        byte_start,
        byte_end,
    })
}

fn load_matrix(
    metadata: &MatrixMetadata,
    reader: &mut ParameterReader,
    runtime: &mut CudaRuntime,
) -> CheckpointResult<Matrix> {
    let len = checked_elements(metadata.rows, metadata.cols)?;
    let values = reader.read_f32(metadata.byte_start, metadata.byte_end, len)?;
    let buffer = DeviceBuffer::from_host(runtime.stream(), &values)?;
    Ok(runtime.create_matrix(buffer, metadata.rows, metadata.cols))
}

fn dump_vector(
    vector: &Vector,
    writer: &mut ParameterWriter,
    runtime: &CudaRuntime,
) -> CheckpointResult<VectorMetadata> {
    let values = vector.to_host(runtime);
    let (byte_start, byte_end) = writer.write_f32(&values)?;
    Ok(VectorMetadata {
        len: values.len(),
        byte_start,
        byte_end,
    })
}

fn load_vector(
    metadata: &VectorMetadata,
    reader: &mut ParameterReader,
    runtime: &mut CudaRuntime,
) -> CheckpointResult<Vector> {
    let values = reader.read_f32(metadata.byte_start, metadata.byte_end, metadata.len)?;
    let buffer = DeviceBuffer::from_host(runtime.stream(), &values)?;
    Ok(runtime.create_vector(buffer))
}

fn validate_header(
    version: u32,
    model_type: &str,
    expected_model_type: &str,
    scalar_type: &str,
) -> CheckpointResult<()> {
    if version != FORMAT_VERSION {
        return Err(invalid_data(format!(
            "unsupported checkpoint format version {version}; expected {FORMAT_VERSION}"
        ))
        .into());
    }
    if model_type != expected_model_type {
        return Err(invalid_data(format!(
            "checkpoint contains model type {model_type:?}, expected {expected_model_type:?}"
        ))
        .into());
    }
    if scalar_type != SCALAR_TYPE {
        return Err(invalid_data(format!(
            "unsupported scalar encoding {scalar_type:?}; expected {SCALAR_TYPE:?}"
        ))
        .into());
    }
    Ok(())
}

fn validate_data_len(metadata_len: u64, actual_len: u64) -> CheckpointResult<()> {
    if metadata_len != actual_len {
        return Err(invalid_data(format!(
            "data file has {actual_len} bytes, metadata declares {metadata_len}"
        ))
        .into());
    }
    Ok(())
}

fn validate_mlp(metadata: &MlpMetadata) -> CheckpointResult<()> {
    if metadata.layer_count == 0 || metadata.layer_count != metadata.layers.len() {
        return Err(invalid_data(format!(
            "MLP layer_count is {}, but {} layers are described",
            metadata.layer_count,
            metadata.layers.len()
        ))
        .into());
    }
    for (index, layer) in metadata.layers.iter().enumerate() {
        validate_linear(layer)?;
        if index > 0 && metadata.layers[index - 1].output_neurons != layer.input_neurons {
            return Err(invalid_data(format!(
                "MLP layer {index} expects {} neurons, previous layer produces {}",
                layer.input_neurons,
                metadata.layers[index - 1].output_neurons
            ))
            .into());
        }
    }
    if let Some(residual) = &metadata.residual {
        if residual.start >= residual.end || residual.end > metadata.layers.len() {
            return Err(invalid_data("invalid MLP residual range").into());
        }
        let input_size = metadata.layers[residual.start].input_neurons;
        let output_size = metadata.layers[residual.end - 1].output_neurons;
        if input_size != output_size {
            return Err(invalid_data(format!(
                "residual connects {input_size} input neurons to {output_size} output neurons"
            ))
            .into());
        }
    }
    Ok(())
}

fn validate_linear(metadata: &LinearMetadata) -> CheckpointResult<()> {
    if metadata.input_neurons == 0 || metadata.output_neurons == 0 {
        return Err(invalid_data("Linear dimensions must be non-zero").into());
    }
    if metadata.weights.rows != metadata.input_neurons
        || metadata.weights.cols != metadata.output_neurons
    {
        return Err(invalid_data("Linear neuron counts do not match its weight shape").into());
    }
    checked_elements(metadata.weights.rows, metadata.weights.cols)?;
    if let Some(bias) = &metadata.bias
        && bias.len != metadata.output_neurons
    {
        return Err(invalid_data(format!(
            "Linear bias has {} values, expected {}",
            bias.len, metadata.output_neurons
        ))
        .into());
    }
    Ok(())
}

fn validate_transformer(metadata: &TransformerMetadata) -> CheckpointResult<()> {
    if metadata.block_count != 1 {
        return Err(invalid_data(format!(
            "this runtime supports one Transformer block per file, found {}",
            metadata.block_count
        ))
        .into());
    }
    if !metadata.attention_residual || !metadata.feed_forward_residual {
        return Err(invalid_data(
            "the current Transformer executor requires both residual connections",
        )
        .into());
    }
    checked_elements(metadata.position.rows, metadata.position.cols)?;
    if metadata.position.rows == 0 || metadata.position.cols == 0 {
        return Err(invalid_data("position matrix dimensions must be non-zero").into());
    }
    validate_linear(&metadata.query)?;
    validate_linear(&metadata.key)?;
    validate_linear(&metadata.value)?;
    validate_linear(&metadata.output)?;
    validate_mlp(&metadata.feed_forward)?;

    let hidden = metadata.position.cols;
    if metadata.query.input_neurons != hidden
        || metadata.key.input_neurons != hidden
        || metadata.value.input_neurons != hidden
    {
        return Err(invalid_data("Q, K and V inputs must match the position width").into());
    }
    if metadata.query.output_neurons != metadata.key.output_neurons {
        return Err(invalid_data("Q and K projection widths must match").into());
    }
    if metadata.value.output_neurons != hidden {
        return Err(invalid_data("V output width must match the residual width").into());
    }
    if metadata.feed_forward.layers[0].input_neurons != hidden
        || metadata.feed_forward.layers.last().unwrap().output_neurons != hidden
    {
        return Err(
            invalid_data("feed-forward input and output must match the residual width").into(),
        );
    }
    if metadata.output.input_neurons != hidden {
        return Err(invalid_data("output projection input must match the encoded width").into());
    }
    Ok(())
}

fn checked_elements(rows: usize, cols: usize) -> CheckpointResult<usize> {
    rows.checked_mul(cols)
        .ok_or_else(|| invalid_data("matrix element count overflow").into())
}

fn normalize_metadata_path(path: &Path) -> CheckpointResult<PathBuf> {
    match path.extension() {
        None => Ok(path.with_extension("toml")),
        Some(extension) if extension == "toml" => Ok(path.to_owned()),
        Some(extension) => Err(invalid_data(format!(
            "checkpoint metadata path must use .toml, found .{}",
            extension.to_string_lossy()
        ))
        .into()),
    }
}

fn data_file_name(path: &Path) -> CheckpointResult<String> {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .ok_or_else(|| invalid_data("checkpoint path has no file name").into())
}

fn resolve_data_path(metadata_path: &Path, data_file: &str) -> PathBuf {
    metadata_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(data_file)
}

fn write_metadata<T: Serialize>(path: &Path, metadata: &T) -> CheckpointResult<()> {
    let document = toml::to_string_pretty(metadata)?;
    fs::write(path, document)?;
    Ok(())
}

fn read_metadata<T: DeserializeOwned>(path: &Path) -> CheckpointResult<T> {
    let document = fs::read_to_string(path)?;
    Ok(toml::from_str(&document)?)
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}
