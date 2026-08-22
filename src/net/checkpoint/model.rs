use super::CheckpointResult;
use super::io::{
    ParameterReader, checkpoint_paths, data_file_name, invalid_data, normalize_metadata_path,
    read_metadata, resolve_data_path, write_metadata, write_parameters,
};
use crate::cuda::container::{Matrix, Vector};
use crate::cuda::runtime::CudaRuntime;
use crate::net::linear::{Linear, LinearMetadata};
use crate::net::metadata::{HostData, MatrixMetadata, MetadataCursor, VectorMetadata};
use crate::net::mlp::{InferenceMLP, MlpExecutor, MlpMetadata, TrainingMlp};
use crate::net::transformer::encoder::{
    InferenceTransformer, TrainingTransformer, TransformerMetadata,
};
use serde::{Deserialize, Serialize};
use std::path::Path;

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
    mlp: MlpMetadata,
}

#[derive(Serialize, Deserialize)]
struct TransformerFileMetadata {
    format_version: u32,
    model_type: String,
    scalar_type: String,
    data_file: String,
    data_bytes: u64,
    transformer: TransformerMetadata,
}

pub fn dump_linear<P: AsRef<Path>>(
    model: &Linear,
    path: P,
    runtime: &CudaRuntime,
) -> CheckpointResult<()> {
    let mut cursor = MetadataCursor::new();
    let linear = model.get_meta_data(&mut cursor);
    let paths = checkpoint_paths(path.as_ref())?;
    let metadata = LinearFileMetadata {
        format_version: FORMAT_VERSION,
        model_type: LINEAR_MODEL_TYPE.to_owned(),
        scalar_type: SCALAR_TYPE.to_owned(),
        data_file: data_file_name(&paths.data)?,
        data_bytes: cursor.data_bytes(),
        linear,
    };
    write_metadata(&paths.metadata, &metadata)?;
    write_parameters(&paths.data, &model.get_data(runtime), metadata.data_bytes)
}

pub fn load_linear<P: AsRef<Path>>(path: P, runtime: &CudaRuntime) -> CheckpointResult<Linear> {
    let (metadata, mut reader) = open_linear(path.as_ref())?;
    load_linear_parameter(&metadata.linear, &mut reader, runtime)
}

pub fn dump_mlp<P: AsRef<Path>>(
    model: &MlpExecutor,
    path: P,
    runtime: &CudaRuntime,
) -> CheckpointResult<()> {
    dump_mlp_data(
        model.get_meta_data(&mut MetadataCursor::new()),
        path,
        || model.get_data(runtime),
    )
}

pub fn load_mlp<P: AsRef<Path>>(path: P, runtime: &CudaRuntime) -> CheckpointResult<MlpExecutor> {
    let (metadata, mut reader) = open_mlp(path.as_ref())?;
    let (layers, residual) = load_mlp_parameters(&metadata.mlp, &mut reader, runtime)?;
    Ok(MlpExecutor::with_loss(layers, residual, metadata.mlp.loss))
}

pub fn dump_inference_mlp<P: AsRef<Path>>(
    model: &InferenceMLP,
    path: P,
    runtime: &CudaRuntime,
) -> CheckpointResult<()> {
    dump_mlp_data(
        model.get_meta_data(&mut MetadataCursor::new()),
        path,
        || model.get_data(runtime),
    )
}

pub fn load_inference_mlp<P: AsRef<Path>>(
    path: P,
    runtime: &CudaRuntime,
) -> CheckpointResult<InferenceMLP> {
    let (metadata, mut reader) = open_mlp(path.as_ref())?;
    let (layers, residual) = load_mlp_parameters(&metadata.mlp, &mut reader, runtime)?;
    Ok(InferenceMLP::with_loss(layers, residual, metadata.mlp.loss))
}

pub fn dump_training_mlp<P: AsRef<Path>>(
    model: &TrainingMlp,
    path: P,
    runtime: &CudaRuntime,
) -> CheckpointResult<()> {
    dump_mlp_data(
        model.get_meta_data(&mut MetadataCursor::new()),
        path,
        || model.get_data(runtime),
    )
}

pub fn load_training_mlp<P: AsRef<Path>>(
    path: P,
    runtime: &CudaRuntime,
) -> CheckpointResult<TrainingMlp> {
    let (metadata, mut reader) = open_mlp(path.as_ref())?;
    let (layers, residual) = load_mlp_parameters(&metadata.mlp, &mut reader, runtime)?;
    Ok(TrainingMlp::with_loss(layers, residual, metadata.mlp.loss))
}

pub fn dump_transformer<P: AsRef<Path>>(
    model: &InferenceTransformer,
    path: P,
    runtime: &CudaRuntime,
) -> CheckpointResult<()> {
    dump_transformer_data(
        model.get_meta_data(&mut MetadataCursor::new()),
        path,
        || model.get_data(runtime),
    )
}

pub fn load_transformer<P: AsRef<Path>>(
    path: P,
    runtime: &CudaRuntime,
) -> CheckpointResult<InferenceTransformer> {
    let (metadata, mut reader) = open_transformer(path.as_ref())?;
    let transformer = &metadata.transformer;
    let q_matrix = load_linear_parameter(&transformer.query, &mut reader, runtime)?;
    let k_matrix = load_linear_parameter(&transformer.key, &mut reader, runtime)?;
    let v_matrix = load_linear_parameter(&transformer.value, &mut reader, runtime)?;
    let position_matrix = load_matrix(&transformer.position, &mut reader, runtime)?;
    let (layers, residual) = load_mlp_parameters(&transformer.feed_forward, &mut reader, runtime)?;
    let output_matrix = load_linear_parameter(&transformer.output, &mut reader, runtime)?;
    Ok(InferenceTransformer::new(
        q_matrix,
        k_matrix,
        v_matrix,
        position_matrix,
        InferenceMLP::with_loss(layers, residual, transformer.feed_forward.loss),
        output_matrix,
        None,
        transformer.normalization,
    ))
}

pub fn dump_training_transformer<P: AsRef<Path>>(
    model: &TrainingTransformer,
    path: P,
    runtime: &CudaRuntime,
) -> CheckpointResult<()> {
    dump_transformer_data(
        model.get_meta_data(&mut MetadataCursor::new()),
        path,
        || model.get_data(runtime),
    )
}

pub fn load_training_transformer<P: AsRef<Path>>(
    path: P,
    runtime: &CudaRuntime,
) -> CheckpointResult<TrainingTransformer> {
    let (metadata, mut reader) = open_transformer(path.as_ref())?;
    let transformer = &metadata.transformer;
    let q_matrix = load_linear_parameter(&transformer.query, &mut reader, runtime)?;
    let k_matrix = load_linear_parameter(&transformer.key, &mut reader, runtime)?;
    let v_matrix = load_linear_parameter(&transformer.value, &mut reader, runtime)?;
    let position_matrix = load_matrix(&transformer.position, &mut reader, runtime)?;
    let (layers, residual) = load_mlp_parameters(&transformer.feed_forward, &mut reader, runtime)?;
    let output_matrix = load_linear_parameter(&transformer.output, &mut reader, runtime)?;
    Ok(TrainingTransformer::new(
        q_matrix,
        k_matrix,
        v_matrix,
        position_matrix,
        TrainingMlp::with_loss(layers, residual, transformer.feed_forward.loss),
        output_matrix,
        transformer.normalization,
    ))
}

fn dump_mlp_data<P: AsRef<Path>, F: FnOnce() -> Vec<HostData>>(
    mlp: MlpMetadata,
    path: P,
    get_parameters: F,
) -> CheckpointResult<()> {
    let paths = checkpoint_paths(path.as_ref())?;
    let data_bytes = metadata_end_for_mlp(&mlp)?;
    let metadata = MlpFileMetadata {
        format_version: FORMAT_VERSION,
        model_type: MLP_MODEL_TYPE.to_owned(),
        scalar_type: SCALAR_TYPE.to_owned(),
        data_file: data_file_name(&paths.data)?,
        data_bytes,
        mlp,
    };
    write_metadata(&paths.metadata, &metadata)?;
    let parameters = get_parameters();
    write_parameters(&paths.data, &parameters, metadata.data_bytes)
}

fn dump_transformer_data<P: AsRef<Path>, F: FnOnce() -> Vec<HostData>>(
    transformer: TransformerMetadata,
    path: P,
    get_parameters: F,
) -> CheckpointResult<()> {
    let paths = checkpoint_paths(path.as_ref())?;
    let data_bytes = metadata_end_for_transformer(&transformer)?;
    let metadata = TransformerFileMetadata {
        format_version: FORMAT_VERSION,
        model_type: TRANSFORMER_MODEL_TYPE.to_owned(),
        scalar_type: SCALAR_TYPE.to_owned(),
        data_file: data_file_name(&paths.data)?,
        data_bytes,
        transformer,
    };
    write_metadata(&paths.metadata, &metadata)?;
    let parameters = get_parameters();
    write_parameters(&paths.data, &parameters, metadata.data_bytes)
}

fn open_linear(path: &Path) -> CheckpointResult<(LinearFileMetadata, ParameterReader)> {
    let metadata_path = normalize_metadata_path(path)?;
    let metadata: LinearFileMetadata = read_metadata(&metadata_path)?;
    validate_header(
        &metadata.model_type,
        LINEAR_MODEL_TYPE,
        &metadata.scalar_type,
        metadata.format_version,
    )?;
    let mut validator = RangeValidator::default();
    validate_linear(&metadata.linear, &mut validator)?;
    validator.finish(metadata.data_bytes)?;
    let data_path = resolve_data_path(&metadata_path, &metadata.data_file)?;
    let reader = ParameterReader::open(&data_path, metadata.data_bytes)?;
    Ok((metadata, reader))
}

fn open_mlp(path: &Path) -> CheckpointResult<(MlpFileMetadata, ParameterReader)> {
    let metadata_path = normalize_metadata_path(path)?;
    let metadata: MlpFileMetadata = read_metadata(&metadata_path)?;
    validate_header(
        &metadata.model_type,
        MLP_MODEL_TYPE,
        &metadata.scalar_type,
        metadata.format_version,
    )?;
    let mut validator = RangeValidator::default();
    validate_mlp(&metadata.mlp, &mut validator)?;
    validator.finish(metadata.data_bytes)?;
    let data_path = resolve_data_path(&metadata_path, &metadata.data_file)?;
    let reader = ParameterReader::open(&data_path, metadata.data_bytes)?;
    Ok((metadata, reader))
}

fn open_transformer(path: &Path) -> CheckpointResult<(TransformerFileMetadata, ParameterReader)> {
    let metadata_path = normalize_metadata_path(path)?;
    let metadata: TransformerFileMetadata = read_metadata(&metadata_path)?;
    validate_header(
        &metadata.model_type,
        TRANSFORMER_MODEL_TYPE,
        &metadata.scalar_type,
        metadata.format_version,
    )?;
    let mut validator = RangeValidator::default();
    validate_transformer(&metadata.transformer, &mut validator)?;
    validator.finish(metadata.data_bytes)?;
    let data_path = resolve_data_path(&metadata_path, &metadata.data_file)?;
    let reader = ParameterReader::open(&data_path, metadata.data_bytes)?;
    Ok((metadata, reader))
}

fn load_mlp_parameters(
    metadata: &MlpMetadata,
    reader: &mut ParameterReader,
    runtime: &CudaRuntime,
) -> CheckpointResult<(Vec<Linear>, Option<(usize, usize)>)> {
    let layers = metadata
        .layers
        .iter()
        .map(|layer| load_linear_parameter(layer, reader, runtime))
        .collect::<CheckpointResult<Vec<_>>>()?;
    let residual = metadata
        .residual
        .as_ref()
        .map(|residual| (residual.start, residual.end));
    Ok((layers, residual))
}

fn load_linear_parameter(
    metadata: &LinearMetadata,
    reader: &mut ParameterReader,
    runtime: &CudaRuntime,
) -> CheckpointResult<Linear> {
    let weights = load_matrix(&metadata.weights, reader, runtime)?;
    let bias = metadata
        .bias
        .as_ref()
        .map(|bias| load_vector(bias, reader, runtime))
        .transpose()?;
    Ok(Linear::new(weights, bias, metadata.activation))
}

fn load_matrix(
    metadata: &MatrixMetadata,
    reader: &mut ParameterReader,
    runtime: &CudaRuntime,
) -> CheckpointResult<Matrix> {
    let len = checked_elements(metadata.rows, metadata.cols)?;
    let values = reader.read_f32(metadata.byte_start, metadata.byte_end, len)?;
    Ok(runtime.matrix_from_host(&values, metadata.rows, metadata.cols)?)
}

fn load_vector(
    metadata: &VectorMetadata,
    reader: &mut ParameterReader,
    runtime: &CudaRuntime,
) -> CheckpointResult<Vector> {
    let values = reader.read_f32(metadata.byte_start, metadata.byte_end, metadata.len)?;
    Ok(runtime.vector_from_host(&values)?)
}

fn validate_header(
    model_type: &str,
    expected_model_type: &str,
    scalar_type: &str,
    format_version: u32,
) -> CheckpointResult<()> {
    if format_version != FORMAT_VERSION {
        return Err(invalid_data(format!(
            "unsupported checkpoint format version {format_version}; expected {FORMAT_VERSION}"
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

fn metadata_end_for_mlp(metadata: &MlpMetadata) -> CheckpointResult<u64> {
    let mut validator = RangeValidator::default();
    validate_mlp(metadata, &mut validator)?;
    Ok(validator.position)
}

fn metadata_end_for_transformer(metadata: &TransformerMetadata) -> CheckpointResult<u64> {
    let mut validator = RangeValidator::default();
    validate_transformer(metadata, &mut validator)?;
    Ok(validator.position)
}

fn validate_mlp(metadata: &MlpMetadata, ranges: &mut RangeValidator) -> CheckpointResult<()> {
    if metadata.layer_count == 0 || metadata.layer_count != metadata.layers.len() {
        return Err(invalid_data(format!(
            "MLP layer_count is {}, but {} layers are described",
            metadata.layer_count,
            metadata.layers.len()
        ))
        .into());
    }
    for (index, layer) in metadata.layers.iter().enumerate() {
        validate_linear(layer, ranges)?;
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

fn validate_linear(metadata: &LinearMetadata, ranges: &mut RangeValidator) -> CheckpointResult<()> {
    if metadata.input_neurons == 0 || metadata.output_neurons == 0 {
        return Err(invalid_data("Linear dimensions must be non-zero").into());
    }
    if metadata.weights.rows != metadata.input_neurons
        || metadata.weights.cols != metadata.output_neurons
    {
        return Err(invalid_data("Linear weight shape does not match its neuron counts").into());
    }
    ranges.matrix(&metadata.weights)?;
    if let Some(bias) = &metadata.bias {
        if bias.len != metadata.output_neurons {
            return Err(invalid_data("Linear bias length does not match output neurons").into());
        }
        ranges.vector(bias)?;
    }
    Ok(())
}

fn validate_transformer(
    metadata: &TransformerMetadata,
    ranges: &mut RangeValidator,
) -> CheckpointResult<()> {
    if metadata.block_count != 1 {
        return Err(invalid_data("this Transformer executor requires exactly one block").into());
    }
    if !metadata.attention_residual || !metadata.feed_forward_residual {
        return Err(invalid_data("this Transformer executor requires both residual paths").into());
    }
    validate_linear(&metadata.query, ranges)?;
    validate_linear(&metadata.key, ranges)?;
    validate_linear(&metadata.value, ranges)?;
    ranges.matrix(&metadata.position)?;
    validate_mlp(&metadata.feed_forward, ranges)?;
    validate_linear(&metadata.output, ranges)?;

    let embedding = metadata.position.cols;
    if metadata.position.rows == 0 || embedding == 0 {
        return Err(invalid_data("position matrix dimensions must be non-zero").into());
    }
    if metadata.query.input_neurons != embedding
        || metadata.key.input_neurons != embedding
        || metadata.value.input_neurons != embedding
    {
        return Err(invalid_data("Q/K/V input dimensions must match the embedding size").into());
    }
    if metadata.query.output_neurons != metadata.key.output_neurons {
        return Err(invalid_data("query and key dimensions must match").into());
    }
    if metadata.value.output_neurons != embedding {
        return Err(
            invalid_data("value output dimension must match the residual embedding").into(),
        );
    }
    if metadata.feed_forward.layers[0].input_neurons != embedding
        || metadata.feed_forward.layers.last().unwrap().output_neurons != embedding
    {
        return Err(invalid_data("feed-forward input/output must match the embedding size").into());
    }
    if metadata.output.input_neurons != embedding {
        return Err(invalid_data("output projection input must match the embedding size").into());
    }
    Ok(())
}

#[derive(Default)]
struct RangeValidator {
    position: u64,
}

impl RangeValidator {
    fn matrix(&mut self, metadata: &MatrixMetadata) -> CheckpointResult<()> {
        self.range(
            metadata.byte_start,
            metadata.byte_end,
            checked_elements(metadata.rows, metadata.cols)?,
        )
    }

    fn vector(&mut self, metadata: &VectorMetadata) -> CheckpointResult<()> {
        self.range(metadata.byte_start, metadata.byte_end, metadata.len)
    }

    fn range(&mut self, start: u64, end: u64, elements: usize) -> CheckpointResult<()> {
        let bytes = u64::try_from(elements)
            .ok()
            .and_then(|elements| elements.checked_mul(size_of::<f32>() as u64))
            .ok_or_else(|| invalid_data("parameter byte length overflow"))?;
        let expected_end = self
            .position
            .checked_add(bytes)
            .ok_or_else(|| invalid_data("checkpoint offset overflow"))?;
        if start != self.position || end != expected_end {
            return Err(invalid_data(format!(
                "parameter range [{start}, {end}) is not the expected contiguous range [{}, {expected_end})",
                self.position
            ))
            .into());
        }
        self.position = expected_end;
        Ok(())
    }

    fn finish(&self, data_bytes: u64) -> CheckpointResult<()> {
        if self.position != data_bytes {
            return Err(invalid_data(format!(
                "parameter ranges end at {}, metadata declares {data_bytes} bytes",
                self.position
            ))
            .into());
        }
        Ok(())
    }
}

fn checked_elements(rows: usize, cols: usize) -> CheckpointResult<usize> {
    rows.checked_mul(cols)
        .ok_or_else(|| invalid_data("matrix element count overflow").into())
}
