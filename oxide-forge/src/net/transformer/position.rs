use crate::cuda::{container::Matrix, runtime::CudaRuntime};
use crate::net::metadata::{HostData, HostDataCursor, MatrixMetadata, MetadataCursor};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum PositionEncodingMetadata {
    Identity,
    Additive { values: MatrixMetadata },
}

/// Positional encoding owned by a Transformer.
///
/// Both variants have an identity input derivative, so no training cache or
/// dedicated graph node is required.
pub enum PositionEncoding {
    Identity,
    Additive(Matrix),
}

impl PositionEncoding {
    pub fn identity() -> Self {
        Self::Identity
    }

    pub fn additive(values: Matrix) -> Self {
        Self::Additive(values)
    }

    pub fn get_meta_data(&self, cursor: &mut MetadataCursor) -> PositionEncodingMetadata {
        match self {
            Self::Identity => PositionEncodingMetadata::Identity,
            Self::Additive(values) => PositionEncodingMetadata::Additive {
                values: cursor.matrix(values.rows(), values.cols()),
            },
        }
    }

    pub fn get_data(&self, runtime: &CudaRuntime) -> Vec<HostData> {
        match self {
            Self::Identity => Vec::new(),
            Self::Additive(values) => vec![HostData::new(values.to_host(runtime, None))],
        }
    }

    pub fn set_data(&mut self, data: &mut HostDataCursor, runtime: &CudaRuntime) {
        if let Self::Additive(values) = self {
            let host = data.take();
            values.copy_from_host(host.values(), runtime, None).unwrap();
        }
    }

    pub fn forward(&self, input: &Matrix, runtime: &mut CudaRuntime) -> Matrix {
        match self {
            Self::Identity => runtime.clone_matrix(input, None),
            Self::Additive(values) => {
                assert_eq!(input.shape(), values.shape());
                runtime.matrix_add(input, values, None)
            }
        }
    }
}

impl Default for PositionEncoding {
    fn default() -> Self {
        Self::Identity
    }
}
