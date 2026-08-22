use serde::{Deserialize, Serialize};

const SCALAR_BYTES: u64 = size_of::<f32>() as u64;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatrixMetadata {
    pub rows: usize,
    pub cols: usize,
    pub byte_start: u64,
    pub byte_end: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorMetadata {
    pub len: usize,
    pub byte_start: u64,
    pub byte_end: u64,
}

#[derive(Default)]
pub struct MetadataCursor {
    position: u64,
}

impl MetadataCursor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn matrix(&mut self, rows: usize, cols: usize) -> MatrixMetadata {
        let elements = rows
            .checked_mul(cols)
            .expect("matrix element count overflow");
        let (byte_start, byte_end) = self.reserve(elements);
        MatrixMetadata {
            rows,
            cols,
            byte_start,
            byte_end,
        }
    }

    pub fn vector(&mut self, len: usize) -> VectorMetadata {
        let (byte_start, byte_end) = self.reserve(len);
        VectorMetadata {
            len,
            byte_start,
            byte_end,
        }
    }

    pub fn data_bytes(&self) -> u64 {
        self.position
    }

    fn reserve(&mut self, elements: usize) -> (u64, u64) {
        let byte_len = u64::try_from(elements)
            .ok()
            .and_then(|elements| elements.checked_mul(SCALAR_BYTES))
            .expect("parameter byte length overflow");
        let byte_start = self.position;
        self.position = self
            .position
            .checked_add(byte_len)
            .expect("checkpoint offset overflow");
        (byte_start, self.position)
    }
}

pub struct HostData {
    values: Vec<f32>,
}

impl HostData {
    pub fn new(values: Vec<f32>) -> Self {
        Self { values }
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }
}
