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

    pub fn into_values(self) -> Vec<f32> {
        self.values
    }
}

/// Sequential parameter source used while restoring a node or a whole graph.
pub struct HostDataCursor {
    data: std::vec::IntoIter<HostData>,
    consumed: usize,
}

impl HostDataCursor {
    pub fn new(data: Vec<HostData>) -> Self {
        Self {
            data: data.into_iter(),
            consumed: 0,
        }
    }

    pub fn take(&mut self) -> HostData {
        let value = self
            .data
            .next()
            .unwrap_or_else(|| panic!("missing host parameter at index {}", self.consumed));
        self.consumed += 1;
        value
    }

    pub fn consumed(&self) -> usize {
        self.consumed
    }

    pub fn remaining(&self) -> usize {
        self.data.len()
    }

    pub fn finish(self) {
        assert_eq!(self.data.len(), 0, "unused host parameters remain");
    }
}

#[cfg(test)]
mod tests {
    use super::{HostData, HostDataCursor};

    #[test]
    fn host_data_cursor_preserves_parameter_order() {
        let mut cursor = HostDataCursor::new(vec![
            HostData::new(vec![1.0, 2.0]),
            HostData::new(vec![3.0]),
        ]);
        assert_eq!(cursor.take().values(), &[1.0, 2.0]);
        assert_eq!(cursor.consumed(), 1);
        assert_eq!(cursor.take().values(), &[3.0]);
        cursor.finish();
    }
}
