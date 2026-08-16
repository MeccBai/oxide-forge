use crate::cuda::container::Matrix;
use crate::cuda::runtime::CudaRuntime;
use crate::net::linear::Linear;

pub struct Mlp {
    layers: Vec<Linear>,
    res_range: Option<(usize, usize)>,
}

impl Mlp {
    pub fn new(layers: Vec<Linear>, res_range: Option<(usize, usize)>) -> Self {
        assert!(!layers.is_empty(), "MLP must contain at least one layer");
        Self { layers, res_range }
    }

    pub fn forward(&self, input: &Matrix, runtime: &CudaRuntime) -> Matrix {
        let mut output = self.layers[0].forward(input, runtime);
        for layer in &self.layers[1..] {
            output = layer.forward(&output, runtime);
        }
        output
    }

    
}
