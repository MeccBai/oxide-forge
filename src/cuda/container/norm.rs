use cuda_core::LaunchConfig1D;

use crate::cuda::{DEFAULT_BLOCK_SIZE, DeviceSpan, DeviceSpanMut, runtime::CudaRuntime};

use super::Matrix;

impl Matrix {
    pub fn softmax_rows(&mut self, runtime: &CudaRuntime) {
        if self.rows == 0 {
            return;
        }
        assert!(self.cols > 0 && self.cols <= DEFAULT_BLOCK_SIZE);
        let config = LaunchConfig1D::new(self.rows as u32, DEFAULT_BLOCK_SIZE as u32, 0);
        let prepared = runtime
            .module()
            .prepare_matrix_softmax_rows(config)
            .unwrap();
        let len = self.buffer.len();
        let matrix = DeviceSpanMut::from_buffer(&mut self.buffer, 0, len);
        runtime
            .module()
            .matrix_softmax_rows(runtime.stream(), &prepared, matrix.descriptor(), self.cols)
            .unwrap();
    }

    pub fn layer_norm(&mut self, runtime: &CudaRuntime) {
        if self.rows == 0 {
            return;
        }
        assert!(self.cols > 0 && self.cols <= DEFAULT_BLOCK_SIZE);
        let config = LaunchConfig1D::new(self.rows as u32, DEFAULT_BLOCK_SIZE as u32, 0);
        let prepared = runtime
            .module()
            .prepare_matrix_layer_norm_rows(config)
            .unwrap();
        let len = self.buffer.len();
        let matrix = DeviceSpanMut::from_buffer(&mut self.buffer, 0, len);
        runtime
            .module()
            .matrix_layer_norm_rows(
                runtime.stream(),
                &prepared,
                matrix.descriptor(),
                self.cols,
                1e-5,
            )
            .unwrap();
    }

    pub fn rms_norm(&mut self, runtime: &CudaRuntime) {
        if self.rows == 0 {
            return;
        }
        assert!(self.cols > 0 && self.cols <= DEFAULT_BLOCK_SIZE);
        let config = LaunchConfig1D::new(self.rows as u32, DEFAULT_BLOCK_SIZE as u32, 0);
        let prepared = runtime.module().prepare_rms_norm_assign(config).unwrap();
        let len = self.buffer.len();
        let matrix = DeviceSpanMut::from_buffer(&mut self.buffer, 0, len);
        runtime
            .module()
            .rms_norm_assign(
                runtime.stream(),
                &prepared,
                matrix.descriptor(),
                self.cols,
                1e-5,
            )
            .unwrap();
    }
}

impl CudaRuntime {
    pub fn softmax_rows_backward(
        &mut self,
        probabilities: &Matrix,
        output_gradient: &Matrix,
    ) -> Matrix {
        assert_eq!(probabilities.rows, output_gradient.rows);
        assert_eq!(probabilities.cols, output_gradient.cols);
        assert!(probabilities.cols <= DEFAULT_BLOCK_SIZE);
        let mut buffer = self.get_uninit_buffer(probabilities.rows * probabilities.cols);
        let config = LaunchConfig1D::new(probabilities.rows as u32, DEFAULT_BLOCK_SIZE as u32, 0);
        let prepared = self.module().prepare_softmax_rows_backward(config).unwrap();
        let probabilities_span =
            DeviceSpan::from_buffer(&probabilities.buffer, 0, probabilities.buffer.len());
        let output_gradient_span =
            DeviceSpan::from_buffer(&output_gradient.buffer, 0, output_gradient.buffer.len());
        let len = buffer.len();
        let result = DeviceSpanMut::from_buffer(&mut buffer, 0, len);
        self.module()
            .softmax_rows_backward(
                self.stream(),
                &prepared,
                probabilities_span.descriptor(),
                output_gradient_span.descriptor(),
                result.descriptor(),
                probabilities.cols,
            )
            .unwrap();
        self.create_matrix(buffer, probabilities.rows, probabilities.cols)
    }

    pub fn layer_norm_backward(&mut self, input: &Matrix, output_gradient: &Matrix) -> Matrix {
        assert_eq!(input.rows, output_gradient.rows);
        assert_eq!(input.cols, output_gradient.cols);
        assert!(input.cols <= DEFAULT_BLOCK_SIZE);
        let mut buffer = self.get_uninit_buffer(input.rows * input.cols);
        let config = LaunchConfig1D::new(input.rows as u32, DEFAULT_BLOCK_SIZE as u32, 0);
        let prepared = self.module().prepare_layer_norm_backward(config).unwrap();
        let input_span = DeviceSpan::from_buffer(&input.buffer, 0, input.buffer.len());
        let output_gradient_span =
            DeviceSpan::from_buffer(&output_gradient.buffer, 0, output_gradient.buffer.len());
        let len = buffer.len();
        let result = DeviceSpanMut::from_buffer(&mut buffer, 0, len);
        self.module()
            .layer_norm_backward(
                self.stream(),
                &prepared,
                input_span.descriptor(),
                output_gradient_span.descriptor(),
                result.descriptor(),
                input.cols,
                1e-5,
            )
            .unwrap();
        self.create_matrix(buffer, input.rows, input.cols)
    }

    pub fn rms_norm_backward(&mut self, input: &Matrix, output_gradient: &Matrix) -> Matrix {
        assert_eq!(input.rows, output_gradient.rows);
        assert_eq!(input.cols, output_gradient.cols);
        assert!(input.cols <= DEFAULT_BLOCK_SIZE);
        let mut buffer = self.get_uninit_buffer(input.rows * input.cols);
        let config = LaunchConfig1D::new(input.rows as u32, DEFAULT_BLOCK_SIZE as u32, 0);
        let prepared = self
            .module()
            .prepare_matrix_rms_norm_backward(config)
            .unwrap();
        let input_span = DeviceSpan::from_buffer(&input.buffer, 0, input.buffer.len());
        let output_gradient_span =
            DeviceSpan::from_buffer(&output_gradient.buffer, 0, output_gradient.buffer.len());
        let len = buffer.len();

        let result = DeviceSpanMut::from_buffer(&mut buffer, 0, len);
        self.module()
            .matrix_rms_norm_backward(
                self.stream(),
                &prepared,
                input_span.descriptor(),
                output_gradient_span.descriptor(),
                result.descriptor(),
                input.cols,
                1e-5,
            )
            .unwrap();

        self.create_matrix(buffer, input.rows, input.cols)
    }
}
