use oxide_forge::cuda::{CudaRuntime, InitType};
use oxide_forge::graph::{Graph, GraphBuilder, InitConfig, LearningRateScheduler, MatrixConfig};
use oxide_forge::net::linear::{Activation, LinearConfig};
use oxide_forge::net::mlp::Loss;

const ROWS: usize = 16;
const WIDTH: usize = 16;

fn main() {
    let mut runtime = CudaRuntime::new().unwrap();
    let draft = GraphBuilder::start(MatrixConfig::new(ROWS, WIDTH))
        .then(LinearConfig::new(
            MatrixConfig::new(WIDTH, WIDTH),
            true,
            Activation::Gelu,
        ))
        .then(LinearConfig::new(
            MatrixConfig::new(WIDTH, WIDTH),
            true,
            Activation::Identity,
        ))
        .end();

    let mut graph: Graph<true> = draft
        .init(
            &mut runtime,
            InitConfig::Random {
                loss: Loss::MeanSquaredError,
                learning_rate: LearningRateScheduler::new(1.0e-3).linear(100, 1.0e-4),
            },
        )
        .unwrap();
    let target = runtime.new_matrix(InitType::Zero, ROWS, WIDTH, None);

    for step in 0..2 {
        let input = runtime.new_matrix(InitType::Random, ROWS, WIDTH, None);
        let loss = graph.train_step(input, &target, 1.0, 0.0, 0.9, &mut runtime);
        let mean_loss = loss.sum(&mut runtime, None) / ROWS as f32;
        runtime.recycle_vector(loss);
        println!(
            "step {} loss={mean_loss:.6} next_lr={:.6}",
            step + 1,
            graph.learning_rate().current()
        );
    }

    runtime.recycle_matrix(target);
    runtime.sync();
}
