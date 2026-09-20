use oxide_forge::cuda::{CudaRuntime, InitType};
use oxide_forge::graph::{
    Branch, Graph, GraphBuilder, GraphNode, InitConfig, LearningRateScheduler, MatrixConfig,
};
use oxide_forge::net::linear::{Activation, Linear, LinearConfig};
use oxide_forge::net::mlp::Loss;
use oxide_forge::net::mlp::{InferenceMLP, TrainingMlp};
use oxide_forge::net::node::{
    BinaryNode, ConcatNode, CopyNode, ReduceNode, ReduceOp, RowReduceNode, SingleNode, SplitNode,
    TrainingBinaryNode, TrainingConcatNode, TrainingCopyNode, TrainingReduceNode,
    TrainingRowReduceNode, TrainingSingleNode, TrainingSplitNode,
};
use oxide_forge::net::swiglu::{InferenceSwiglu, TrainingSwiglu};
use oxide_forge::net::transformer::{
    InferenceDecoder, InferenceEncoder, TrainingDecoder, TrainingEncoder,
};

fn assert_graph_node<T: GraphNode>() {}

#[test]
fn linear_and_transformers_are_graph_nodes() {
    assert_graph_node::<Graph<false>>();
    assert_graph_node::<Graph<true>>();
    assert_graph_node::<Branch<false>>();
    assert_graph_node::<Branch<true>>();
    assert_graph_node::<Linear>();
    assert_graph_node::<InferenceEncoder<1>>();
    assert_graph_node::<TrainingEncoder<1>>();
    assert_graph_node::<InferenceDecoder<1>>();
    assert_graph_node::<TrainingDecoder<1>>();
    assert_graph_node::<InferenceMLP>();
    assert_graph_node::<TrainingMlp>();
    assert_graph_node::<InferenceSwiglu>();
    assert_graph_node::<TrainingSwiglu>();
    assert_graph_node::<BinaryNode>();
    assert_graph_node::<TrainingBinaryNode>();
    assert_graph_node::<SingleNode>();
    assert_graph_node::<TrainingSingleNode>();
    assert_graph_node::<ConcatNode>();
    assert_graph_node::<TrainingConcatNode>();
    assert_graph_node::<SplitNode>();
    assert_graph_node::<TrainingSplitNode>();
    assert_graph_node::<CopyNode>();
    assert_graph_node::<TrainingCopyNode>();
    assert_graph_node::<ReduceNode>();
    assert_graph_node::<TrainingReduceNode>();
    assert_graph_node::<RowReduceNode>();
    assert_graph_node::<TrainingRowReduceNode>();
}

#[test]
#[ignore = "requires a CUDA device and a CUDA-Oxide device artifact"]
fn graph_runs_two_complete_training_steps() {
    let mut runtime = CudaRuntime::new().unwrap();
    let draft = GraphBuilder::start(MatrixConfig::new(16, 16))
        .then(LinearConfig::new(
            MatrixConfig::new(16, 16),
            true,
            Activation::Identity,
        ))
        .then(LinearConfig::new(
            MatrixConfig::new(16, 16),
            true,
            Activation::Identity,
        ))
        .end();
    let mut graph: Graph<true> = draft
        .init(
            &mut runtime,
            InitConfig::Random {
                loss: Loss::MeanSquaredError,
                learning_rate: LearningRateScheduler::new(0.01),
            },
        )
        .unwrap();
    let target = runtime.new_matrix(InitType::Zero, 16, 16, None);

    for _ in 0..2 {
        let input = runtime.new_matrix(InitType::Random, 16, 16, None);
        let loss = graph.train_step(input, &target, 1.0, 0.0, 0.0, &mut runtime);
        runtime.recycle_vector(loss);
    }

    assert_eq!(graph.learning_rate().current_step(), 2);
    runtime.recycle_matrix(target);
    runtime.sync();
}

#[test]
#[ignore = "requires a CUDA device and a CUDA-Oxide device artifact"]
fn span_set_initializers_preserve_values() {
    let mut runtime = CudaRuntime::new().unwrap();

    let sequence = runtime.new_vector(InitType::Sequence, 5, None);
    assert_eq!(
        sequence.to_host(&runtime, None),
        vec![0.0, 1.0, 2.0, 3.0, 4.0]
    );

    let reverse = runtime.new_vector(InitType::Reverse, 5, None);
    assert_eq!(
        reverse.to_host(&runtime, None),
        vec![5.0, 4.0, 3.0, 2.0, 1.0]
    );

    let zero = runtime.new_vector(InitType::Zero, 5, None);
    assert_eq!(zero.to_host(&runtime, None), vec![0.0; 5]);

    let random = runtime.new_vector(InitType::Random, 64, None);
    assert!(
        random
            .to_host(&runtime, None)
            .into_iter()
            .all(|value| (0.0..=1.0).contains(&value))
    );
}

#[test]
#[ignore = "requires a CUDA device and a CUDA-Oxide device artifact"]
fn copy_and_reduce_are_backward_pairs() {
    let mut runtime = CudaRuntime::new().unwrap();

    let input = runtime.matrix_from_host(&[1.0, 2.0], 1, 2, None).unwrap();
    let mut copy = TrainingCopyNode::new(2);
    let outputs = copy.forward(input, &mut runtime);
    assert_eq!(outputs[0].to_host(&runtime, None), vec![1.0, 2.0]);
    assert_eq!(outputs[1].to_host(&runtime, None), vec![1.0, 2.0]);

    let lhs = runtime.matrix_from_host(&[1.0, 1.0], 1, 2, None).unwrap();
    let rhs = runtime.matrix_from_host(&[2.0, 2.0], 1, 2, None).unwrap();
    let gradient = copy.backward(vec![lhs, rhs], &mut runtime);
    assert_eq!(gradient.to_host(&runtime, None), vec![3.0, 3.0]);

    let lhs = runtime.matrix_from_host(&[2.0, 4.0], 1, 2, None).unwrap();
    let rhs = runtime.matrix_from_host(&[4.0, 8.0], 1, 2, None).unwrap();
    let mut mean = TrainingReduceNode::new(ReduceOp::Mean, 2);
    let output = mean.forward(vec![lhs, rhs], &mut runtime);
    assert_eq!(output.to_host(&runtime, None), vec![3.0, 6.0]);

    let upstream = runtime.matrix_from_host(&[2.0, 4.0], 1, 2, None).unwrap();
    let gradients = mean.backward(upstream, &mut runtime);
    assert_eq!(gradients[0].to_host(&runtime, None), vec![1.0, 2.0]);
    assert_eq!(gradients[1].to_host(&runtime, None), vec![1.0, 2.0]);
}
