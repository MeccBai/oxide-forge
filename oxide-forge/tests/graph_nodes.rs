use oxide_forge::cuda::{CudaRuntime, InitType};
use oxide_forge::graph::{GraphNode, LearnConfig};
use oxide_forge::net::linear::{Activation, Linear, TrainingLinear};
use oxide_forge::net::mlp::{InferenceMLP, TrainingMlp};
use oxide_forge::net::node::{
    BinaryNode, ConcatNode, FanOutNode, RowReduceNode, SingleNode, SplitNode, TrainingBinaryNode,
    TrainingConcatNode, TrainingFanOutNode, TrainingRowReduceNode, TrainingSingleNode,
    TrainingSplitNode,
};
use oxide_forge::net::swiglu::{InferenceSwiglu, TrainingSwiglu};
use oxide_forge::net::transformer::{
    InferenceDecoder, InferenceEncoder, TrainingDecoder, TrainingEncoder,
};

fn assert_graph_node<T: GraphNode>() {}

#[test]
fn linear_and_transformers_are_graph_nodes() {
    assert_graph_node::<Linear>();
    assert_graph_node::<TrainingLinear>();
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
    assert_graph_node::<FanOutNode>();
    assert_graph_node::<TrainingFanOutNode>();
    assert_graph_node::<RowReduceNode>();
    assert_graph_node::<TrainingRowReduceNode>();
}

/// Persistent device smoke test. Run it through the CUDA-Oxide test workflow
/// on a CUDA-capable machine.
#[test]
#[ignore = "requires a CUDA device and a CUDA-Oxide device artifact"]
fn training_linear_graph_node_forward_backward_and_learn() {
    let mut runtime = CudaRuntime::new().unwrap();
    let weights = runtime
        .matrix_from_host(&[1.0, 0.0, 0.0, 1.0], 2, 2, None)
        .unwrap();
    let bias = runtime.vector_from_host(&[0.0, 0.0], None).unwrap();
    let mut linear = TrainingLinear::new(Linear::new(weights, Some(bias), Activation::Identity));

    let input = runtime
        .matrix_from_host(&[1.0, 2.0, 3.0, 4.0], 2, 2, None)
        .unwrap();
    let output = GraphNode::forward(&mut linear, vec![input], &mut runtime)
        .pop()
        .unwrap();
    assert_eq!(output.to_host(&runtime, None), vec![1.0, 2.0, 3.0, 4.0]);

    let gradient = runtime
        .matrix_from_host(&[1.0, 1.0, 1.0, 1.0], 2, 2, None)
        .unwrap();
    let input_gradient = GraphNode::backward(&mut linear, vec![gradient], &mut runtime)
        .pop()
        .unwrap();
    assert_eq!(input_gradient.to_host(&runtime, None), vec![1.0; 4]);

    GraphNode::learn(&mut linear, LearnConfig::single(0.01), &mut runtime);
    GraphNode::clear_cache(&mut linear, &mut runtime);
    runtime.recycle_matrix(output);
    runtime.recycle_matrix(input_gradient);
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
