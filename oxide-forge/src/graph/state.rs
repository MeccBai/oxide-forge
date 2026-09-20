use crate::cuda::CudaRuntime;
use crate::cuda::container::{Matrix, Vector};

/// Forward state with the lifetime of one forward/backward pair.
#[derive(Default)]
pub struct NodeCache {
    pub(crate) inputs: Vec<Matrix>,
    pub(crate) outputs: Vec<Matrix>,
    pub(crate) input_shapes: Vec<(usize, usize)>,
    pub(crate) forward_pending: bool,
}

/// Parameter-sized storage whose lifetime spans optimizer steps.
pub(crate) enum ParameterBuffer {
    Matrix(Matrix),
    Vector(Vector),
}

#[derive(Default)]
pub struct ParameterTrainState {
    pub(crate) gradient: Option<ParameterBuffer>,
    pub(crate) velocity: Option<ParameterBuffer>,
}

/// Training-only state paired with one otherwise inference-capable block.
#[derive(Default)]
pub struct NodeTrainState {
    pub(crate) cache: NodeCache,
    pub(crate) parameters: Vec<ParameterTrainState>,
    pub(crate) children: Vec<NodeTrainState>,
}

impl NodeTrainState {
    pub fn with_parameter_count(parameter_count: usize) -> Self {
        Self {
            cache: NodeCache::default(),
            parameters: (0..parameter_count)
                .map(|_| ParameterTrainState::default())
                .collect(),
            children: Vec::new(),
        }
    }

    pub fn with_children(children: Vec<NodeTrainState>) -> Self {
        Self {
            children,
            ..Self::default()
        }
    }

    pub fn cache(&self) -> &NodeCache {
        &self.cache
    }

    pub fn cache_mut(&mut self) -> &mut NodeCache {
        &mut self.cache
    }

    pub fn children(&self) -> &[NodeTrainState] {
        &self.children
    }

    pub fn children_mut(&mut self) -> &mut [NodeTrainState] {
        &mut self.children
    }

    pub fn parameter_count(&self) -> usize {
        self.parameters.len()
    }

    pub fn clear_cache(&mut self, runtime: &mut CudaRuntime) {
        for input in self.cache.inputs.drain(..) {
            runtime.recycle_matrix(input);
        }
        for output in self.cache.outputs.drain(..) {
            runtime.recycle_matrix(output);
        }
        self.cache.input_shapes.clear();
        self.cache.forward_pending = false;
        for child in &mut self.children {
            child.clear_cache(runtime);
        }
    }
}

/// Graph-owned training arena. Its indices match the executable node list.
#[derive(Default)]
pub struct TrainState {
    nodes: Vec<NodeTrainState>,
}

impl TrainState {
    pub fn new(nodes: Vec<NodeTrainState>) -> Self {
        Self { nodes }
    }

    pub fn nodes(&self) -> &[NodeTrainState] {
        &self.nodes
    }

    pub fn nodes_mut(&mut self) -> &mut [NodeTrainState] {
        &mut self.nodes
    }
}

#[cfg(test)]
mod tests {
    use super::NodeTrainState;

    #[test]
    fn compound_state_keeps_parameter_state_outside_blocks() {
        let state = NodeTrainState::with_children(vec![
            NodeTrainState::with_parameter_count(2),
            NodeTrainState::with_parameter_count(1),
        ]);

        assert_eq!(state.parameter_count(), 0);
        assert_eq!(state.children().len(), 2);
        assert_eq!(state.children()[0].parameter_count(), 2);
        assert_eq!(state.children()[1].parameter_count(), 1);
    }
}
