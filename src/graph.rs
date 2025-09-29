//! Computation graph representation and execution engine
//!
//! This module provides the core graph abstraction for representing and executing
//! neural networks. It handles operator dependencies, topological sorting, and
//! memory management during inference.
use crate::errors::{Result, OrysError};
use crate::ops::Operator;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// A single node in the computation graph
///
/// Each node represents one operation with its inputs and outputs.
/// Nodes are connected via named tensors that flow between them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Unique identifier for this node
    pub name: String,
    /// Type of operation (e.g., "MatMul", "Add", "ReLU")
    pub op_type: String,
    /// Names of input tensors this node consumes
    pub inputs: Vec<String>,
    /// Names of output tensors this node produces
    pub outputs: Vec<String>,
}

/// Initializer tensor with constant data
///
/// Represents model weights and biases that are loaded once
/// and remain constant during inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Initializer {
    /// Name of the tensor
    pub name: String,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Constant data values
    pub data: Vec<f32>,
}

impl Initializer {
    /// Convert initializer to a tensor
    pub fn to_tensor(&self) -> Result<Tensor> {
        Tensor::new(self.shape.clone(), self.data.clone())
    }
}

/// Complete computation graph with execution capabilities
///
/// Represents a neural network as a directed acyclic graph (DAG)
/// of operations connected by named tensors.
///
/// # Examples
/// ```rust
/// use orys::graph::ComputeGraph;
/// use orys::ops::create_operator;
///
/// let mut graph = ComputeGraph::new();
/// 
/// // Add a simple linear layer: input -> MatMul -> Add -> output
/// graph.add_node("matmul", create_operator("MatMul")?, vec!["input", "weight"], vec!["hidden"])?;
/// graph.add_node("add", create_operator("Add")?, vec!["hidden", "bias"], vec!["output"])?;
/// 
/// graph.add_input("input");
/// graph.add_output("output");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct ComputeGraph {
    /// All nodes in the graph, indexed by name
    nodes: HashMap<String, GraphNode>,
    /// Operators for each node, indexed by node name
    operators: HashMap<String, Box<dyn Operator>>,
    /// Constant tensors (weights, biases)
    initializers: HashMap<String, Tensor>,
    /// Names of graph input tensors
    inputs: Vec<String>,
    /// Names of graph output tensors  
    outputs: Vec<String>,
    /// Cached topological execution order
    execution_order: Option<Vec<String>>,
}

impl ComputeGraph {
    /// Create a new empty computation graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            operators: HashMap::new(),
            initializers: HashMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            execution_order: None,
        }
    }

    /// Add a node to the graph
    ///
    /// # Arguments
    /// * `name` - Unique identifier for the node
    /// * `operator` - The operator implementation
    /// * `inputs` - Names of input tensors
    /// * `outputs` - Names of output tensors
    ///
    /// # Errors
    /// Returns error if node name already exists
    pub fn add_node(
        &mut self,
        name: String,
        operator: Box<dyn Operator>,
        inputs: Vec<String>,
        outputs: Vec<String>,
    ) -> Result<()> {
        if self.nodes.contains_key(&name) {
            return Err(OrysError::invalid_model(format!(
                "Node '{}' already exists in graph",
                name
            )));
        }

        let node = GraphNode {
            name: name.clone(),
            op_type: operator.op_type().to_string(),
            inputs,
            outputs,
        };

        self.nodes.insert(name.clone(), node);
        self.operators.insert(name, operator);
        
        // Invalidate cached execution order
        self.execution_order = None;
        
        Ok(())
    }

    /// Add an initializer tensor (weights, biases)
    ///
    /// # Arguments
    /// * `name` - Name of the tensor
    /// * `tensor` - The constant tensor data
    pub fn add_initializer(&mut self, name: String, tensor: Tensor) {
        self.initializers.insert(name, tensor);
    }

    /// Add an input tensor name to the graph
    pub fn add_input(&mut self, name: String) {
        if !self.inputs.contains(&name) {
            self.inputs.push(name);
        }
    }

    /// Add an output tensor name to the graph
    pub fn add_output(&mut self, name: String) {
        if !self.outputs.contains(&name) {
            self.outputs.push(name);
        }
    }

    /// Get the graph input names
    pub fn inputs(&self) -> &[String] {
        &self.inputs
    }

    /// Get the graph output names
    pub fn outputs(&self) -> &[String] {
        &self.outputs
    }

    /// Get a node by name
    pub fn get_node(&self, name: &str) -> Option<&GraphNode> {
        self.nodes.get(name)
    }

    /// Get an initializer tensor by name
    pub fn get_initializer(&self, name: &str) -> Option<&Tensor> {
        self.initializers.get(name)
    }

    /// Validate the graph structure
    ///
    /// Checks for:
    /// - At least one input and output
    /// - No circular dependencies
    /// - All tensor references are valid
    pub fn validate(&self) -> Result<()> {
        // Check for inputs and outputs
        if self.inputs.is_empty() {
            return Err(OrysError::NoInputNodes);
        }
        if self.outputs.is_empty() {
            return Err(OrysError::NoOutputNodes);
        }

        // Check for circular dependencies by attempting topological sort
        self.compute_execution_order()?;

        // Validate all tensor references
        self.validate_tensor_references()?;

        Ok(())
    }

    /// Validate that all tensor references are valid
    fn validate_tensor_references(&self) -> Result<()> {
        let mut all_tensors = HashSet::new();
        
        // Add graph inputs
        for input in &self.inputs {
            all_tensors.insert(input.clone());
        }
        
        // Add initializers
        for name in self.initializers.keys() {
            all_tensors.insert(name.clone());
        }
        
        // Add node outputs
        for node in self.nodes.values() {
            for output in &node.outputs {
                all_tensors.insert(output.clone());
            }
        }

        // Check that all node inputs reference valid tensors
        for node in self.nodes.values() {
            for input in &node.inputs {
                if !all_tensors.contains(input) {
                    return Err(OrysError::invalid_model(format!(
                        "Node '{}' references undefined tensor '{}'",
                        node.name, input
                    )));
                }
            }
        }

        // Check that all graph outputs are produced by some node
        for output in &self.outputs {
            if !all_tensors.contains(output) {
                return Err(OrysError::invalid_model(format!(
                    "Graph output '{}' is not produced by any node",
                    output
                )));
            }
        }

        Ok(())
    }

    /// Compute topological execution order using Kahn's algorithm
    ///
    /// Returns the order in which nodes should be executed to respect dependencies.
    fn compute_execution_order(&self) -> Result<Vec<String>> {
        if let Some(ref order) = self.execution_order {
            return Ok(order.clone());
        }

        // Build dependency graph
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut dependencies: HashMap<String, Vec<String>> = HashMap::new();
        
        // Initialize in-degree count for all nodes
        for node_name in self.nodes.keys() {
            in_degree.insert(node_name.clone(), 0);
            dependencies.insert(node_name.clone(), Vec::new());
        }

        // Build the dependency relationships
        for node in self.nodes.values() {
            for input_tensor in &node.inputs {
                // Find which node produces this input tensor
                if let Some(producer) = self.find_tensor_producer(input_tensor) {
                    dependencies.get_mut(&producer).unwrap().push(node.name.clone());
                    *in_degree.get_mut(&node.name).unwrap() += 1;
                }
                // If no producer found, it's either a graph input or initializer (OK)
            }
        }

        // Kahn's algorithm for topological sorting
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        // Start with nodes that have no dependencies
        for (node_name, &degree) in &in_degree {
            if degree == 0 {
                queue.push_back(node_name.clone());
            }
        }

        while let Some(node_name) = queue.pop_front() {
            result.push(node_name.clone());

            // Remove this node and decrease in-degree of dependents
            for dependent in &dependencies[&node_name] {
                let new_degree = in_degree[dependent] - 1;
                in_degree.insert(dependent.clone(), new_degree);
                
                if new_degree == 0 {
                    queue.push_back(dependent.clone());
                }
            }
        }

        // Check for circular dependencies
        if result.len() != self.nodes.len() {
            let remaining_nodes: Vec<String> = in_degree
                .iter()
                .filter(|(_, degree)| **degree > 0)
                .map(|(name, _)| name.clone())
                .collect();
            
            return Err(OrysError::CircularDependency {
                nodes: remaining_nodes,
            });
        }

        Ok(result)
    }

    /// Find which node produces a given tensor
    fn find_tensor_producer(&self, tensor_name: &str) -> Option<String> {
        for node in self.nodes.values() {
            if node.outputs.contains(&tensor_name.to_string()) {
                return Some(node.name.clone());
            }
        }
        None
    }

    /// Execute the computation graph with given inputs
    ///
    /// # Arguments
    /// * `inputs` - Map of input tensor names to tensor data
    ///
    /// # Returns
    /// * `Result<HashMap<String, Tensor>>` - Map of output tensor names to results
    ///
    /// # Examples
    /// ```rust
    /// # use orys::graph::ComputeGraph;
    /// # use orys::tensor::Tensor;
    /// # use std::collections::HashMap;
    /// # let graph = ComputeGraph::new();
    /// let mut inputs = HashMap::new();
    /// inputs.insert("input".to_string(), Tensor::ones(vec![1, 3]));
    /// 
    /// let outputs = graph.execute(inputs)?;
    /// let result = &outputs["output"];
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn execute(&mut self, inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        // Validate inputs
        self.validate_execution_inputs(&inputs)?;

        // Get execution order
        let execution_order = self.compute_execution_order()?;
        self.execution_order = Some(execution_order.clone());

        // Initialize tensor storage with inputs and initializers
        let mut tensors = inputs;
        for (name, tensor) in &self.initializers {
            tensors.insert(name.clone(), tensor.clone());
        }

        // Execute nodes in topological order
        for node_name in execution_order {
            let node = self.nodes[&node_name].clone();
            let operator = &self.operators[&node_name];

            // Gather input tensors for this node
            let mut node_inputs = Vec::new();
            for input_name in &node.inputs {
                match tensors.get(input_name) {
                    Some(tensor) => node_inputs.push(tensor.clone()),
                    None => {
                        return Err(OrysError::NodeNotFound {
                            node_name: input_name.clone(),
                        });
                    }
                }
            }

            // Execute the operator
            let result = operator.execute(&node_inputs).map_err(|e| {
                OrysError::inference_error(format!(
                    "Node '{}' ({}) execution failed: {}",
                    node_name, node.op_type, e
                ))
            })?;

            // Store output tensors
            if node.outputs.len() != 1 {
                return Err(OrysError::inference_error(format!(
                    "Node '{}' produces {} outputs, but only single-output nodes are currently supported",
                    node_name, node.outputs.len()
                )));
            }
            
            tensors.insert(node.outputs[0].clone(), result);
        }

        // Extract output tensors
        let mut outputs = HashMap::new();
        for output_name in &self.outputs {
            match tensors.get(output_name) {
                Some(tensor) => {
                    outputs.insert(output_name.clone(), tensor.clone());
                }
                None => {
                    return Err(OrysError::inference_error(format!(
                        "Output tensor '{}' was not produced during execution",
                        output_name
                    )));
                }
            }
        }

        Ok(outputs)
    }

    /// Validate that provided inputs match graph requirements
    fn validate_execution_inputs(&self, inputs: &HashMap<String, Tensor>) -> Result<()> {
        // Check that all required inputs are provided
        for required_input in &self.inputs {
            if !inputs.contains_key(required_input) {
                return Err(OrysError::MissingInput {
                    input_name: required_input.clone(),
                });
            }
        }

        // Check for unexpected inputs
        for provided_input in inputs.keys() {
            if !self.inputs.contains(provided_input) {
                return Err(OrysError::inference_error(format!(
                    "Unexpected input '{}' provided. Expected inputs: {:?}",
                    provided_input, self.inputs
                )));
            }
        }

        Ok(())
    }

    /// Get execution statistics
    pub fn execution_stats(&self) -> ExecutionStats {
        ExecutionStats {
            node_count: self.nodes.len(),
            initializer_count: self.initializers.len(),
            input_count: self.inputs.len(),
            output_count: self.outputs.len(),
        }
    }
}

impl Default for ComputeGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the computation graph
#[derive(Debug, Clone)]
pub struct ExecutionStats {
    pub node_count: usize,
    pub initializer_count: usize,
    pub input_count: usize,
    pub output_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::create_operator;
    use std::collections::HashMap;

    #[test]
    fn test_graph_construction() {
        let mut graph = ComputeGraph::new();
        
        let matmul_op = create_operator("MatMul").unwrap();
        graph.add_node(
            "matmul1".to_string(),
            matmul_op,
            vec!["input".to_string(), "weight".to_string()],
            vec!["hidden".to_string()],
        ).unwrap();
        
        graph.add_input("input".to_string());
        graph.add_output("hidden".to_string());
        
        // Add weight initializer
        let weight = Tensor::ones(vec![3, 2]);
        graph.add_initializer("weight".to_string(), weight);
        
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs().len(), 1);
        assert_eq!(graph.outputs().len(), 1);
    }

    #[test]
    fn test_duplicate_node_error() {
        let mut graph = ComputeGraph::new();
        
        let op1 = create_operator("Add").unwrap();
        let op2 = create_operator("Add").unwrap();
        
        graph.add_node("node1".to_string(), op1, vec![], vec![]).unwrap();
        let result = graph.add_node("node1".to_string(), op2, vec![], vec![]);
        
        assert!(result.is_err());
    }

    #[test]
    fn test_topological_sort() {
        let mut graph = ComputeGraph::new();
        
        // Create a simple linear network: input -> matmul -> add -> relu -> output
        graph.add_node(
            "matmul".to_string(),
            create_operator("MatMul").unwrap(),
            vec!["input".to_string(), "weight1".to_string()],
            vec!["hidden1".to_string()],
        ).unwrap();
        
        graph.add_node(
            "add".to_string(),
            create_operator("Add").unwrap(),
            vec!["hidden1".to_string(), "bias1".to_string()],
            vec!["hidden2".to_string()],
        ).unwrap();
        
        graph.add_node(
            "relu".to_string(),
            create_operator("ReLU").unwrap(),
            vec!["hidden2".to_string()],
            vec!["output".to_string()],
        ).unwrap();
        
        graph.add_input("input".to_string());
        graph.add_output("output".to_string());
        
        let order = graph.compute_execution_order().unwrap();
        
        // Verify correct order: matmul -> add -> relu
        assert_eq!(order, vec!["matmul", "add", "relu"]);
    }

    #[test]
    fn test_circular_dependency_detection() {
        let mut graph = ComputeGraph::new();
        
        // Create an impossible circular dependency
        graph.add_node(
            "node1".to_string(),
            create_operator("Add").unwrap(),
            vec!["output2".to_string()],
            vec!["output1".to_string()],
        ).unwrap();
        
        graph.add_node(
            "node2".to_string(),
            create_operator("Add").unwrap(),
            vec!["output1".to_string()],
            vec!["output2".to_string()],
        ).unwrap();
        
        let result = graph.compute_execution_order();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OrysError::CircularDependency { .. }));
    }

    #[test]
    fn test_simple_execution() {
        let mut graph = ComputeGraph::new();
        
        // Create: input -> ReLU -> output
        graph.add_node(
            "relu".to_string(),
            create_operator("ReLU").unwrap(),
            vec!["input".to_string()],
            vec!["output".to_string()],
        ).unwrap();
        
        graph.add_input("input".to_string());
        graph.add_output("output".to_string());
        
        // Execute with test input
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0]));
        
        let outputs = graph.execute(inputs).unwrap();
        let result = &outputs["output"];
        
        assert_eq!(result.data(), &[0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_linear_layer_execution() {
        let mut graph = ComputeGraph::new();
        
        // Create: input -> MatMul(weight) -> Add(bias) -> output
        graph.add_node(
            "matmul".to_string(),
            create_operator("MatMul").unwrap(),
            vec!["input".to_string(), "weight".to_string()],
            vec!["hidden".to_string()],
        ).unwrap();
        
        graph.add_node(
            "add".to_string(),
            create_operator("Add").unwrap(),
            vec!["hidden".to_string(), "bias".to_string()],
            vec!["output".to_string()],
        ).unwrap();
        
        graph.add_input("input".to_string());
        graph.add_output("output".to_string());
        
        // Add initializers
        let weight = Tensor::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap(); // Identity matrix
        let bias = Tensor::from_vec(vec![1.0, 1.0]);
        
        graph.add_initializer("weight".to_string(), weight);
        graph.add_initializer("bias".to_string(), bias);
        
        // Execute
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), Tensor::new(vec![1, 2], vec![2.0, 3.0]).unwrap());
        
        let outputs = graph.execute(inputs).unwrap();
        let result = &outputs["output"];
        
        // Input [2, 3] * Identity + [1, 1] = [3, 4]
        assert_eq!(result.data(), &[3.0, 4.0]);
    }

    #[test]
    fn test_missing_input_error() {
        let mut graph = ComputeGraph::new();
        
        graph.add_node(
            "relu".to_string(),
            create_operator("ReLU").unwrap(),
            vec!["input".to_string()],
            vec!["output".to_string()],
        ).unwrap();
        
        graph.add_input("input".to_string());
        graph.add_output("output".to_string());
        
        // Execute without providing required input
        let inputs = HashMap::new();
        let result = graph.execute(inputs);
        
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OrysError::MissingInput { .. }));
    }

    #[test]
    fn test_graph_validation() {
        let mut graph = ComputeGraph::new();
        
        // Graph with no inputs should fail validation
        graph.add_output("output".to_string());
        assert!(graph.validate().is_err());
        
        // Graph with no outputs should fail validation  
        let mut graph2 = ComputeGraph::new();
        graph2.add_input("input".to_string());
        assert!(graph2.validate().is_err());
    }

    #[test]
    fn test_execution_stats() {
        let mut graph = ComputeGraph::new();
        
        graph.add_node(
            "node1".to_string(),
            create_operator("ReLU").unwrap(),
            vec!["input".to_string()],
            vec!["output".to_string()],
        ).unwrap();
        
        graph.add_input("input".to_string());
        graph.add_output("output".to_string());
        graph.add_initializer("weight".to_string(), Tensor::ones(vec![2, 2]));
        
        let stats = graph.execution_stats();
        
        assert_eq!(stats.node_count, 1);
        assert_eq!(stats.input_count, 1);
        assert_eq!(stats.output_count, 1);
        assert_eq!(stats.initializer_count, 1);
    }
}