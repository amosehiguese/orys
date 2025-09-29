//! JSON model format loader
//!
//! This module implements loading of models from a custom JSON format.
//! The JSON format is designed to be human-readable and easy to create
//! for testing and development purposes.

use crate::errors::{Result, ZekeError};
use crate::graph::{ComputeGraph, Initializer};
use crate::ops::create_operator;
use crate::loader::ModelLoader;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// JSON model format schema
///
/// This represents the complete structure of our JSON model files.
/// The format is designed to closely mirror ONNX structure while
/// remaining human-readable and easy to create manually.
///
/// # Example JSON Structure
/// ```json
/// {
///   "version": "1.0",
///   "metadata": {
///     "name": "simple_classifier",
///     "description": "A simple 2-layer neural network"
///   },
///   "inputs": [
///     {
///       "name": "input",
///       "shape": [1, 784],
///       "description": "Flattened 28x28 image"
///     }
///   ],
///   "outputs": [
///     {
///       "name": "output",
///       "shape": [1, 10],
///       "description": "Class probabilities"
///     }
///   ],
///   "initializers": [
///     {
///       "name": "weight1",
///       "shape": [784, 128],
///       "data": [0.1, 0.2, ...]
///     }
///   ],
///   "nodes": [
///     {
///       "name": "fc1",
///       "op_type": "MatMul",
///       "inputs": ["input", "weight1"],
///       "outputs": ["hidden1"]
///     }
///   ]
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonModel {
    /// Schema version for compatibility checking
    pub version: String,
    /// Optional metadata about the model
    #[serde(default)]
    pub metadata: ModelMetadata,
    /// Graph input specifications
    pub inputs: Vec<TensorSpec>,
    /// Graph output specifications
    pub outputs: Vec<TensorSpec>,
    /// Constant tensors (weights, biases)
    #[serde(default)]
    pub initializers: Vec<JsonInitializer>,
    /// Computation nodes
    pub nodes: Vec<JsonNode>,
}

/// Model metadata information
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Human-readable model name
    #[serde(default)]
    pub name: String,
    /// Model description
    #[serde(default)]
    pub description: String,
    /// Model author/creator
    #[serde(default)]
    pub author: String,
    /// Model version
    #[serde(default)]
    pub model_version: String,
    /// Additional key-value properties
    #[serde(default)]
    pub properties: HashMap<String, String>,
}

/// Tensor specification with shape and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpec {
    /// Tensor name (used for graph connections)
    pub name: String,
    /// Expected tensor shape
    pub shape: Vec<usize>,
    /// Optional human-readable description
    #[serde(default)]
    pub description: String,
    /// Optional data type (currently only f32 supported)
    #[serde(default = "default_dtype")]
    pub dtype: String,
}

fn default_dtype() -> String {
    "f32".to_string()
}

/// Initializer tensor in JSON format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonInitializer {
    /// Tensor name
    pub name: String,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Tensor data as flat array
    pub data: Vec<f32>,
    /// Optional description
    #[serde(default)]
    pub description: String,
}

impl JsonInitializer {
    /// Convert to internal Initializer format
    pub fn to_initializer(&self) -> Result<Initializer> {
        Ok(Initializer {
            name: self.name.clone(),
            shape: self.shape.clone(),
            data: self.data.clone(),
        })
    }
}

/// Computation node in JSON format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonNode {
    /// Unique node identifier
    pub name: String,
    /// Operator type (e.g., "MatMul", "Add", "ReLU")
    pub op_type: String,
    /// Input tensor names
    pub inputs: Vec<String>,
    /// Output tensor names
    pub outputs: Vec<String>,
    /// Optional node description
    #[serde(default)]
    pub description: String,
    /// Optional operator-specific attributes
    #[serde(default)]
    pub attributes: HashMap<String, serde_json::Value>,
}

/// JSON model format loader
pub struct JsonLoader;

impl JsonLoader {
    /// Create a new JSON loader
    pub fn new() -> Self {
        Self
    }

    /// Validate a JSON model file without full loading
    ///
    /// Performs basic JSON parsing and schema validation without
    /// building the computation graph.
    pub fn validate_file(path: &Path) -> Result<()> {
        let content = std::fs::read_to_string(path).map_err(ZekeError::Io)?;
        let _model: JsonModel = serde_json::from_str(&content).map_err(ZekeError::Json)?;
        
        // Additional validation could go here (version compatibility, etc.)
        
        Ok(())
    }

    /// Inspect a JSON model file and return metadata
    ///
    /// Returns structural information about the model without building
    /// the full computation graph.
    pub fn inspect_file(
        path: &Path,
    ) -> Result<(usize, usize, usize, usize, Vec<String>, Vec<String>)> {
        let content = std::fs::read_to_string(path).map_err(ZekeError::Io)?;
        let model: JsonModel = serde_json::from_str(&content).map_err(ZekeError::Json)?;
        
        let node_count = model.nodes.len();
        let input_count = model.inputs.len();
        let output_count = model.outputs.len();
        let initializer_count = model.initializers.len();
        let input_names = model.inputs.iter().map(|i| i.name.clone()).collect();
        let output_names = model.outputs.iter().map(|o| o.name.clone()).collect();
        
        Ok((node_count, input_count, output_count, initializer_count, input_names, output_names))
    }

    /// Parse and validate the JSON model structure
    fn parse_model(&self, path: &Path) -> Result<JsonModel> {
        let content = std::fs::read_to_string(path).map_err(ZekeError::Io)?;
        let model: JsonModel = serde_json::from_str(&content).map_err(ZekeError::Json)?;
        
        // Validate version compatibility
        self.validate_version(&model.version)?;
        
        // Validate model structure
        self.validate_model_structure(&model)?;
        
        Ok(model)
    }

    /// Check version compatibility
    fn validate_version(&self, version: &str) -> Result<()> {
        // For now, we only support version 1.0
        // Future versions could be handled with migration logic
        if version != "1.0" {
            return Err(ZekeError::invalid_model(format!(
                "Unsupported JSON model version: {}. Supported versions: 1.0",
                version
            )));
        }
        Ok(())
    }

    /// Validate the overall model structure
    fn validate_model_structure(&self, model: &JsonModel) -> Result<()> {
        // Check that we have at least one input and output
        if model.inputs.is_empty() {
            return Err(ZekeError::invalid_model(
                "Model must have at least one input".to_string(),
            ));
        }
        
        if model.outputs.is_empty() {
            return Err(ZekeError::invalid_model(
                "Model must have at least one output".to_string(),
            ));
        }
        
        // Check that we have at least one node
        if model.nodes.is_empty() {
            return Err(ZekeError::invalid_model(
                "Model must have at least one computation node".to_string(),
            ));
        }

        // Validate tensor specs
        for input in &model.inputs {
            self.validate_tensor_spec(input, "input")?;
        }
        
        for output in &model.outputs {
            self.validate_tensor_spec(output, "output")?;
        }

        // Validate initializers
        for init in &model.initializers {
            self.validate_initializer(init)?;
        }

        // Validate nodes
        for node in &model.nodes {
            self.validate_node(node)?;
        }

        Ok(())
    }

    /// Validate a tensor specification
    fn validate_tensor_spec(&self, spec: &TensorSpec, context: &str) -> Result<()> {
        if spec.name.is_empty() {
            return Err(ZekeError::invalid_model(format!(
                "{} tensor must have a non-empty name",
                context
            )));
        }

        if spec.shape.is_empty() {
            return Err(ZekeError::invalid_model(format!(
                "{} tensor '{}' must have a non-empty shape",
                context, spec.name
            )));
        }

        if spec.shape.iter().any(|&dim| dim == 0) {
            return Err(ZekeError::invalid_model(format!(
                "{} tensor '{}' has zero-sized dimension in shape {:?}",
                context, spec.name, spec.shape
            )));
        }

        if spec.dtype != "f32" {
            return Err(ZekeError::invalid_model(format!(
                "{} tensor '{}' has unsupported dtype '{}'. Only 'f32' is supported",
                context, spec.name, spec.dtype
            )));
        }

        Ok(())
    }

    /// Validate an initializer
    fn validate_initializer(&self, init: &JsonInitializer) -> Result<()> {
        if init.name.is_empty() {
            return Err(ZekeError::invalid_model(
                "Initializer must have a non-empty name".to_string(),
            ));
        }

        if init.shape.is_empty() {
            return Err(ZekeError::invalid_model(format!(
                "Initializer '{}' must have a non-empty shape",
                init.name
            )));
        }

        let expected_size: usize = init.shape.iter().product();
        if init.data.len() != expected_size {
            return Err(ZekeError::invalid_model(format!(
                "Initializer '{}' data size {} doesn't match shape {:?} (expected {} elements)",
                init.name, init.data.len(), init.shape, expected_size
            )));
        }

        Ok(())
    }

    /// Validate a computation node
    fn validate_node(&self, node: &JsonNode) -> Result<()> {
        if node.name.is_empty() {
            return Err(ZekeError::invalid_model(
                "Node must have a non-empty name".to_string(),
            ));
        }

        if node.op_type.is_empty() {
            return Err(ZekeError::invalid_model(format!(
                "Node '{}' must have a non-empty op_type",
                node.name
            )));
        }

        if node.outputs.is_empty() {
            return Err(ZekeError::invalid_model(format!(
                "Node '{}' must have at least one output",
                node.name
            )));
        }

        // Validate that operator type is supported
        create_operator(&node.op_type).map_err(|_| {
            ZekeError::invalid_model(format!(
                "Node '{}' uses unsupported operator type '{}'",
                node.name, node.op_type
            ))
        })?;

        Ok(())
    }

    /// Convert JSON model to computation graph
    fn build_graph(&self, model: JsonModel) -> Result<ComputeGraph> {
        let mut graph = ComputeGraph::new();

        // Add initializers first
        for json_init in model.initializers {
            let initializer = json_init.to_initializer()?;
            let tensor = initializer.to_tensor()?;
            graph.add_initializer(initializer.name, tensor);
        }

        // Add nodes
        for json_node in model.nodes {
            let operator = create_operator(&json_node.op_type)?;
            graph.add_node(
                json_node.name,
                operator,
                json_node.inputs,
                json_node.outputs,
            )?;
        }

        // Add inputs and outputs
        for input_spec in model.inputs {
            graph.add_input(input_spec.name);
        }

        for output_spec in model.outputs {
            graph.add_output(output_spec.name);
        }

        // Validate the constructed graph
        graph.validate()?;

        Ok(graph)
    }
}

impl Default for JsonLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelLoader for JsonLoader {
    fn load(&self, path: &Path) -> Result<ComputeGraph> {
        let model = self.parse_model(path)?;
        self.build_graph(model)
    }

    fn format_name(&self) -> &'static str {
        "JSON"
    }

    fn can_load(&self, path: &Path) -> bool {
        // Check file extension
        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            if extension.to_lowercase() == "json" {
                return true;
            }
        }
        
        // Check file content
        if let Ok(content) = std::fs::read_to_string(path) {
            if let Ok(_) = serde_json::from_str::<JsonModel>(&content) {
                return true;
            }
        }
        
        false
    }
}

/// Helper function to create a simple JSON model for testing
///
/// Creates a basic linear classifier model with the given input/output dimensions.
/// Useful for generating test models and examples.
///
/// # Arguments
/// * `input_size` - Size of the input layer
/// * `output_size` - Size of the output layer
/// * `hidden_size` - Size of the hidden layer (optional)
///
/// # Returns
/// * `JsonModel` - Complete model ready for serialization
pub fn create_simple_model(input_size: usize, output_size: usize, hidden_size: Option<usize>) -> JsonModel {
    let mut model = JsonModel {
        version: "1.0".to_string(),
        metadata: ModelMetadata {
            name: "simple_model".to_string(),
            description: "Auto-generated simple model".to_string(),
            ..Default::default()
        },
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![1, input_size],
            description: "Model input".to_string(),
            dtype: "f32".to_string(),
        }],
        outputs: vec![TensorSpec {
            name: "output".to_string(),
            shape: vec![1, output_size],
            description: "Model output".to_string(),
            dtype: "f32".to_string(),
        }],
        initializers: Vec::new(),
        nodes: Vec::new(),
    };

    if let Some(hidden_size) = hidden_size {
        // Two-layer model: input -> hidden -> output
        
        // Weight matrices (Xavier initialization placeholder)
        model.initializers.push(JsonInitializer {
            name: "weight1".to_string(),
            shape: vec![input_size, hidden_size],
            data: vec![0.1; input_size * hidden_size],
            description: "First layer weights".to_string(),
        });
        
        model.initializers.push(JsonInitializer {
            name: "bias1".to_string(),
            shape: vec![hidden_size],
            data: vec![0.0; hidden_size],
            description: "First layer bias".to_string(),
        });
        
        model.initializers.push(JsonInitializer {
            name: "weight2".to_string(),
            shape: vec![hidden_size, output_size],
            data: vec![0.1; hidden_size * output_size],
            description: "Second layer weights".to_string(),
        });
        
        model.initializers.push(JsonInitializer {
            name: "bias2".to_string(),
            shape: vec![output_size],
            data: vec![0.0; output_size],
            description: "Second layer bias".to_string(),
        });

        // Network structure
        model.nodes.extend(vec![
            JsonNode {
                name: "fc1".to_string(),
                op_type: "MatMul".to_string(),
                inputs: vec!["input".to_string(), "weight1".to_string()],
                outputs: vec!["hidden1".to_string()],
                description: "First linear layer".to_string(),
                attributes: HashMap::new(),
            },
            JsonNode {
                name: "add1".to_string(),
                op_type: "Add".to_string(),
                inputs: vec!["hidden1".to_string(), "bias1".to_string()],
                outputs: vec!["hidden2".to_string()],
                description: "First layer bias".to_string(),
                attributes: HashMap::new(),
            },
            JsonNode {
                name: "relu1".to_string(),
                op_type: "ReLU".to_string(),
                inputs: vec!["hidden2".to_string()],
                outputs: vec!["hidden3".to_string()],
                description: "First layer activation".to_string(),
                attributes: HashMap::new(),
            },
            JsonNode {
                name: "fc2".to_string(),
                op_type: "MatMul".to_string(),
                inputs: vec!["hidden3".to_string(), "weight2".to_string()],
                outputs: vec!["hidden4".to_string()],
                description: "Second linear layer".to_string(),
                attributes: HashMap::new(),
            },
            JsonNode {
                name: "add2".to_string(),
                op_type: "Add".to_string(),
                inputs: vec!["hidden4".to_string(), "bias2".to_string()],
                outputs: vec!["output".to_string()],
                description: "Second layer bias".to_string(),
                attributes: HashMap::new(),
            },
        ]);
    } else {
        // Single-layer model: input -> output
        
        model.initializers.push(JsonInitializer {
            name: "weight".to_string(),
            shape: vec![input_size, output_size],
            data: vec![0.1; input_size * output_size],
            description: "Layer weights".to_string(),
        });
        
        model.initializers.push(JsonInitializer {
            name: "bias".to_string(),
            shape: vec![output_size],
            data: vec![0.0; output_size],
            description: "Layer bias".to_string(),
        });

        model.nodes.extend(vec![
            JsonNode {
                name: "fc".to_string(),
                op_type: "MatMul".to_string(),
                inputs: vec!["input".to_string(), "weight".to_string()],
                outputs: vec!["hidden".to_string()],
                description: "Linear layer".to_string(),
                attributes: HashMap::new(),
            },
            JsonNode {
                name: "add".to_string(),
                op_type: "Add".to_string(),
                inputs: vec!["hidden".to_string(), "bias".to_string()],
                outputs: vec!["output".to_string()],
                description: "Bias addition".to_string(),
                attributes: HashMap::new(),
            },
        ]);
    }

    model
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_json_model_deserialization() {
        let json_content = r#"
        {
            "version": "1.0",
            "metadata": {
                "name": "test_model",
                "description": "A test model"
            },
            "inputs": [
                {
                    "name": "input",
                    "shape": [1, 3],
                    "dtype": "f32"
                }
            ],
            "outputs": [
                {
                    "name": "output",
                    "shape": [1, 2],
                    "dtype": "f32"
                }
            ],
            "initializers": [
                {
                    "name": "weight",
                    "shape": [3, 2],
                    "data": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
                }
            ],
            "nodes": [
                {
                    "name": "matmul",
                    "op_type": "MatMul",
                    "inputs": ["input", "weight"],
                    "outputs": ["output"]
                }
            ]
        }
        "#;

        let model: JsonModel = serde_json::from_str(json_content).unwrap();
        
        assert_eq!(model.version, "1.0");
        assert_eq!(model.metadata.name, "test_model");
        assert_eq!(model.inputs.len(), 1);
        assert_eq!(model.outputs.len(), 1);
        assert_eq!(model.initializers.len(), 1);
        assert_eq!(model.nodes.len(), 1);
    }

    #[test]
    fn test_json_loader_load() {
        let json_content = r#"
        {
            "version": "1.0",
            "inputs": [{"name": "input", "shape": [1, 2], "dtype": "f32"}],
            "outputs": [{"name": "output", "shape": [1, 2], "dtype": "f32"}],
            "initializers": [],
            "nodes": [
                {
                    "name": "relu",
                    "op_type": "ReLU",
                    "inputs": ["input"],
                    "outputs": ["output"]
                }
            ]
        }
        "#;

        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "{}", json_content).unwrap();
        temp_file.flush().unwrap();

        let loader = JsonLoader::new();
        let graph = loader.load(temp_file.path()).unwrap();
        
        let stats = graph.execution_stats();
        assert_eq!(stats.node_count, 1);
        assert_eq!(stats.input_count, 1);
        assert_eq!(stats.output_count, 1);
    }

    #[test]
    fn test_invalid_version() {
        let json_content = r#"
        {
            "version": "2.0",
            "inputs": [{"name": "input", "shape": [1, 2], "dtype": "f32"}],
            "outputs": [{"name": "output", "shape": [1, 2], "dtype": "f32"}],
            "nodes": []
        }
        "#;

        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "{}", json_content).unwrap();
        temp_file.flush().unwrap();

        let loader = JsonLoader::new();
        let result = loader.load(temp_file.path());
        
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unsupported JSON model version"));
    }

    #[test]
    fn test_validation_errors() {
        let loader = JsonLoader::new();
        
        // Test empty inputs
        let model = JsonModel {
            version: "1.0".to_string(),
            metadata: ModelMetadata::default(),
            inputs: vec![],
            outputs: vec![TensorSpec {
                name: "output".to_string(),
                shape: vec![1],
                description: String::new(),
                dtype: "f32".to_string(),
            }],
            initializers: vec![],
            nodes: vec![],
        };
        
        assert!(loader.validate_model_structure(&model).is_err());
    }

    #[test]
    fn test_create_simple_model() {
        let model = create_simple_model(3, 2, Some(5));
        
        assert_eq!(model.inputs.len(), 1);
        assert_eq!(model.outputs.len(), 1);
        assert_eq!(model.inputs[0].shape, vec![1, 3]);
        assert_eq!(model.outputs[0].shape, vec![1, 2]);
        assert_eq!(model.initializers.len(), 4); // 2 weights + 2 biases
        assert_eq!(model.nodes.len(), 5); // fc1, add1, relu1, fc2, add2
    }

    #[test]
    fn test_loader_can_load() {
        let loader = JsonLoader::new();
        
        // Should detect .json files
        assert!(loader.can_load(Path::new("model.json")));
        assert!(!loader.can_load(Path::new("model.onnx")));
        
        // Test with actual JSON content
        let json_content = r#"{"version": "1.0", "inputs": [], "outputs": [], "nodes": []}"#;
        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "{}", json_content).unwrap();
        temp_file.flush().unwrap();
        
        assert!(loader.can_load(temp_file.path()));
    }
}