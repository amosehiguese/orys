//! Error types for the zeke inference runtime
//! 
//! This module defines all error types used thoughout the Zeke runtime.
//! We use `thiserror` for ergonomic error handling with automatic trait implementation
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ZekeError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Invalid tensor operation: {message}")]
    InvalidTensorOperation { message: String },

    #[error("Tensor size mismatch: shape {shape:?} requires {expected} elements, got {actual}")]
    TensorSizeMismatch {
        shape: Vec<usize>,
        expected: usize,
        actual: usize,
    },

    #[error("Node '{node_name}' not found in graph")]
    NodeNotFound { node_name: String },

    #[error("Circular dependency detected in computation graph")]
    CircularDependency { nodes: Vec<String> },

    #[error("Graph has no input nodes - at least one input is required")]
    NoInputNodes,

    #[error("Graph has no output nodes - at lease one output is required")]
    NoOutputNodes,

    #[error("Unsupported operator: '{op_type}' (supported: MatMul, Add, ReLU, Sigmoid)")]
    UnsupportedOperator { op_type: String },

    /// Operator received wrong number of inputs
    #[error("Operator '{op_type}' expects {expected} inputs, got {actual}")]
    InvalidInputCount {
        op_type: String,
        expected: usize,
        actual: usize,
    },

    #[error("Operator '{op_type}' validation failed: {message}")]
    OperatorValidation { op_type: String, message: String },

    #[error("Unrecognized model format for file: {filename}")]
    UnrecognizedFormat { filename: String },

    #[error("Invalid model file: {message}")]
    InvalidModel { message: String },

    #[error("Model missing required component: {component}")]
    MissingModelComponent { component: String },

    #[error("Inference execution failed: {message}")]
    InferenceError { message: String },

    #[error("Missing required input: '{input_name}'")]
    MissingInput { input_name: String },

    #[error("Input '{input_name}' shape mismatch: expected {expected:?}, got {actual:?}")]
    InputShapeMismatch {
        input_name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[cfg(feature = "onnx")]
    /// Onnx protobuf parsing error 
    #[error("ONNX protobuf error: {0}")]
    Protobuf(#[from] prost::DecodeError),

    #[cfg(feature = "onnx")]
    /// Unsupported ONNX version
    #[error("Unsupported ONNX version: {version} (supported: 1.0+)")]
    UnsupportedOnnxVersion { version: String },

    #[cfg(feature = "onnx")]
    /// ONNX model uses unsupported features
    #[error("ONNX model uses unsupported features: {features:?}")]
    UnsupportedOnnxFeatures { features: Vec<String> },
}

pub type Result<T> = std::result::Result<T, ZekeError>;

impl ZekeError {
    /// Create a new InvalidTensorOperation error
    pub fn invalid_tensor_op<S: Into<String>>(message: S) -> Self {
        ZekeError::InvalidTensorOperation { message: message.into() }
    }

    /// Create a new OperatorValidation error
    pub fn operator_validation<S1: Into<String>, S2: Into<String>>(
        op_type: S1,
        message: S2,
    ) -> Self {
        ZekeError::OperatorValidation {
            op_type: op_type.into(),
            message: message.into(),
        }
    }

    /// Create a new InvalidModel error
    pub fn invalid_model<S: Into<String>>(message: S) -> Self {
        ZekeError::InvalidModel {
            message: message.into(),
        }
    }

    /// Create a new InferenceError
    pub fn inference_error<S: Into<String>>(message: S) -> Self {
        ZekeError::InferenceError {
            message: message.into(),
        }
    }

    /// Check if this error is related to tensor operations
    pub fn is_tensor_error(&self) -> bool {
        matches!(
            self,
            ZekeError::ShapeMismatch { .. }
                | ZekeError::InvalidTensorOperation { .. }
                | ZekeError::TensorSizeMismatch {  .. }
        )
    }

    /// Check if this error is related to graph structure
    pub fn is_graph_error(&self) -> bool {
        matches!(
            self,
            ZekeError::NodeNotFound { .. }
                | ZekeError::CircularDependency { .. }
                | ZekeError::NoInputNodes
                | ZekeError::NoOutputNodes
        )
    }

    /// Check if this error is related to operator issues
    pub fn is_operator_error(&self) -> bool {
        matches!(
            self,
            ZekeError::UnsupportedOperator { .. }
                | ZekeError::InvalidInputCount { .. }
                | ZekeError::OperatorValidation { .. }
        )
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let error = ZekeError::ShapeMismatch { 
            expected: vec![2, 3],
            actual: vec![3, 2],
        };

        let display = format!("{}", error);
        assert!(display.contains("Shape mismatch"));
        assert!(display.contains("[2, 3]"));
        assert!(display.contains("[3, 2]"));
    }

    #[test]
    fn test_error_categorization() {
        let tensor_error = ZekeError::ShapeMismatch {
            expected: vec![1],
            actual: vec![2],
        };
        assert!(tensor_error.is_tensor_error());
        assert!(!tensor_error.is_graph_error());

        let graph_error = ZekeError::NodeNotFound {
            node_name: "test".to_string(),
        };
        assert!(graph_error.is_graph_error());
        assert!(!graph_error.is_tensor_error());
    }

    #[test]
    fn test_convenience_constructors() {
        let error = ZekeError::invalid_tensor_op("Test message");
        assert!(matches!(error, ZekeError::InvalidTensorOperation { .. }));

        let error = ZekeError::operator_validation("MatMul", "Invalid dimensions");
        assert!(matches!(error, ZekeError::OperatorValidation { .. }));
    }
}