//! orys - Lightweight Edge AI Inference Runtime
//!
//! orys is a high-performance, memory-safe inference runtime for neural networks,
//! written in Rust. It supports a subset of ONNX operators and provides both
//! JSON and ONNX model loading capabilities.
//!
//! # Features
//!
//! - **Core Operators**: MatMul, Add (with broadcasting), ReLU, Sigmoid
//! - **Multiple Formats**: JSON (human-readable) and ONNX (standard ML format)
//! - **Memory Safe**: Rust's ownership system prevents common ML runtime bugs
//! - **Edge Optimized**: Lightweight runtime suitable for embedded and edge devices
//! - **Python Bindings**: Available through PyO3 (see python-bindings crate)
//!
//! # Quick Start
//!
//! ```rust
//! use orys::{run_model, Tensor};
//! use std::collections::HashMap;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Load and run a model in one call
//! let mut inputs = HashMap::new();
//! inputs.insert("input".to_string(), Tensor::ones(vec![1, 784]));
//!
//! let outputs = run_model("models/classifier.json", inputs)?;
//! let prediction = &outputs["output"];
//!
//! println!("Prediction: {:?}", prediction.data());
//! # Ok(())
//! # }
//! ```
//!
//! # Advanced Usage
//!
//! For more control over the inference process:
//!
//! ```rust
//! use orys::{loader, graph::ComputeGraph, Tensor};
//! use std::collections::HashMap;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Load model once, run multiple times
//! let mut graph = loader::load_model("model.json")?;
//!
//! // Run inference multiple times
//! for i in 0..10 {
//!     let mut inputs = HashMap::new();
//!     inputs.insert("input".to_string(), Tensor::from_vec(vec![i as f32; 784]));
//!     
//!     let outputs = graph.execute(inputs)?;
//!     println!("Batch {}: {:?}", i, outputs["output"].data());
//! }
//! # Ok(())
//! # }
//! ```

// Core modules
pub mod errors;
pub mod tensor;
pub mod ops;
pub mod graph;
pub mod loader;

// Re-export commonly used types for convenience
pub use errors::{Result, OrysError};
pub use tensor::Tensor;
pub use graph::{ComputeGraph, GraphNode, Initializer, ExecutionStats};
pub use ops::{Operator, MatMul, Add, ReLU, Sigmoid, create_operator};

use std::collections::HashMap;
use std::path::Path;

/// Run inference on a model with given inputs
///
/// This is the simplest way to use orys - provide a model file and inputs,
/// get outputs back. The model format is automatically detected.
///
/// # Arguments
/// * `model_path` - Path to the model file (.json or .onnx)
/// * `inputs` - Map of input tensor names to tensor data
///
/// # Returns
/// * `Result<HashMap<String, Tensor>>` - Map of output tensor names to results
///
/// # Errors
/// * Model loading errors (file not found, invalid format, etc.)
/// * Input validation errors (missing inputs, shape mismatches)
/// * Inference execution errors (operator failures, memory issues)
///
/// # Examples
///
/// ```rust
/// use orys::{run_model, Tensor};
/// use std::collections::HashMap;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Prepare inputs
/// let mut inputs = HashMap::new();
/// inputs.insert("input".to_string(), Tensor::from_vec(vec![1.0, 2.0, 3.0]));
///
/// // Run inference
/// let outputs = run_model("examples/simple.json", inputs)?;
///
/// // Access results
/// let result = &outputs["output"];
/// println!("Output shape: {:?}", result.shape());
/// println!("Output data: {:?}", result.data());
/// # Ok(())
/// # }
/// ```
pub fn run_model<P: AsRef<Path>>(
    model_path: P,
    inputs: HashMap<String, Tensor>,
) -> Result<HashMap<String, Tensor>> {
    let mut graph = loader::load_model(model_path)?;
    graph.execute(inputs)
}

/// Load a model from file and return the computation graph
///
/// This provides more control than `run_model` by allowing you to:
/// - Inspect the model structure before running inference
/// - Run inference multiple times with different inputs
/// - Access execution statistics and debugging information
///
/// # Arguments
/// * `model_path` - Path to the model file
///
/// # Returns
/// * `Result<ComputeGraph>` - Loaded and validated computation graph
///
/// # Examples
///
/// ```rust
/// use orys::{load_model, Tensor};
/// use std::collections::HashMap;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Load model once
/// let mut graph = load_model("model.json")?;
///
/// // Inspect model structure
/// let stats = graph.execution_stats();
/// println!("Model has {} nodes", stats.node_count);
/// println!("Inputs: {:?}", graph.inputs());
/// println!("Outputs: {:?}", graph.outputs());
///
/// // Run inference
/// let mut inputs = HashMap::new();
/// inputs.insert("input".to_string(), Tensor::ones(vec![1, 10]));
/// let outputs = graph.execute(inputs)?;
/// # Ok(())
/// # }
/// ```
pub fn load_model<P: AsRef<Path>>(model_path: P) -> Result<ComputeGraph> {
    loader::load_model(model_path)
}

/// Validate a model file without loading it
///
/// Performs format detection and basic validation without building
/// the full computation graph. Useful for checking model files
/// quickly or in batch processing scenarios.
///
/// # Arguments
/// * `model_path` - Path to the model file
///
/// # Returns
/// * `Result<loader::ModelFormat>` - Detected format if valid
///
/// # Examples
///
/// ```rust
/// use orys::validate_model;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// match validate_model("model.json") {
///     Ok(format) => println!("Valid {:?} model", format),
///     Err(e) => println!("Invalid model: {}", e),
/// }
/// # Ok(())
/// # }
/// ```
pub fn validate_model<P: AsRef<Path>>(model_path: P) -> Result<loader::ModelFormat> {
    loader::validate_model_file(model_path)
}

/// Get information about a model file
///
/// Extracts metadata about the model including format, size, and structure
/// information without fully loading the model. More efficient than full
/// loading for inspection purposes.
///
/// # Arguments
/// * `model_path` - Path to the model file
///
/// # Returns
/// * `Result<loader::ModelInfo>` - Model metadata
///
/// # Examples
///
/// ```rust
/// use orys::inspect_model;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let info = inspect_model("model.json")?;
/// 
/// println!("Format: {:?}", info.format);
/// println!("File size: {} bytes", info.file_size);
/// println!("Nodes: {}", info.node_count);
/// println!("Inputs: {:?}", info.input_names);
/// println!("Outputs: {:?}", info.output_names);
/// # Ok(())
/// # }
/// ```
pub fn inspect_model<P: AsRef<Path>>(model_path: P) -> Result<loader::ModelInfo> {
    loader::inspect_model(model_path)
}

/// Get list of available model loaders
///
/// Returns information about all compiled-in model format loaders.
/// The available loaders depend on which features are enabled.
///
/// # Returns
/// * `Vec<&'static str>` - Available loader format names
///
/// # Examples
///
/// ```rust
/// use orys::available_loaders;
///
/// let loaders = available_loaders();
/// for loader in loaders {
///     println!("Available loader: {}", loader);
/// }
/// ```
pub fn available_loaders() -> Vec<&'static str> {
    loader::available_loader_names()
}

/// Create a tensor from a flat vector of data
///
/// Convenience function for creating tensors with automatic shape inference.
/// The tensor will have a 1D shape matching the vector length.
///
/// # Arguments
/// * `data` - Vector of f32 values
///
/// # Returns
/// * `Tensor` - 1D tensor containing the data
///
/// # Examples
///
/// ```rust
/// use orys::tensor_from_vec;
///
/// let tensor = tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// assert_eq!(tensor.shape(), &[4]);
/// assert_eq!(tensor.data(), &[1.0, 2.0, 3.0, 4.0]);
/// ```
pub fn tensor_from_vec(data: Vec<f32>) -> Tensor {
    Tensor::from_vec(data)
}

/// Create a tensor with specified shape and data
///
/// # Arguments
/// * `shape` - Desired tensor dimensions
/// * `data` - Flat vector of data in row-major order
///
/// # Returns
/// * `Result<Tensor>` - Tensor with specified shape, or error if data size doesn't match
///
/// # Examples
///
/// ```rust
/// use orys::tensor_with_shape;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let tensor = tensor_with_shape(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
/// assert_eq!(tensor.shape(), &[2, 3]);
/// assert_eq!(tensor.size(), 6);
/// # Ok(())
/// # }
/// ```
pub fn tensor_with_shape(shape: Vec<usize>, data: Vec<f32>) -> Result<Tensor> {
    Tensor::new(shape, data)
}

/// Create a tensor filled with zeros
///
/// # Arguments
/// * `shape` - Desired tensor dimensions
///
/// # Returns
/// * `Tensor` - Zero-filled tensor with specified shape
///
/// # Examples
///
/// ```rust
/// use orys::zeros;
///
/// let tensor = zeros(vec![2, 3]);
/// assert_eq!(tensor.shape(), &[2, 3]);
/// assert_eq!(tensor.data(), &[0.0; 6]);
/// ```
pub fn zeros(shape: Vec<usize>) -> Tensor {
    Tensor::zeros(shape)
}

/// Create a tensor filled with ones
///
/// # Arguments
/// * `shape` - Desired tensor dimensions
///
/// # Returns
/// * `Tensor` - One-filled tensor with specified shape
///
/// # Examples
///
/// ```rust
/// use orys::ones;
///
/// let tensor = ones(vec![2, 2]);
/// assert_eq!(tensor.data(), &[1.0, 1.0, 1.0, 1.0]);
/// ```
pub fn ones(shape: Vec<usize>) -> Tensor {
    Tensor::ones(shape)
}

/// Create a scalar tensor (0-dimensional)
///
/// # Arguments
/// * `value` - Scalar value
///
/// # Returns
/// * `Tensor` - Scalar tensor containing the value
///
/// # Examples
///
/// ```rust
/// use orys::scalar;
///
/// let tensor = scalar(42.0);
/// assert_eq!(tensor.shape(), &[]);
/// assert_eq!(tensor.data(), &[42.0]);
/// assert!(tensor.is_scalar());
/// ```
pub fn scalar(value: f32) -> Tensor {
    Tensor::scalar(value)
}

/// Library version information
///
/// Returns the current version of the orys library.
/// Useful for debugging and compatibility checking.
///
/// # Returns
/// * `&'static str` - Version string in semver format
///
/// # Examples
///
/// ```rust
/// use orys::version;
///
/// println!("orys version: {}", version());
/// ```
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Library build information
///
/// Returns detailed information about how the library was built,
/// including enabled features and build configuration.
///
/// # Returns
/// * `BuildInfo` - Structured build information
///
/// # Examples
///
/// ```rust
/// use orys::build_info;
///
/// let info = build_info();
/// println!("Version: {}", info.version);
/// println!("Features: {:?}", info.features);
/// ```
pub fn build_info() -> BuildInfo {
    BuildInfo {
        version: env!("CARGO_PKG_VERSION"),
        features: enabled_features(),
        target: "unknown", // Would need build script to get actual target
        profile: if cfg!(debug_assertions) { "debug" } else { "release" },
    }
}

/// Information about how the library was built
#[derive(Debug, Clone)]
pub struct BuildInfo {
    /// Library version
    pub version: &'static str,
    /// Enabled Cargo features
    pub features: Vec<&'static str>,
    /// Target architecture
    pub target: &'static str,
    /// Build profile (debug/release)
    pub profile: &'static str,
}

/// Get list of enabled Cargo features
fn enabled_features() -> Vec<&'static str> {
    let mut features = vec!["json"]; // JSON is always enabled
    
    #[cfg(feature = "onnx")]
    features.push("onnx");
    
    features
}

/// Prelude module for convenient imports
///
/// The prelude includes the most commonly used types and functions
/// from the orys library. Import this module to get started quickly.
///
/// # Examples
///
/// ```rust
/// use orys::prelude::*;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Now you have access to all common orys types
/// let tensor = ones(vec![2, 2]);
/// let mut inputs = HashMap::new();
/// inputs.insert("input".to_string(), tensor);
/// // let outputs = run_model("model.json", inputs)?;
/// # Ok(())
/// # }
/// ```
pub mod prelude {
    pub use crate::{
        run_model, load_model, validate_model, inspect_model,
        tensor_from_vec, tensor_with_shape, zeros, ones, scalar,
        Tensor, ComputeGraph, Result, OrysError,
    };
    pub use std::collections::HashMap;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_info() {
        let version = version();
        assert!(!version.is_empty());
        
        let build_info = build_info();
        assert_eq!(build_info.version, version);
        assert!(!build_info.features.is_empty());
        assert!(build_info.features.contains(&"json"));
    }

    #[test]
    fn test_tensor_convenience_functions() {
        // Test tensor_from_vec
        let tensor = tensor_from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(tensor.shape(), &[3]);
        assert_eq!(tensor.data(), &[1.0, 2.0, 3.0]);

        // Test tensor_with_shape
        let tensor = tensor_with_shape(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.size(), 4);

        // Test zeros
        let tensor = zeros(vec![2, 3]);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.data(), &[0.0; 6]);

        // Test ones
        let tensor = ones(vec![2, 2]);
        assert_eq!(tensor.data(), &[1.0, 1.0, 1.0, 1.0]);

        // Test scalar
        let tensor = scalar(42.0);
        assert!(tensor.is_scalar());
        assert_eq!(tensor.data(), &[42.0]);
    }

    #[test]
    fn test_tensor_shape_mismatch() {
        let result = tensor_with_shape(vec![2, 2], vec![1.0, 2.0, 3.0]); // Wrong size
        assert!(result.is_err());
    }

    #[test]
    fn test_enabled_features() {
        let features = enabled_features();
        assert!(features.contains(&"json"));
        
        #[cfg(feature = "onnx")]
        assert!(features.contains(&"onnx"));
        
        #[cfg(not(feature = "onnx"))]
        assert!(!features.contains(&"onnx"));
    }

    #[test]
    fn test_available_loaders() {
        let loaders = available_loaders();
        assert!(!loaders.is_empty());
        
        // JSON loader should always be available
        assert!(loaders.contains(&"JSON"));
    }

    #[test]
    fn test_prelude_imports() {
        use crate::prelude::*;
        
        // Test that prelude provides access to key functions
        let _tensor = ones(vec![2, 2]);
        let _inputs: HashMap<String, Tensor> = HashMap::new();
        
        // These should compile without additional imports
        let _result: Result<Tensor> = Ok(scalar(1.0));
    }
}