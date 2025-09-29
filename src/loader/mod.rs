//! Model loading infrastructure for multiple formats
//!
//! This module provides a unified interface for loading neural network models
//! from different file formats (JSON, ONNX) into executable computation graphs.

use crate::errors::{Result, ZekeError};
use crate::graph::ComputeGraph;
use std::path::Path;

pub mod json;
// Note: ONNX module will be added in future when ONNX support is implemented

/// Supported model file formats
#[derive(Debug, Clone, PartialEq)]
pub enum ModelFormat {
    /// JSON-based model format (human-readable, for testing/debugging)
    Json,
    /// ONNX protobuf format (standard ML model interchange)
    #[cfg(feature = "onnx")]
    Onnx,
}

/// Trait for model loaders that can parse different file formats
///
/// This trait provides a common interface for loading models from various
/// formats into our internal computation graph representation.
pub trait ModelLoader {
    /// Load a model from the given file path
    ///
    /// # Arguments
    /// * `path` - Path to the model file
    ///
    /// # Returns
    /// * `Result<ComputeGraph>` - Loaded and validated computation graph
    ///
    /// # Errors
    /// * File I/O errors
    /// * Format parsing errors  
    /// * Model validation errors
    fn load(&self, path: &Path) -> Result<ComputeGraph>;

    /// Get the format name for this loader
    fn format_name(&self) -> &'static str;

    /// Check if this loader can handle the given file
    fn can_load(&self, path: &Path) -> bool;
}

/// Detect the model format from file extension and/or content
///
/// # Arguments
/// * `path` - Path to the model file
///
/// # Returns
/// * `Result<ModelFormat>` - Detected format or error if unrecognized
///
/// # Examples
/// ```rust
/// use zeke::loader::detect_format;
/// 
/// let format = detect_format("model.json")?;
/// assert_eq!(format, ModelFormat::Json);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn detect_format<P: AsRef<Path>>(path: P) -> Result<ModelFormat> {
    let path = path.as_ref();
    
    // Check if file exists first
    if !path.exists() {
        return Err(ZekeError::UnrecognizedFormat {
            filename: path.display().to_string(),
        });
    }
    
    // Try file extension
    if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
        match extension.to_lowercase().as_str() {
            "json" => return Ok(ModelFormat::Json),
            #[cfg(feature = "onnx")]
            "onnx" => return Ok(ModelFormat::Onnx),
            _ => {}
        }
    }

    // If extension doesn't help, try to read file header
    match std::fs::read(path) {
        Ok(bytes) => {
            // Check for JSON format (starts with '{' or '[')
            if !bytes.is_empty() && (bytes[0] == b'{' || bytes[0] == b'[') {
                return Ok(ModelFormat::Json);
            }
            
            #[cfg(feature = "onnx")]
            {
                // Check for ONNX protobuf magic bytes
                // ONNX files typically start with protobuf field tags
                if bytes.len() >= 2 && bytes[0] == 0x08 {
                    return Ok(ModelFormat::Onnx);
                }
            }
        }
        Err(_) => {
            // File doesn't exist or can't be read
            return Err(ZekeError::UnrecognizedFormat {
                filename: path.display().to_string(),
            });
        }
    }

    Err(ZekeError::UnrecognizedFormat {
        filename: path.display().to_string(),
    })
}

/// Load a model from file, automatically detecting the format
///
/// This is the main entry point for loading models. It automatically
/// detects the file format and uses the appropriate loader.
///
/// # Arguments
/// * `path` - Path to the model file
///
/// # Returns
/// * `Result<ComputeGraph>` - Loaded and validated computation graph
///
/// # Examples
/// ```rust
/// use zeke::loader::load_model;
/// 
/// let graph = load_model("models/classifier.json")?;
/// println!("Loaded model with {} nodes", graph.execution_stats().node_count);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn load_model<P: AsRef<Path>>(path: P) -> Result<ComputeGraph> {
    let path = path.as_ref();
    let format = detect_format(&path)?;
    
    match format {
        ModelFormat::Json => {
            let loader = json::JsonLoader::new();
            loader.load(path)
        }
        #[cfg(feature = "onnx")]
        ModelFormat::Onnx => {
            // For now, return an error since ONNX is not implemented yet
            Err(ZekeError::UnsupportedOnnxFeatures {
                features: vec!["ONNX support not yet implemented".to_string()],
            })
        }
    }
}

/// Get a list of available loader format names
///
/// Returns format names for all compiled-in formats based on feature flags.
pub fn available_loader_names() -> Vec<&'static str> {
    let mut loaders = vec!["JSON"];
    
    #[cfg(feature = "onnx")]
    loaders.push("ONNX");
    
    loaders
}

/// Load a model using a specific format, bypassing auto-detection
///
/// Useful when you know the format and want to skip detection overhead,
/// or when the file extension doesn't match the actual format.
///
/// # Arguments
/// * `path` - Path to the model file
/// * `format` - Explicit format to use
///
/// # Returns
/// * `Result<ComputeGraph>` - Loaded and validated computation graph
pub fn load_model_with_format<P: AsRef<Path>>(
    path: P,
    format: ModelFormat,
) -> Result<ComputeGraph> {
    let path = path.as_ref();
    match format {
        ModelFormat::Json => {
            let loader = json::JsonLoader::new();
            loader.load(path)
        }
        #[cfg(feature = "onnx")]
        ModelFormat::Onnx => {
            Err(ZekeError::UnsupportedOnnxFeatures {
                features: vec!["ONNX support not yet implemented".to_string()],
            })
        }
    }
}

/// Validate that a model file can be loaded without actually loading it
///
/// Performs format detection and basic validation without building
/// the full computation graph. Useful for checking model files.
///
/// # Arguments
/// * `path` - Path to the model file
///
/// # Returns
/// * `Result<ModelFormat>` - Detected format if valid, error otherwise
pub fn validate_model_file<P: AsRef<Path>>(path: P) -> Result<ModelFormat> {
    let path = path.as_ref();
    
    // Check if file exists and is readable
    if !path.exists() {
        return Err(ZekeError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Model file not found: {}", path.display()),
        )));
    }
    
    if !path.is_file() {
        return Err(ZekeError::invalid_model(format!(
            "Path is not a file: {}",
            path.display()
        )));
    }
    
    // Detect format
    let format = detect_format(path)?;
    
    // Perform format-specific validation without full loading
    match format {
        ModelFormat::Json => {
            json::JsonLoader::validate_file(path)?;
        }
        #[cfg(feature = "onnx")]
        ModelFormat::Onnx => {
            return Err(ZekeError::UnsupportedOnnxFeatures {
                features: vec!["ONNX validation not yet implemented".to_string()],
            });
        }
    }
    
    Ok(format)
}

/// Model loading statistics and metadata
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Detected or specified format
    pub format: ModelFormat,
    /// File size in bytes
    pub file_size: u64,
    /// Number of nodes in the computation graph
    pub node_count: usize,
    /// Number of graph inputs
    pub input_count: usize,
    /// Number of graph outputs
    pub output_count: usize,
    /// Number of initializer tensors (weights, biases)
    pub initializer_count: usize,
    /// List of input tensor names
    pub input_names: Vec<String>,
    /// List of output tensor names
    pub output_names: Vec<String>,
}

/// Get information about a model file without fully loading it
///
/// Provides metadata about the model including format, size, and structure
/// information. More efficient than full loading for inspection purposes.
///
/// # Arguments
/// * `path` - Path to the model file
///
/// # Returns
/// * `Result<ModelInfo>` - Model metadata or error
pub fn inspect_model<P: AsRef<Path>>(path: P) -> Result<ModelInfo> {
    let path = path.as_ref();
    
    // Get file metadata
    let metadata = std::fs::metadata(path).map_err(ZekeError::Io)?;
    let file_size = metadata.len();
    
    // Detect format
    let format = detect_format(path)?;
    
    // Get format-specific information
    let (node_count, input_count, output_count, initializer_count, input_names, output_names) = 
        match format {
            ModelFormat::Json => {
                json::JsonLoader::inspect_file(path)?
            }
            #[cfg(feature = "onnx")]
            ModelFormat::Onnx => {
                return Err(ZekeError::UnsupportedOnnxFeatures {
                    features: vec!["ONNX inspection not yet implemented".to_string()],
                });
            }
        };
    
    Ok(ModelInfo {
        format,
        file_size,
        node_count,
        input_count,
        output_count,
        initializer_count,
        input_names,
        output_names,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_format_detection_by_extension() {
        assert_eq!(detect_format("model.json").unwrap(), ModelFormat::Json);
        
        #[cfg(feature = "onnx")]
        assert_eq!(detect_format("model.onnx").unwrap(), ModelFormat::Onnx);
        
        assert!(detect_format("model.txt").is_err());
    }

    #[test]
    fn test_format_detection_by_content() {
        // Test JSON content detection
        let mut json_file = NamedTempFile::new().unwrap();
        writeln!(json_file, r#"{{"nodes": []}}"#).unwrap();
        json_file.flush().unwrap();
        
        assert_eq!(detect_format(json_file.path()).unwrap(), ModelFormat::Json);
    }

    #[test]
    fn test_unrecognized_format() {
        let mut unknown_file = NamedTempFile::new().unwrap();
        writeln!(unknown_file, "This is not a recognized format").unwrap();
        unknown_file.flush().unwrap();
        
        assert!(detect_format(unknown_file.path()).is_err());
    }

    #[test]
    fn test_nonexistent_file() {
        assert!(detect_format("nonexistent.json").is_err());
    }

    #[test]
    fn test_available_loader_names() {
        let loaders = available_loader_names();
        
        // JSON loader should always be available
        assert!(loaders.len() >= 1);
        assert!(loaders.contains(&"JSON"));
        
        #[cfg(feature = "onnx")]
        assert!(loaders.contains(&"ONNX"));
    }

    #[test]
    fn test_model_info_fields() {
        // Test that ModelInfo has all expected fields
        let info = ModelInfo {
            format: ModelFormat::Json,
            file_size: 1024,
            node_count: 5,
            input_count: 1,
            output_count: 1,
            initializer_count: 3,
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
        };
        
        assert_eq!(info.format, ModelFormat::Json);
        assert_eq!(info.file_size, 1024);
        assert_eq!(info.node_count, 5);
        assert_eq!(info.input_names.len(), 1);
        assert_eq!(info.output_names.len(), 1);
    }
}