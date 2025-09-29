//! Python bindings for the Zeke inference runtime
//!
//! This module provides Python bindings using PyO3, allowing Python users
//! to load and run neural network models using the Zeke runtime.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use std::collections::HashMap;

// Re-export core Zeke types with Python wrappers
use zeke::{Tensor as ZekeTensor, ComputeGraph as ZekeGraph, run_model as zeke_run_model};

/// Python wrapper for Zeke Tensor
///
/// Provides a Python-friendly interface to the Rust Tensor type,
/// with automatic conversion to/from NumPy arrays.
#[pyclass(name = "Tensor")]
#[derive(Clone)]
pub struct PyTensor {
    inner: ZekeTensor,
}

#[pymethods]
impl PyTensor {
    /// Create a new tensor from shape and data
    ///
    /// Args:
    ///     shape: List of dimension sizes
    ///     data: Flat list of values in row-major order
    ///
    /// Returns:
    ///     Tensor: New tensor instance
    ///
    /// Example:
    ///     >>> tensor = zeke.Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    #[new]
    fn new(shape: Vec<usize>, data: Vec<f32>) -> PyResult<Self> {
        let tensor = ZekeTensor::new(shape, data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(PyTensor { inner: tensor })
    }

    /// Create a tensor filled with zeros
    ///
    /// Args:
    ///     shape: List of dimension sizes
    ///
    /// Returns:
    ///     Tensor: Zero-filled tensor
    ///
    /// Example:
    ///     >>> tensor = zeke.Tensor.zeros([2, 3])
    #[staticmethod]
    fn zeros(shape: Vec<usize>) -> Self {
        PyTensor {
            inner: ZekeTensor::zeros(shape),
        }
    }

    /// Create a tensor filled with ones
    ///
    /// Args:
    ///     shape: List of dimension sizes
    ///
    /// Returns:
    ///     Tensor: One-filled tensor
    ///
    /// Example:
    ///     >>> tensor = zeke.Tensor.ones([2, 3])
    #[staticmethod]
    fn ones(shape: Vec<usize>) -> Self {
        PyTensor {
            inner: ZekeTensor::ones(shape),
        }
    }

    /// Create a tensor from a NumPy array
    ///
    /// Args:
    ///     array: NumPy array to convert
    ///
    /// Returns:
    ///     Tensor: New tensor with same shape and data
    ///
    /// Example:
    ///     >>> import numpy as np
    ///     >>> arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    ///     >>> tensor = zeke.Tensor.from_numpy(arr)
    #[staticmethod]
    fn from_numpy(array: PyReadonlyArray1<f32>) -> PyResult<Self> {
        let data = array.as_slice()?.to_vec();
        let shape = vec![data.len()];
        
        let tensor = ZekeTensor::new(shape, data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(PyTensor { inner: tensor })
    }

    /// Convert tensor to NumPy array
    ///
    /// Returns:
    ///     numpy.ndarray: NumPy array with same data
    ///
    /// Example:
    ///     >>> tensor = zeke.Tensor.ones([2, 3])
    ///     >>> array = tensor.to_numpy()
    fn to_numpy<'py>(&self, py: Python<'py>) -> &'py PyArray1<f32> {
        self.inner.data().to_pyarray(py)
    }

    /// Get tensor shape
    ///
    /// Returns:
    ///     List[int]: Tensor dimensions
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    /// Get tensor data as a list
    ///
    /// Returns:
    ///     List[float]: Flattened tensor data
    #[getter]
    fn data(&self) -> Vec<f32> {
        self.inner.data().to_vec()
    }

    /// Get number of dimensions
    ///
    /// Returns:
    ///     int: Number of dimensions
    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    /// Get total number of elements
    ///
    /// Returns:
    ///     int: Total element count
    #[getter]
    fn size(&self) -> usize {
        self.inner.size()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("Tensor(shape={:?}, size={})", self.inner.shape(), self.inner.size())
    }

    /// Get tensor item (for 1D tensors)
    fn __getitem__(&self, index: usize) -> PyResult<f32> {
        if self.inner.ndim() != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Item access only supported for 1D tensors",
            ));
        }
        
        if index >= self.inner.size() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Index out of bounds",
            ));
        }
        
        Ok(self.inner.data()[index])
    }

    /// Get tensor length (for 1D tensors)
    fn __len__(&self) -> PyResult<usize> {
        if self.inner.ndim() != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Length only defined for 1D tensors",
            ));
        }
        Ok(self.inner.size())
    }
}

/// Python wrapper for ComputeGraph
///
/// Represents a loaded neural network model that can be executed
/// multiple times with different inputs.
#[pyclass(name = "ComputeGraph")]
pub struct PyComputeGraph {
    inner: ZekeGraph,
}

#[pymethods]
impl PyComputeGraph {
    /// Execute the model with given inputs
    ///
    /// Args:
    ///     inputs: Dictionary mapping input names to tensors or NumPy arrays
    ///
    /// Returns:
    ///     Dict[str, Tensor]: Dictionary mapping output names to result tensors
    ///
    /// Example:
    ///     >>> inputs = {"input": zeke.Tensor.ones([1, 784])}
    ///     >>> outputs = graph.execute(inputs)
    ///     >>> prediction = outputs["output"]
    fn execute(&mut self, inputs: &PyDict) -> PyResult<PyObject> {
        let py = inputs.py();
        
        // Convert Python inputs to Rust HashMap
        let mut rust_inputs = HashMap::new();
        for (key, value) in inputs {
            let name: String = key.extract()?;
            
            // Handle both Tensor objects and NumPy arrays
            let tensor = if let Ok(py_tensor) = value.extract::<PyRef<PyTensor>>() {
                py_tensor.inner.clone()
            } else if let Ok(array) = value.extract::<PyReadonlyArray1<f32>>() {
                let data = array.as_slice()?.to_vec();
                let shape = vec![data.len()];
                ZekeTensor::new(shape, data)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Input values must be Tensor objects or NumPy arrays",
                ));
            };
            
            rust_inputs.insert(name, tensor);
        }

        // Execute the model
        let rust_outputs = self.inner.execute(rust_inputs)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

        // Convert outputs back to Python dict
        let py_outputs = PyDict::new(py);
        for (name, tensor) in rust_outputs {
            let py_tensor = PyTensor { inner: tensor };
            py_outputs.set_item(name, py_tensor)?;
        }

        Ok(py_outputs.into())
    }

    /// Get model input names
    ///
    /// Returns:
    ///     List[str]: Names of model inputs
    #[getter]
    fn inputs(&self) -> Vec<String> {
        self.inner.inputs().to_vec()
    }

    /// Get model output names
    ///
    /// Returns:
    ///     List[str]: Names of model outputs
    #[getter]
    fn outputs(&self) -> Vec<String> {
        self.inner.outputs().to_vec()
    }

    /// Get execution statistics
    ///
    /// Returns:
    ///     Dict: Statistics about the model
    fn stats(&self) -> PyResult<PyObject> {
        let py = Python::acquire_gil().python();
        let stats = self.inner.execution_stats();
        
        let py_stats = PyDict::new(py);
        py_stats.set_item("node_count", stats.node_count)?;
        py_stats.set_item("input_count", stats.input_count)?;
        py_stats.set_item("output_count", stats.output_count)?;
        py_stats.set_item("initializer_count", stats.initializer_count)?;
        
        Ok(py_stats.into())
    }

    /// String representation
    fn __repr__(&self) -> String {
        let stats = self.inner.execution_stats();
        format!(
            "ComputeGraph(nodes={}, inputs={}, outputs={})",
            stats.node_count, stats.input_count, stats.output_count
        )
    }
}

/// Load a model from file
///
/// Args:
///     model_path: Path to the model file (.json or .onnx)
///
/// Returns:
///     ComputeGraph: Loaded model ready for inference
///
/// Raises:
///     ValueError: If model file is invalid
///     FileNotFoundError: If model file doesn't exist
///
/// Example:
///     >>> graph = zeke.load_model("classifier.json")
///     >>> print(f"Model has {len(graph.inputs)} inputs")
#[pyfunction]
fn load_model(model_path: &str) -> PyResult<PyComputeGraph> {
    let graph = zeke::load_model(model_path)
        .map_err(|e| match e {
            zeke::ZekeError::Io(_) => PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(format!("{}", e)),
            _ => PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)),
        })?;
    
    Ok(PyComputeGraph { inner: graph })
}

/// Run inference on a model with given inputs (one-shot execution)
///
/// Args:
///     model_path: Path to the model file
///     inputs: Dictionary mapping input names to tensors or NumPy arrays
///
/// Returns:
///     Dict[str, Tensor]: Dictionary mapping output names to result tensors
///
/// Example:
///     >>> import numpy as np
///     >>> inputs = {"input": np.ones((1, 784), dtype=np.float32)}
///     >>> outputs = zeke.run_model("mnist.json", inputs)
///     >>> prediction = outputs["output"]
#[pyfunction]
fn run_model(model_path: &str, inputs: &PyDict) -> PyResult<PyObject> {
    let py = inputs.py();
    
    // Convert Python inputs to Rust HashMap
    let mut rust_inputs = HashMap::new();
    for (key, value) in inputs {
        let name: String = key.extract()?;
        
        // Handle both Tensor objects and NumPy arrays
        let tensor = if let Ok(py_tensor) = value.extract::<PyRef<PyTensor>>() {
            py_tensor.inner.clone()
        } else if let Ok(array) = value.extract::<PyReadonlyArray1<f32>>() {
            let data = array.as_slice()?.to_vec();
            let shape = vec![data.len()];
            ZekeTensor::new(shape, data)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Input values must be Tensor objects or NumPy arrays",
            ));
        };
        
        rust_inputs.insert(name, tensor);
    }

    // Run inference
    let rust_outputs = zeke_run_model(model_path, rust_inputs)
        .map_err(|e| match e {
            zeke::ZekeError::Io(_) => PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(format!("{}", e)),
            _ => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)),
        })?;

    // Convert outputs back to Python dict
    let py_outputs = PyDict::new(py);
    for (name, tensor) in rust_outputs {
        let py_tensor = PyTensor { inner: tensor };
        py_outputs.set_item(name, py_tensor)?;
    }

    Ok(py_outputs.into())
}

/// Get library version
///
/// Returns:
///     str: Version string
#[pyfunction]
fn version() -> &'static str {
    zeke::version()
}

/// Get available model format loaders
///
/// Returns:
///     List[str]: Available loader names
#[pyfunction]
fn available_loaders() -> Vec<&'static str> {
    zeke::available_loaders()
}

/// Python module definition
#[pymodule]
fn zeke(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_class::<PyComputeGraph>()?;
    m.add_function(wrap_pyfunction!(load_model, m)?)?;
    m.add_function(wrap_pyfunction!(run_model, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(available_loaders, m)?)?;
    
    // Add module metadata
    m.add("__version__", zeke::version())?;
    m.add("__doc__", "Zeke - Lightweight Edge AI Inference Runtime")?;
    
    Ok(())
}