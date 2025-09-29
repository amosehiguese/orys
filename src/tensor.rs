//! Core tensor implementation for the orys inference runtime
//!
//! This module provides the `Tensor` struct and associated operations for
//! storing and manipulating multi-dimensional arrays of floating-point data.
use crate::errors::{Result, OrysError};
use serde::{Deserialize, Serialize};

/// N-dimensional tensor for storing floating-point data
///
/// The tensor uses row-major memory layout (C-style) where the last dimension
/// varies fastest. For a 2D tensor with shape [2, 3]:
/// ```text
/// Index:  [0,0] [0,1] [0,2] [1,0] [1,1] [1,2]
/// Memory: [ 0 ]  [ 1 ]  [ 2 ]  [ 3 ]  [ 4 ]  [ 5 ]
/// ```
///
/// # Examples
///
/// ```rust
/// use orys::tensor::Tensor;
///
/// // Create a 2x3 tensor
/// let tensor = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
/// assert_eq!(tensor.shape(), &[2, 3]);
/// assert_eq!(tensor.size(), 6);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tensor {
    /// Shape of the tensor (dimensions)
    shape: Vec<usize>,
    /// Flattened data in row-major order
    data: Vec<f32>,
}

impl Tensor {
    /// Create a new tensor with given shape and data
    ///
    /// # Arguments
    /// * `shape` - Dimensions of the tensor
    /// * `data` - Flattened data in row-major order
    ///
    /// # Errors
    /// Returns `TensorSizeMismatch` if data length doesn't match shape requirements
    ///
    /// # Examples
    /// ```rust
    /// # use orys::tensor::Tensor;
    /// let tensor = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        let expected_size = shape.iter().product();
        if data.len() != expected_size {
            return Err(OrysError::TensorSizeMismatch { 
                shape: shape.clone(), 
                expected: expected_size, 
                actual: data.len(), 
            });
        }
        Ok(Tensor { shape, data})
    }

    /// Create a tensor filled with zeros
    ///
    /// # Examples
    /// ```rust
    /// # use orys::tensor::Tensor;
    /// let tensor = Tensor::zeros(vec![2, 3]);
    /// assert_eq!(tensor.data(), &[0.0; 6]);
    /// ```
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Tensor {
            shape,
            data: vec![0.0; size],
        }
    }

    /// Create a tensor filled with ones
    ///
    /// # Examples
    /// ```rust
    /// # use orys::tensor::Tensor;
    /// let tensor = Tensor::ones(vec![2, 2]);
    /// assert_eq!(tensor.data(), &[1.0, 1.0, 1.0, 1.0]);
    /// ```
    pub fn ones(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Tensor {
            shape,
            data: vec![1.0; size],
        }
    }

    /// Create a tensor from a flat vector, inferring 1D shape
    ///
    /// # Examples
    /// ```rust
    /// # use orys::tensor::Tensor;
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0]);
    /// assert_eq!(tensor.shape(), &[3]);
    /// ```
    pub fn from_vec(data: Vec<f32>) -> Self {
        let shape = vec![data.len()];
        Tensor { shape, data }
    }

    /// Create a scalar tensor (0-dimensional)
    ///
    /// # Examples
    /// ```rust
    /// # use orys::tensor::Tensor;
    /// let tensor = Tensor::scalar(42.0);
    /// assert_eq!(tensor.shape(), &[]);
    /// assert_eq!(tensor.data(), &[42.0]);
    /// ```
    pub fn scalar(value: f32) -> Self {
        Tensor {
            shape: vec![],
            data: vec![value],
        }
    }

    // === Accessors ===
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the raw data of the tensor
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Get mutable access to the raw data
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the total number of elements
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Check if the tensor is a scalar (0-dimensional)
    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }

    // === Indexing ===

    /// Convert multi-dimensional indices to flat index
    ///
    /// # Arguments
    /// * `indices` - Multi-dimensional indices (must match tensor dimensions)
    ///
    /// # Errors
    /// Returns `InvalidTensorOperation` if indices are invalid
    ///
    /// # Examples
    /// ```rust
    /// # use orys::tensor::Tensor;
    /// let tensor = Tensor::zeros(vec![2, 3]);
    /// let flat_idx = tensor.flat_index(&[1, 2])?;
    /// assert_eq!(flat_idx, 5); // 1 * 3 + 2
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn flat_index(&self, indices: &[usize]) -> Result<usize> {
        if indices.len() != self.ndim() {
            return Err(OrysError::invalid_tensor_op(format!(
                "Index dimensions {} don't match tensor dimensions {}",
                indices.len(),
                self.ndim()
            )));
        }

        let mut flat_idx = 0;
        let mut stride = 1;

        // Calculate index using row-major order (rightmost dimension varies fastest)
        for (i, &dim_size) in self.shape.iter().enumerate().rev() {
            let idx = indices[i];
            if idx >= dim_size {
                return Err(OrysError::invalid_tensor_op(format!(
                    "Index {} out of bounds for dimension {} with size {}",
                    idx, i, dim_size
                )));
            }
            flat_idx += idx * stride;
            stride *= dim_size;
        }

        Ok(flat_idx)
    }

    /// Get a value at the specified indices
    ///
    /// # Examples
    /// ```rust
    /// # use orys::tensor::Tensor;
    /// let tensor = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
    /// assert_eq!(tensor.get(&[1, 1])?, 4.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn get(&self, indices: &[usize]) -> Result<f32> {
        let flat_idx = self.flat_index(indices)?;
        Ok(self.data[flat_idx])
    }

    /// Set a value at the specified indices
    ///
    /// # Examples
    /// ```rust
    /// # use orys::tensor::Tensor;
    /// let mut tensor = Tensor::zeros(vec![2, 2]);
    /// tensor.set(&[0, 1], 5.0)?;
    /// assert_eq!(tensor.get(&[0, 1])?, 5.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn set(&mut self, indices: &[usize], value: f32) -> Result<()> {
        let flat_idx = self.flat_index(indices)?;
        self.data[flat_idx] = value;
        Ok(())
    }

    /// Reshape the tensor to new dimensions
    ///
    /// The total number of elements must remain the same.
    ///
    /// # Examples
    /// ```rust
    /// # use orys::tensor::Tensor;
    /// let mut tensor = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    /// tensor.reshape(vec![3, 2])?;
    /// assert_eq!(tensor.shape(), &[3, 2]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reshape(&mut self, new_shape: Vec<usize>) -> Result<()> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.size() {
            return Err(OrysError::invalid_tensor_op(format!(
                "Cannot reshape tensor with {} elements to shape {:?} requiring {} elements",
                self.size(),
                new_shape,
                new_size
            )));
        }
        self.shape = new_shape;
        Ok(())
    }

    /// Create a reshaped copy of the tensor
    ///
    /// # Examples
    /// ```rust
    /// # use orys::tensor::Tensor;
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    /// let reshaped = tensor.reshaped(vec![2, 2])?;
    /// assert_eq!(reshaped.shape(), &[2, 2]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reshaped(&self, new_shape: Vec<usize>) -> Result<Self> {
        let mut result = self.clone();
        result.reshape(new_shape)?;
        Ok(result)
    }

    /// Check if this tensor's shape is compatible with another for broadcasting
    ///
    /// Broadcasting rules:
    /// 1. Start from the trailing dimensions
    /// 2. Dimensions are compatible if they are equal or one of them is 1
    /// 3. Missing dimensions are assumed to be 1
    ///
    /// # Examples
    /// ```rust
    /// # use orys::tensor::Tensor;
    /// let a = Tensor::zeros(vec![3, 1]);     // Shape: [3, 1]
    /// let b = Tensor::zeros(vec![1, 4]);     // Shape: [1, 4]
    /// assert!(a.is_broadcastable_with(&b));  // Can broadcast to [3, 4]
    /// ```
    pub fn is_broadcastable_with(&self, other: &Tensor) -> bool {
        let max_dims = self.ndim().max(other.ndim());

        for i in 0..max_dims {
            let dim_a = self.shape.get(self.ndim().saturating_sub(i + 1)).copied().unwrap_or(1);
            let dim_b = other.shape.get(other.ndim().saturating_sub(i + 1)).copied().unwrap_or(1);

            if dim_a != dim_b && dim_a != 1 && dim_b != 1 {
                return false;
            }
        }
        true
    }

    // Calculate the broadcasted shape when combining with another tensor
    ///
    /// # Examples
    /// ```rust
    /// # use orys::tensor::Tensor;
    /// let a = Tensor::zeros(vec![3, 1]);
    /// let b = Tensor::zeros(vec![1, 4]);
    /// let broadcast_shape = a.broadcast_shape(&b)?;
    /// assert_eq!(broadcast_shape, vec![3, 4]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn broadcast_shape(&self, other: &Tensor) -> Result<Vec<usize>> {
        if !self.is_broadcastable_with(other) {
            return Err(OrysError::ShapeMismatch {
                expected: self.shape.clone(),
                actual: other.shape.clone(),
            });
        }

        let max_dims = self.ndim().max(other.ndim());
        let mut result_shape = Vec::with_capacity(max_dims);

        for i in 0..max_dims {
            let dim_a = self.shape.get(self.ndim().saturating_sub(i + 1)).copied().unwrap_or(1);
            let dim_b = other.shape.get(other.ndim().saturating_sub(i + 1)).copied().unwrap_or(1);
            result_shape.push(dim_a.max(dim_b));
        }

        result_shape.reverse();
        Ok(result_shape)
    }

    /// Apply a function to each element of the tensor
    ///
    /// # Examples
    /// ```rust
    /// # use orys::tensor::Tensor;
    /// let mut tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0]);
    /// tensor.map_inplace(|x| x * 2.0);
    /// assert_eq!(tensor.data(), &[2.0, 4.0, 6.0]);
    /// ```
    pub fn map_inplace<F>(&mut self, f: F)
    where
        F: Fn(f32) -> f32,
    {
        for value in &mut self.data {
            *value = f(*value);
        }
    }

    // Create a new tensor by applying a function to each element
    ///
    /// # Examples
    /// ```rust
    /// # use orys::tensor::Tensor;
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0]);
    /// let squared = tensor.map(|x| x * x);
    /// assert_eq!(squared.data(), &[1.0, 4.0, 9.0]);
    /// ```
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(f32) -> f32,
    {
        let new_data = self.data.iter().map(|&x| f(x)).collect();
        Tensor {
            shape: self.shape.clone(),
            data: new_data,
        }
    }
} 

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_tensor_creation() {
        let tensor = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.size(), 6);
        assert_eq!(tensor.ndim(), 2);
    }

    #[test]
    fn test_tensor_size_mismatch() {
        let result = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0]);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OrysError::TensorSizeMismatch { .. }));
    }

    #[test]
    fn test_tensor_zeros_ones() {
        let zeros = Tensor::zeros(vec![2, 2]);
        assert_eq!(zeros.data(), &[0.0, 0.0, 0.0, 0.0]);

        let ones = Tensor::ones(vec![2, 2]);
        assert_eq!(ones.data(), &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_scalar_tensor() {
        let scalar = Tensor::scalar(42.0);
        let expected: &[usize] = &[];
        assert!(scalar.is_scalar());
        assert_eq!(scalar.shape(), expected);
        assert_eq!(scalar.data(), &[42.0]);
    }

    #[test]
    fn test_indexing() {
        let tensor = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        
        // Test flat_index calculation
        assert_eq!(tensor.flat_index(&[0, 0]).unwrap(), 0);
        assert_eq!(tensor.flat_index(&[0, 2]).unwrap(), 2);
        assert_eq!(tensor.flat_index(&[1, 0]).unwrap(), 3);
        assert_eq!(tensor.flat_index(&[1, 2]).unwrap(), 5);

        // Test get/set
        assert_relative_eq!(tensor.get(&[1, 2]).unwrap(), 6.0);
        
        let mut tensor_mut = tensor.clone();
        tensor_mut.set(&[0, 1], 99.0).unwrap();
        assert_relative_eq!(tensor_mut.get(&[0, 1]).unwrap(), 99.0);
    }

    #[test]
    fn test_reshape() {
        let mut tensor = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        tensor.reshape(vec![3, 2]).unwrap();
        assert_eq!(tensor.shape(), &[3, 2]);
        
        // Data should remain the same
        assert_eq!(tensor.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_broadcasting_compatibility() {
        let a = Tensor::zeros(vec![3, 1]);
        let b = Tensor::zeros(vec![1, 4]);
        assert!(a.is_broadcastable_with(&b));

        let c = Tensor::zeros(vec![3, 2]);
        let d = Tensor::zeros(vec![3, 4]);
        assert!(!c.is_broadcastable_with(&d));
    }

    #[test]
    fn test_broadcast_shape() {
        let a = Tensor::zeros(vec![3, 1]);
        let b = Tensor::zeros(vec![1, 4]);
        let broadcast_shape = a.broadcast_shape(&b).unwrap();
        assert_eq!(broadcast_shape, vec![3, 4]);

        let c = Tensor::zeros(vec![2, 1, 3]);
        let d = Tensor::zeros(vec![1, 4, 1]);
        let broadcast_shape = c.broadcast_shape(&d).unwrap();
        assert_eq!(broadcast_shape, vec![2, 4, 3]);
    }

    #[test]
    fn test_map_operations() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0]);
        
        let squared = tensor.map(|x| x * x);
        assert_eq!(squared.data(), &[1.0, 4.0, 9.0]);

        let mut tensor_mut = tensor.clone();
        tensor_mut.map_inplace(|x| x + 1.0);
        assert_eq!(tensor_mut.data(), &[2.0, 3.0, 4.0]);
    }
}