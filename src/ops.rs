//! Core operator implementations for the orys inference runtime
//!
//! This module provides implementations of some ONNX operators:
//! - MatMul: Matrix multiplication
//! - Add: Element-wise addition with broadcasting
//! - ReLU: Rectified Linear Unit activation
//! - Sigmoid: Sigmoid activation function
//! 
use crate::errors::{Result, OrysError};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use ndarray::ArrayView2;

/// Trait for all operators in the computation graph
///
/// This trait provides a uniform interface for executing operations
/// on tensors, enabling dynamic dispatch in the graph executor.
pub trait Operator: std::fmt::Debug + Send + Sync {
    /// Execute the operator on input tensors
    ///
    /// # Arguments
    /// * `inputs` - Slice of input tensors
    ///
    /// # Returns
    /// * `Result<Tensor>` - The output tensor or an error
    fn execute(&self, inputs: &[Tensor]) -> Result<Tensor>;

    /// Get the operator type name
    fn op_type(&self) -> &'static str;

    /// Validate input shapes and count before execution
    fn validate_inputs(&self, inputs: &[Tensor]) -> Result<()>;

    /// Get expected number of inputs for this operator
    fn expected_input_count(&self) -> usize;
}

/// Matrix multiplication operator
///
/// Performs standard matrix multiplication between two 2D tensors.
/// For tensors A(m×k) and B(k×n), produces output C(m×n) where C[i,j] = Σ(A[i,k] * B[k,j])
///
/// # Input Requirements
/// - Exactly 2 input tensors
/// - Both tensors must be 2D
/// - Inner dimensions must match (A.shape[1] == B.shape[0])
///
/// # Examples
/// ```rust
/// # use orys::ops::{Operator, MatMul};
/// # use orys::tensor::Tensor;
/// let matmul = MatMul;
/// let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
/// let b = Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
/// let result = matmul.execute(&[a, b])?;
/// assert_eq!(result.shape(), &[2, 2]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatMul;

impl Operator for MatMul {
    fn execute(&self, inputs: &[Tensor]) -> Result<Tensor> {
        self.validate_inputs(inputs)?;

        let a = &inputs[0];
        let b = &inputs[1];

        let (m, k) = (a.shape()[0], a.shape()[1]);
        let (k2, n) = (b.shape()[0], b.shape()[1]);

        // Double-check dimensions (should be caught in validate_inputs)
        if k != k2 {
            return Err(OrysError::operator_validation(
                "MatMul", 
                format!("Inner dimensions don't match: {} vs {}", k, k2),
            ));
        }

        let a_mat = ArrayView2::from_shape((m, k), a.data())
            .map_err(|_| OrysError::invalid_tensor_op("Invalid A shape".to_string()))?;

        let b_mat = ArrayView2::from_shape((k, n), b.data())
            .map_err(|_| OrysError::invalid_tensor_op("Invalid B shape".to_string()))?;

        let result = a_mat.dot(&b_mat);
        let result = result.into_raw_vec_and_offset().0;
        Ok(Tensor::new(vec![m, n], result)?)
    }

    fn op_type(&self) -> &'static str {
        "MatMul"
    }

    fn validate_inputs(&self, inputs: &[Tensor]) -> Result<()> {
        if inputs.len() != self.expected_input_count() {
            return Err(OrysError::InvalidInputCount {
                op_type: self.op_type().to_string(),
                expected: self.expected_input_count(),
                actual: inputs.len(),
            });
        }

        let a = &inputs[0];
        let b = &inputs[1];

        // Check that both tensors are 2D
        if a.ndim() != 2 {
            return Err(OrysError::operator_validation(
                "MatMul",
                format!("First input must be 2D, got {}D tensor", a.ndim()),
            ));
        }
        if b.ndim() != 2 {
            return Err(OrysError::operator_validation(
                "MatMul",
                format!("Second input must be 2D, got {}D tensor", b.ndim()),
            ));
        }

        // Check dimension compatibility
        if a.shape()[1] != b.shape()[0] {
            return Err(OrysError::operator_validation(
                "MatMul",
                format!(
                    "Incompatible dimensions for matrix multiplication: [{}×{}] × [{}×{}]",
                    a.shape()[0], a.shape()[1], b.shape()[0], b.shape()[1]
                ),
            ));
        }

        Ok(())
    }
    
    fn expected_input_count(&self) -> usize {
        2
    }
}

/// Element-wise addition operator with broadcasting
///
/// Adds two tensors element-wise, supporting broadcasting
/// for tensors with compatible shapes.
///
/// # Input Requirements
/// - Exactly 2 input tensors
/// - Tensors must be broadcastable
///
/// # Broadcasting Rules
/// - Dimensions are compared from right to left
/// - Missing dimensions are treated as 1
/// - Dimensions are compatible if they're equal or one is 1
///
/// # Examples
/// ```rust
/// # use orys::ops::{Operator, Add};
/// # use orys::tensor::Tensor;
/// let add = Add;
/// let a = Tensor::new(vec![2, 1], vec![1.0, 2.0])?;
/// let b = Tensor::new(vec![1, 3], vec![10.0, 20.0, 30.0])?;
/// let result = add.execute(&[a, b])?;
/// assert_eq!(result.shape(), &[2, 3]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Add;

impl Operator for Add {
    fn execute(&self, inputs: &[Tensor]) -> Result<Tensor> {
        self.validate_inputs(inputs)?;
        
        let a = &inputs[0];
        let b = &inputs[1];
        
        // Calculate the broadcast shape
        let broadcast_shape = a.broadcast_shape(b)?;
        let result_size = broadcast_shape.iter().product();
        let mut result_data = vec![0.0; result_size];

        // Perform element-wise addition with broadcasting
        for i in 0..result_size {
            let a_idx = self.broadcast_index(i, &broadcast_shape, a.shape())?;
            let b_idx = self.broadcast_index(i, &broadcast_shape, b.shape())?;
            
            result_data[i] = a.data()[a_idx] + b.data()[b_idx];
        }

        Tensor::new(broadcast_shape, result_data)
    }

    fn op_type(&self) -> &'static str {
        "Add"
    }

    fn validate_inputs(&self, inputs: &[Tensor]) -> Result<()> {
        if inputs.len() != self.expected_input_count() {
            return Err(OrysError::InvalidInputCount {
                op_type: self.op_type().to_string(),
                expected: self.expected_input_count(),
                actual: inputs.len(),
            });
        }

        let a = &inputs[0];
        let b = &inputs[1];

        if !a.is_broadcastable_with(b) {
            return Err(OrysError::operator_validation(
                "Add",
                format!(
                    "Tensors with shapes {:?} and {:?} are not broadcastable",
                    a.shape(),
                    b.shape()
                ),
            ));
        }

        Ok(())
    }

    fn expected_input_count(&self) -> usize {
        2
    }
}

impl Add {
    /// Calculate the source index for broadcasting
    ///
    /// Given a flat index in the broadcast result, calculates the corresponding
    /// index in the source tensor, handling dimension differences and size-1 dimensions.
    fn broadcast_index(
        &self,
        flat_idx: usize,
        broadcast_shape: &[usize],
        source_shape: &[usize],
    ) -> Result<usize> {
        // Convert flat index to multi-dimensional indices in broadcast space
        let mut broadcast_indices = Vec::new();
        let mut remaining = flat_idx;
        
        for &dim_size in broadcast_shape.iter().rev() {
            broadcast_indices.push(remaining % dim_size);
            remaining /= dim_size;
        }
        broadcast_indices.reverse();

        // Map broadcast indices to source indices
        let mut source_indices = Vec::new();
        let shape_diff = broadcast_shape.len().saturating_sub(source_shape.len());
        
        for (i, &broadcast_idx) in broadcast_indices.iter().enumerate() {
            if i < shape_diff {
                // This dimension doesn't exist in source, skip
                continue;
            }
            
            let source_dim_idx = i - shape_diff;
            let source_dim_size = source_shape[source_dim_idx];
            
            if source_dim_size == 1 {
                // Broadcast dimension, always use index 0
                source_indices.push(0);
            } else {
                // Normal dimension, use the broadcast index
                source_indices.push(broadcast_idx);
            }
        }

        // Convert multi-dimensional indices to flat index for source tensor
        let mut source_flat_idx = 0;
        let mut stride = 1;
        
        for (i, &dim_size) in source_shape.iter().enumerate().rev() {
            let idx = source_indices[source_indices.len() - 1 - (source_shape.len() - 1 - i)];
            source_flat_idx += idx * stride;
            stride *= dim_size;
        }

        Ok(source_flat_idx)
    }
}

/// Rectified Linear Unit (ReLU) activation function
///
/// Applies the ReLU function element-wise: f(x) = max(0, x)
///
/// # Input Requirements
/// - Exactly 1 input tensor
/// - Any shape is supported
///
/// # Examples
/// ```rust
/// # use orys::ops::{Operator, ReLU};
/// # use orys::tensor::Tensor;
/// let relu = ReLU;
/// let input = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0]);
/// let result = relu.execute(&[input])?;
/// assert_eq!(result.data(), &[0.0, 0.0, 1.0, 2.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```     
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReLU;

impl Operator for ReLU {
    fn execute(&self, inputs: &[Tensor]) -> Result<Tensor> {
        self.validate_inputs(inputs)?;
        
        let input = &inputs[0];
        let result = input.map(|x| x.max(0.0));
        
        Ok(result)
    }

    fn op_type(&self) -> &'static str {
        "ReLU"
    }

    fn validate_inputs(&self, inputs: &[Tensor]) -> Result<()> {
        if inputs.len() != self.expected_input_count() {
            return Err(OrysError::InvalidInputCount {
                op_type: self.op_type().to_string(),
                expected: self.expected_input_count(),
                actual: inputs.len(),
            });
        }
        Ok(())
    }

    fn expected_input_count(&self) -> usize {
        1
    }
}

/// Sigmoid activation function
///
/// Applies the sigmoid function element-wise: f(x) = 1 / (1 + e^(-x))
///
/// # Input Requirements
/// - Exactly 1 input tensor
/// - Any shape is supported
///
/// # Examples
/// ```rust
/// # use orys::ops::{Operator, Sigmoid};
/// # use orys::tensor::Tensor;
/// let sigmoid = Sigmoid;
/// let input = Tensor::from_vec(vec![0.0, 1.0, -1.0]);
/// let result = sigmoid.execute(&[input])?;
/// // result.data() ≈ [0.5, 0.731, 0.269]
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sigmoid;

impl Operator for Sigmoid {
    fn execute(&self, inputs: &[Tensor]) -> Result<Tensor> {
        self.validate_inputs(inputs)?;
        
        let input = &inputs[0];
        let result = input.map(|x| {
            // Numerically stable sigmoid computation
            if x >= 0.0 {
                let exp_neg_x = (-x).exp();
                1.0 / (1.0 + exp_neg_x)
            } else {
                let exp_x = x.exp();
                exp_x / (1.0 + exp_x)
            }
        });
        
        Ok(result)
    }

    fn op_type(&self) -> &'static str {
        "Sigmoid"
    }

    fn validate_inputs(&self, inputs: &[Tensor]) -> Result<()> {
        if inputs.len() != self.expected_input_count() {
            return Err(OrysError::InvalidInputCount {
                op_type: self.op_type().to_string(),
                expected: self.expected_input_count(),
                actual: inputs.len(),
            });
        }
        Ok(())
    }

    fn expected_input_count(&self) -> usize {
        1
    }
}

/// Factory function to create operators from string names
///
/// This function enables dynamic operator creation from model files.
///
/// # Arguments
/// * `op_type` - String name of the operator ("MatMul", "Add", "ReLU", "Sigmoid")
///
/// # Returns
/// * `Result<Box<dyn Operator>>` - Boxed operator or error for unsupported types
///
/// # Examples
/// ```rust
/// # use orys::ops::create_operator;
/// let matmul_op = create_operator("MatMul")?;
/// assert_eq!(matmul_op.op_type(), "MatMul");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn create_operator(op_type: &str) -> Result<Box<dyn Operator>> {
    match op_type {
        "MatMul" => Ok(Box::new(MatMul)),
        "Add" => Ok(Box::new(Add)),
        "ReLU" => Ok(Box::new(ReLU)),
        "Sigmoid" => Ok(Box::new(Sigmoid)),
        _ => Err(OrysError::UnsupportedOperator {
            op_type: op_type.to_string(),
        }),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_matmul_basic() {
        let matmul = MatMul;
        
        // 2×3 × 3×2 = 2×2
        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        
        let result = matmul.execute(&[a, b]).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        
        // Expected: [[22, 28], [49, 64]]
        let expected = vec![22.0, 28.0, 49.0, 64.0];
        for (actual, expected) in result.data().iter().zip(expected.iter()) {
            assert_relative_eq!(actual, expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_matmul_validation() {
        let matmul = MatMul;
        
        // Wrong number of inputs
        let a = Tensor::ones(vec![2, 2]);
        assert!(matmul.execute(&[a.clone()]).is_err());
        assert!(matmul.execute(&[a.clone(), a.clone(), a.clone()]).is_err());
        
        // Wrong dimensions
        let a = Tensor::ones(vec![2, 3]);
        let b = Tensor::ones(vec![4, 2]); // 3 != 4
        assert!(matmul.execute(&[a, b]).is_err());
        
        // Non-2D tensors
        let a = Tensor::ones(vec![2]);
        let b = Tensor::ones(vec![2, 2]);
        assert!(matmul.execute(&[a, b]).is_err());
    }

    #[test]
    fn test_add_broadcasting() {
        let add = Add;
        
        // [2, 1] + [1, 3] = [2, 3]
        let a = Tensor::new(vec![2, 1], vec![1.0, 2.0]).unwrap();
        let b = Tensor::new(vec![1, 3], vec![10.0, 20.0, 30.0]).unwrap();
        
        let result = add.execute(&[a, b]).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        
        // Expected: [[11, 21, 31], [12, 22, 32]]
        let expected = vec![11.0, 21.0, 31.0, 12.0, 22.0, 32.0];
        for (actual, expected) in result.data().iter().zip(expected.iter()) {
            assert_relative_eq!(actual, expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_add_same_shape() {
        let add = Add;
        
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0]);
        
        let result = add.execute(&[a, b]).unwrap();
        assert_eq!(result.data(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_add_incompatible_shapes() {
        let add = Add;
        
        let a = Tensor::ones(vec![2, 3]);
        let b = Tensor::ones(vec![2, 4]); // 3 != 4, not broadcastable
        
        assert!(add.execute(&[a, b]).is_err());
    }

    #[test]
    fn test_relu() {
        let relu = ReLU;
        
        let input = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = relu.execute(&[input]).unwrap();
        
        assert_eq!(result.data(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sigmoid() {
        let sigmoid = Sigmoid;
        
        let input = Tensor::from_vec(vec![0.0, 1.0, -1.0]);
        let result = sigmoid.execute(&[input]).unwrap();
        
        // Check known values
        assert_relative_eq!(result.data()[0], 0.5, epsilon = 1e-6);
        assert_relative_eq!(result.data()[1], 1.0 / (1.0 + (-1.0_f32).exp()), epsilon = 1e-6);
        assert_relative_eq!(result.data()[2], (-1.0_f32).exp() / (1.0 + (-1.0_f32).exp()), epsilon = 1e-6);
    }

    #[test]
    fn test_sigmoid_numerical_stability() {
        let sigmoid = Sigmoid;
        
        // Test with extreme values
        let input = Tensor::from_vec(vec![-100.0, 100.0]);
        let result = sigmoid.execute(&[input]).unwrap();
        
        // Should not produce NaN or infinity
        assert!(result.data()[0].is_finite());
        assert!(result.data()[1].is_finite());
        assert!(result.data()[0] >= 0.0 && result.data()[0] <= 1.0);
        assert!(result.data()[1] >= 0.0 && result.data()[1] <= 1.0);
    }

    #[test]
    fn test_operator_factory() {
        assert!(create_operator("MatMul").is_ok());
        assert!(create_operator("Add").is_ok());
        assert!(create_operator("ReLU").is_ok());
        assert!(create_operator("Sigmoid").is_ok());
        
        assert!(create_operator("InvalidOp").is_err());
    }

    #[test]
    fn test_operator_metadata() {
        let ops: Vec<Box<dyn Operator>> = vec![
            Box::new(MatMul),
            Box::new(Add),
            Box::new(ReLU),
            Box::new(Sigmoid),
        ];
        
        for op in ops {
            assert!(!op.op_type().is_empty());
            assert!(op.expected_input_count() > 0);
        }
    }
}