// This module provides activation functions and their derivatives.
// Activation functions introduce non-linearity into the neural network, allowing it to learn complex patterns.

use ndarray::Array2;

/// Implements the ReLU (Rectified Linear Unit) activation function.
/// ReLU sets all negative values to 0 while keeping positive values unchanged.
/// Why ReLU? It is computationally efficient and mitigates the vanishing gradient problem in deep networks.
pub fn relu(input: &Array2<f64>) -> Array2<f64> {
    input.mapv(|x| if x > 0.0 { x } else { 0.0 }) // Element-wise comparison to set negatives to 0.
}

/// Computes the derivative of the ReLU activation function.
/// The derivative is 1 for positive inputs and 0 for negative inputs.
/// Why needed? It is used during backpropagation to update weights and biases.
pub fn relu_derivative(input: &Array2<f64>) -> Array2<f64> {
    input.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }) // Returns 1 or 0 based on the input value.
}

/// Implements the Sigmoid activation function.
/// Sigmoid maps any real-valued input to a range between 0 and 1.
/// Why Sigmoid? It is useful for binary classification problems.
pub fn sigmoid(input: &Array2<f64>) -> Array2<f64> {
    input.mapv(|x| 1.0 / (1.0 + (-x).exp())) // Applies the sigmoid formula to each element.
}

/// Computes the derivative of the Sigmoid function.
/// The derivative is sig(x) * (1 - sig(x)) for a given input x.
/// Why needed? It allows the network to propagate error gradients during backpropagation.
pub fn sigmoid_derivative(input: &Array2<f64>) -> Array2<f64> {
    let sig = sigmoid(input); // Compute the sigmoid first.
    &sig * &(1.0 - &sig)      // Apply the derivative formula element-wise.
}
