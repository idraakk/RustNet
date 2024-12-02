// This module defines the structure and functionality of a dense (fully connected) layer in the neural network.
// Dense layers are the building blocks of most neural networks, connecting every input to every output neuron.

use ndarray::{Array, Array2}; // Provides multi-dimensional array support.
use rand::Rng;               // Used to initialize weights randomly.
use serde::{Deserialize, Serialize}; // Allows serialization and deserialization of layers for saving/loading.
use crate::serde_arrays;     // Utilities for handling ndarray serialization.

/// Represents a dense layer in the neural network.
/// A dense layer has weights, biases, inputs, and outputs.
#[derive(Serialize, Deserialize)]
pub struct DenseLayer {
    #[serde(with = "serde_arrays")]
    pub weights: Array2<f64>, // The weight matrix connecting input to output neurons.
    #[serde(with = "serde_arrays")]
    pub biases: Array2<f64>,  // The bias vector added to the weighted sum.
    #[serde(skip_serializing, skip_deserializing)]
    pub inputs: Option<Array2<f64>>, // Stores inputs during the forward pass.
    #[serde(skip_serializing, skip_deserializing)]
    pub outputs: Option<Array2<f64>>, // Stores outputs during the forward pass.
}

impl DenseLayer {
    /// Creates a new dense layer with randomly initialized weights and biases.
    /// - `input_size`: Number of input neurons.
    /// - `output_size`: Number of output neurons.
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Array::from_shape_fn((input_size, output_size), |_| rng.gen_range(-0.5..0.5)); // Random values between -0.5 and 0.5.
        let biases = Array::zeros((1, output_size)); // Initialize biases to zeros.

        DenseLayer {
            weights,
            biases,
            inputs: None,
            outputs: None,
        }
    }

    /// Performs the forward pass for the layer.
    /// - Multiplies inputs by weights, adds biases, and applies an activation function.
    pub fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        self.inputs = Some(input.clone()); // Store inputs for backpropagation.
        let output = input.dot(&self.weights) + &self.biases; // Compute the weighted sum.
        output // Return the result for use in the next layer.
    }
}
