// This module implements the core functionality for training and using the neural network.
// It includes methods for forward propagation, backpropagation, weight updates, and model persistence.

use crate::activations::*; // Import activation functions and their derivatives.
use crate::layers::*;      // Import dense layer structures.
use crate::loss::*;        // Import loss function for evaluating the network's performance.
use ndarray::Array2;       // Import 2D array support for numerical computations.
use serde::{Deserialize, Serialize}; // Enable saving and loading the model.
use std::fs::File;         // File operations for model persistence.
use std::io::Write;        // Write operations for saving predictions.

/// Represents the neural network, which is composed of multiple layers.
#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub layers: Vec<DenseLayer>, // A vector of dense layers that make up the network.
}

impl NeuralNetwork {
    /// Creates a new neural network with the specified layer sizes.
    /// - `layer_sizes`: A vector specifying the number of neurons in each layer.
    pub fn new(layer_sizes: Vec<usize>) -> Self {
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            // Initialize each dense layer with random weights and biases.
            layers.push(DenseLayer::new(layer_sizes[i], layer_sizes[i + 1]));
        }
        NeuralNetwork { layers }
    }

    /// Performs a forward pass through the network.
    /// - Passes input data sequentially through all layers, applying activation functions as needed.
    /// Returns the final output.
    pub fn forward(&mut self, mut input: Array2<f64>) -> Array2<f64> {
        let num_layers = self.layers.len(); // Get the total number of layers.
        for (i, layer) in self.layers.iter_mut().enumerate() {
            println!("Layer {}: Input shape: {:?}", i, input.dim());
            input = layer.forward(&input); // Compute the output for the current layer.

            // Apply ReLU activation function for all layers except the output layer.
            if i < num_layers - 1 {
                input = relu(&input);
            }

            layer.outputs = Some(input.clone()); // Store the output for backpropagation.
            println!("Layer {}: Output shape: {:?}", i, input.dim());
        }
        input // Return the final output.
    }

    /// Performs backpropagation to compute gradients and update weights and biases.
    /// - `targets`: The actual target values for the inputs.
    /// - `learning_rate`: Determines the step size for updating parameters.
    pub fn backward(&mut self, targets: &Array2<f64>, learning_rate: f64) {
        let num_layers = self.layers.len();
        let mut gradient = None; // Used to store the gradient for the previous layer.

        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
            println!("Backward pass - Layer {}", i);

            let outputs = layer.outputs.as_ref().unwrap(); // Get the stored outputs.
            let inputs = layer.inputs.as_ref().unwrap();   // Get the stored inputs.

            if i == num_layers - 1 {
                // For the output layer, compute the gradient of the loss with respect to the output.
                let d_z = outputs - targets;
                println!("Layer {}: d_z shape: {:?}", i, d_z.dim());

                // Compute gradients for weights and biases.
                let d_w = inputs.t().dot(&d_z) / inputs.dim().0 as f64;
                let db = d_z.sum_axis(ndarray::Axis(0)) / inputs.dim().0 as f64;

                println!("Layer {}: d_w shape: {:?}", i, d_w.dim());
                println!("Layer {}: db shape: {:?}", i, db.dim());

                // Update weights and biases using the computed gradients.
                layer.weights -= &(d_w * learning_rate);
                layer.biases -= &(db.insert_axis(ndarray::Axis(0)) * learning_rate);

                // Compute the gradient for the next layer.
                gradient = Some(d_z.dot(&layer.weights.t()));
            } else {
                // For hidden layers, compute the gradient of the activation function.
                let d_a = gradient.unwrap();
                let d_z = d_a * relu_derivative(outputs);

                println!("Layer {}: d_z shape: {:?}", i, d_z.dim());

                // Compute gradients for weights and biases.
                let d_w = inputs.t().dot(&d_z) / inputs.dim().0 as f64;
                let db = d_z.sum_axis(ndarray::Axis(0)) / inputs.dim().0 as f64;

                println!("Layer {}: d_w shape: {:?}", i, d_w.dim());
                println!("Layer {}: db shape: {:?}", i, db.dim());

                // Update weights and biases.
                layer.weights -= &(d_w * learning_rate);
                layer.biases -= &(db.insert_axis(ndarray::Axis(0)) * learning_rate);

                // Compute the gradient for the next layer.
                gradient = Some(d_z.dot(&layer.weights.t()));
            }
        }
    }

    /// Trains the neural network using the provided inputs and targets.
    /// - `inputs`: Input features for the training dataset.
    /// - `targets`: Target labels corresponding to the inputs.
    /// - `learning_rate`: Step size for parameter updates.
    /// - `epochs`: Number of times to iterate over the entire dataset.
    pub fn train(
        &mut self,
        inputs: &Array2<f64>,
        targets: &Array2<f64>,
        learning_rate: f64,
        epochs: usize,
    ) {
        let mut file = File::create("predictions.csv").expect("Unable to create file");

        for epoch in 0..epochs {
            // Perform a forward pass to get predictions.
            let predictions = self.forward(inputs.clone());
            let loss = mean_squared_error(&predictions, targets); // Compute the loss.
            println!("Epoch {}: Loss = {}", epoch, loss);

            // Save predictions and targets at the last epoch.
            if epoch == epochs - 1 {
                for (pred, target) in predictions.outer_iter().zip(targets.outer_iter()) {
                    writeln!(file, "{},{}", pred[0], target[0]).unwrap();
                }
            }

            // Perform backpropagation to update parameters.
            self.backward(targets, learning_rate);

            // Check for NaN values in weights and biases to avoid invalid states.
            if self
                .layers
                .iter()
                .any(|layer| has_nan(&layer.weights) || has_nan(&layer.biases))
            {
                panic!("Encountered NaN in weights or biases during training");
            }
        }
    }

    /// Saves the trained model to a JSON file.
    pub fn save(&self, path: &str) {
        let file = File::create(path).expect("Unable to create file");
        serde_json::to_writer(file, &self.layers).expect("Unable to write model");
    }

    /// Loads a saved model from a JSON file.
    pub fn load(path: &str) -> Self {
        let file = File::open(path).expect("Unable to open file");
        let layers: Vec<DenseLayer> = serde_json::from_reader(file).expect("Unable to read model");
        NeuralNetwork { layers }
    }
}

/// Checks if an array contains NaN values.
/// NaNs can disrupt training and lead to invalid computations.
fn has_nan(array: &Array2<f64>) -> bool {
    array.iter().any(|&x| x.is_nan())
}
