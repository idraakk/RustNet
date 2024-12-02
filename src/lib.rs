// This file declares all the modules that are part of the RustNet library.
// Each module corresponds to a specific component or functionality of the neural network system.

pub mod activations; // Contains activation functions like ReLU and Sigmoid.
pub mod data;        // Handles data loading and preprocessing.
pub mod layers;      // Defines the neural network layers and their operations.
pub mod loss;        // Implements loss functions used for training the network.
pub mod train;       // Manages the overall training and prediction workflow of the neural network.
pub mod serde_arrays; // Provides utilities for serializing and deserializing ndarray structures.
