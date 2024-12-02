# Neural Network Library in Rust

Welcome to the **Neural Network Library in Rust**, a comprehensive, from-scratch implementation of a neural network designed for classifying RF wave data. This project demonstrates how to build a basic neural network library with matrix operations, training, and prediction functionality, entirely in Rust. The neural network is demonstrated through a simple classification of radio frequency waves through synthetic self generated data. It includes:

- **Custom Serialization and Deserialization** for model persistence.

- **Matrix Operations** using the `ndarray` crate.

- **Training and Prediction** pipelines with detailed data preprocessing.

- **Command-line Interface (CLI)** for ease of use.


## **Table of Contents**

- Project Overview

- Features

- Directory Structure

- Dependencies

- Getting Started

- Usage

- Training the Model

- Making Predictions

- File-wise Functions and Significance

- Technical Details

- Data Preprocessing

- Network Architecture

- Training

- Prediction

- Serialization

- Future Improvements

- Contributing

- License


## **Project Overview**

This library implements a basic feedforward neural network for classifying RF wave data. The network uses fully connected layers, ReLU activation, and mean squared error as the loss function. The focus is on understanding neural network mechanics, matrix operations, and Rust's ownership and borrowing concepts.


## **Features**

- **Customizable Network Architecture:** Easily specify the number of layers and neurons.

- **CLI Interface:** Train models and make predictions directly from the command line.

- **Serialization Support:** Save and load models to/from JSON files.

- **Data Preprocessing:** Normalization and dataset splitting for training and prediction.

- **Detailed Logging:** Step-by-step matrix operations logged during training and prediction for debugging and learning.


## **Directory Structure**

```
.
├── Cargo.toml        # Project metadata and dependencies
├── src
│   ├── main.rs       # Entry point with CLI commands
│   ├── activations.rs # Activation functions and derivatives
│   ├── data.rs       # Data loading and preprocessing
│   ├── layers.rs     # Dense layer implementation
│   ├── loss.rs       # Loss function
│   ├── serde_arrays.rs # Custom serialization for ndarray
│   ├── train.rs      # Training logic
│   ├── lib.rs        # Module declarations
├── rfdatagenerator.py # Script for generating RF wave data
</code></pre>

## **Dependencies**

The project relies on the following crates:

- **ndarray**: For matrix operations.

- **serde**: For serialization and deserialization.

- **serde_json**: For JSON file handling.

- **rand**: For random number generation (e.g., weights initialization).

- **clap**: For command-line argument parsing.

- **csv**: For loading and saving CSV files.

Add them to `Cargo.toml`:

```
[dependencies]
ndarray = { version = "0.15", features = ["serde"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
rand = "0.8"
clap = { version = "4.0", features = ["derive"] }
csv = "1.1"
</code></pre>

## **Getting Started**

### **Prerequisites**

- **Rust**: Install Rust and Cargo from[ ](https://www.rust-lang.org/)[Rust's official website](https://www.rust-lang.org/).

- **Python**: Required for generating the dataset using the included Python script.

### **Clone the Repository**

```
git clone https://github.com/your-username/neural-net-rust.git
cd neural-net-rust
</code></pre>
### **Install Dependencies**

```
cargo build
</code></pre>

## **Usage**

### **1. Generate Dataset**

Run the Python script to generate RF wave data:

```
python rfdatagenerator.py
</code></pre>
This creates a file `rf_data.csv` with 1000 samples of RF wave data.


### **2. Training the Model**

Use the `train` command to train the model:

```
cargo run -- train --data rf_data.csv
</code></pre>
- **Input Data**: `rf_data.csv` (CSV file with features and labels).

- **Output Files**:

- `model.json`: Trained model weights and biases.

- `predictions.csv`: Predictions and actual labels during the final epoch.


### **3. Making Predictions**

Use the `predict` command to make predictions with the trained model:

```
cargo run -- predict --model model.json --input rf_data.csv
</code></pre>
- **Input Data**: `rf_data.csv` (only features are used).

- **Output**: Predictions are printed to the console.


## **File-wise Functions and Significance**

Here is a detailed breakdown of each file and its purpose:

### **1. **`main.rs`

The entry point of the application. Provides a CLI interface for training and prediction.

- **Functions**:

- `normalize`: Normalizes input features to zero mean and unit variance.

- Command handlers:

- `Train`: Handles the training pipeline.

- `Predict`: Handles the prediction pipeline.

### **2. **`activations.rs`

Implements activation functions and their derivatives.

- **Functions**:

- `relu`: Applies the ReLU activation function.

- `relu_derivative`: Computes the derivative of the ReLU function.

- `sigmoid`: Applies the sigmoid activation function (optional for future use).

- `sigmoid_derivative`: Computes the derivative of the sigmoid function.

### **3. **`data.rs`

Handles data loading and preprocessing.

- **Functions**:

- `load_csv`: Loads a dataset from a CSV file into an `ndarray::Array2<f64>`.

### **4. **`layers.rs`

Defines the `DenseLayer` struct, which represents a fully connected layer.

- **Functions**:

- `new`: Initializes a dense layer with random weights and zero biases.

- `forward`: Computes the output of the layer during the forward pass.

### **5. **`loss.rs`

Implements the loss function.

- **Functions**:

- `mean_squared_error`: Computes the mean squared error between predictions and targets.

### **6. **`serde_arrays.rs`

Handles custom serialization and deserialization for `ndarray::Array2<f64>`.

- **Functions**:

- `serialize`: Serializes an `Array2<f64>` into a nested vector.

- `deserialize`: Reconstructs an `Array2<f64>` from serialized data.

### **7. **`train.rs`

Contains the core logic for training and saving the model.

- **Structs**:

- `NeuralNetwork`: Represents the entire neural network.

- **Functions**:

- `new`: Initializes a neural network with specified architecture.

- `forward`: Performs a forward pass through the network.

- `backward`: Performs backpropagation and updates weights and biases.

- `train`: Trains the network using gradient descent.

- `save`: Saves the model to a JSON file.

- `load`: Loads the model from a JSON file.

### **8. **`rfdatagenerator.py`

A Python script to simulate RF wave data for training and testing.

- **Functions**:

- Simulates `Frequency`, `Amplitude`, and corresponding labels.

- Exports data to a CSV file (`rf_data.csv`).


## **Technical Details**

### **Data Preprocessing**

1. **Dataset Structure**: Each row in `rf_data.csv` consists of:

- `Frequency`: Feature 1

- `Amplitude`: Feature 2

- `Label`: Target

- **Normalization**: Both features are normalized to zero mean and unit variance.

- **Splitting**: During training, the label column is separated from the features.

### **Network Architecture**

- Example: `[2, 10, 1]`

- **Input Layer**: 2 neurons (features: Frequency and Amplitude).

- **Hidden Layer**: 10 neurons (ReLU activation).

- **Output Layer**: 1 neuron (linear activation).

### **Training**

- **Forward Pass**:

- Computes layer-wise activations and outputs.

- **Loss Function**:

- Mean Squared Error (MSE).

- **Backward Pass**:

- Computes gradients using backpropagation.

- Updates weights and biases using gradient descent.

- **Hyperparameters**:

- Learning Rate: 0.001

- Epochs: 1000

### **Prediction**

- **Input**:

- Normalized features.

- **Output**:

- Final layer activations.

### **Serialization**

- **Model Format**:

- Saved as `model.json`.

- Contains layer weights and biases serialized as nested vectors.

- **Custom Serialization**:

- Uses `serde_arrays` for handling `ndarray::Array2`.


## **Future Improvements**

- **Enhanced Model**:

- Support for additional activation functions (e.g., Tanh, Sigmoid).

- Option for multiple hidden layers.

- **Regularization**:

- Implement dropout or L2 regularization.

- **Batch Training**:

- Support for mini-batch gradient descent.

- **Performance Optimization**:

- Parallelize matrix operations.

- **Testing**:

- Add unit tests for critical components.

- **Visualization**:

- Plot training loss and validation metrics.

- **Deployment**:

- Wrap the library in a web API for easy access.


## **Contributing**

Contributions are welcome! To get started:

1. Fork the repository.

2. Create a new branch: `git checkout -b feature-name`.

3. Commit your changes: `git commit -m 'Add feature-name'`.

4. Push to the branch: `git push origin feature-name`.

5. Open a pull request.


## **License**

This project is licensed under the MIT License.


**Thank you for exploring the Neural Network Library in Rust! If you have questions, feel free to raise an issue or contact me.**
