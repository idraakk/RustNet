# RustNet: Neural Networks Library ( WORK IN PROGRESS... )
## Project Description

**RustNet** is a neural networks library developed from scratch in Rust. It incorporates advanced data structures and algorithms to support efficient machine learning computations. The library leverages Rust's performance and safety features while implementing core neural network functionalities.

## Key Features

- **Tensor Operations**: Implemented a custom tensor class using dynamic arrays to efficiently handle multi-dimensional data and perform matrix operations, including multiplication, addition, and transposition.
- **Graph Data Structures**: Designed and utilized Directed Acyclic Graphs (DAGs) to model the computational flow of neural networks, facilitating the forward and backward passes through various layers.
- **Neural Network Layers**: Developed core layer types, including Dense (Fully Connected) and Activation Layers, using matrix operations and efficient algorithms to compute layer outputs and gradients.
- **Activation Functions**: Implemented essential activation functions (ReLU, Sigmoid) as element-wise operations on tensors, applying concepts from numerical methods and functional programming.
- **Backpropagation**: Employed dynamic programming techniques to efficiently compute gradients during backpropagation, optimizing memory usage and computational efficiency.
- **Optimization Algorithms**: Integrated basic gradient descent and its variants, such as Stochastic Gradient Descent (SGD), leveraging efficient algorithms for parameter updates and convergence.

## Technologies Used

- **Programming Language**: Rust
- **Libraries**: `ndarray` for tensor operations, `nalgebra` for linear algebra

## DSA Concepts Applied

- **Matrix Operations**: Efficient implementation of matrix multiplication and other linear algebra operations using custom algorithms and data structures.
- **Dynamic Programming**: Utilized for optimizing backpropagation and gradient computation, reducing redundant calculations and improving performance.
- **Graph Theory**: Applied DAGs to manage the computational flow and dependencies in neural networks, ensuring accurate and efficient data processing.
- **Hash Tables**: Used for managing and accessing parameters and gradients during training, ensuring quick updates and retrieval.

## Impact

- **Performance**: Achieved high computational efficiency and memory safety with Rust’s low-level control and advanced DSA techniques.
- **Scalability**: Designed to support various neural network architectures and datasets, providing a robust foundation for further enhancements and real-world applications.


# Setting up the Rust Project
First, create a new Rust project:
```bash
cargo new neural_network
cd neural_network
```
Edit the Cargo.toml file to add the necessary dependencies. We’ll use the rand crate to initialize weights and biases with random values:
```toml
[dependencies]
rand = "0.8.5"
```
Now, let’s write the code for the neural network.

# Step 1: Define the Neural Network Structure
Create a main.rs file inside the src folder and define the basic structure of the network:
```rust
extern crate rand;
use rand::Rng;

struct NeuralNetwork {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    weights_ih: Vec<Vec<f64>>, // Weights between input and hidden layers
    weights_ho: Vec<Vec<f64>>, // Weights between hidden and output layers
    biases_h: Vec<f64>,        // Biases for hidden layer
    biases_o: Vec<f64>,        // Biases for output layer
}

impl NeuralNetwork {
    // Initialize the neural network with random weights and biases
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> NeuralNetwork {
        let mut rng = rand::thread_rng();

        let weights_ih: Vec<Vec<f64>> = (0..input_size)
            .map(|_| (0..hidden_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        
        let weights_ho: Vec<Vec<f64>> = (0..hidden_size)
            .map(|_| (0..output_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();

        let biases_h: Vec<f64> = (0..hidden_size).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let biases_o: Vec<f64> = (0..output_size).map(|_| rng.gen_range(-1.0..1.0)).collect();

        NeuralNetwork {
            input_size,
            hidden_size,
            output_size,
            weights_ih,
            weights_ho,
            biases_h,
            biases_o,
        }
    }
}
```
This initializes a neural network with random weights and biases.

# Step 2: Implement the Activation Functions
Let’s implement some basic activation functions: ReLU for the hidden layer and Sigmoid for the output layer.
```rust
fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
```
# Step 3: Forward Propagation
Next, implement the forward propagation method. The data will pass from the input layer to the hidden layer and then to the output layer.
```rust
impl NeuralNetwork {
    fn forward(&self, input: Vec<f64>) -> (Vec<f64>, Vec<f64>) {
        // Calculate hidden layer activations
        let hidden_activations: Vec<f64> = (0..self.hidden_size)
            .map(|i| {
                let weighted_sum: f64 = input.iter()
                    .zip(self.weights_ih.iter().map(|row| row[i]))
                    .map(|(x, w)| x * w)
                    .sum::<f64>() + self.biases_h[i];
                relu(weighted_sum)
            })
            .collect();

        // Calculate output layer activations
        let output_activations: Vec<f64> = (0..self.output_size)
            .map(|i| {
                let weighted_sum: f64 = hidden_activations.iter()
                    .zip(self.weights_ho.iter().map(|row| row[i]))
                    .map(|(x, w)| x * w)
                    .sum::<f64>() + self.biases_o[i];
                sigmoid(weighted_sum)
            })
            .collect();

        (hidden_activations, output_activations)
    }
}
```
# Step 5: Backpropagation and Training
Next, implement backpropagation to update the weights and biases based on the error and gradients calculated.
```rust
impl NeuralNetwork {
    fn train(&mut self, input: Vec<f64>, target: Vec<f64>, learning_rate: f64) {
        let (hidden_activations, output_activations) = self.forward(input.clone());

        // Output layer error and gradients
        let output_errors: Vec<f64> = output_activations.iter()
            .zip(target.iter())
            .map(|(o, t)| t - o)
            .collect();

        let output_gradients: Vec<f64> = output_activations.iter()
            .zip(output_errors.iter())
            .map(|(o, e)| e * o * (1.0 - o)) // derivative of sigmoid
            .collect();

        // Hidden layer error and gradients
        let hidden_errors: Vec<f64> = (0..self.hidden_size)
            .map(|i| self.weights_ho[i].iter()
                .zip(output_gradients.iter())
                .map(|(w, g)| w * g)
                .sum())
            .collect();

        let hidden_gradients: Vec<f64> = hidden_activations.iter()
            .zip(hidden_errors.iter())
            .map(|(h, e)| if *h > 0.0 { *e } else { 0.0 }) // derivative of ReLU
            .collect();

        // Update weights and biases for hidden -> output
        for i in 0..self.hidden_size {
            for j in 0..self.output_size {
                self.weights_ho[i][j] += learning_rate * output_gradients[j] * hidden_activations[i];
            }
        }
        for j in 0..self.output_size {
            self.biases_o[j] += learning_rate * output_gradients[j];
        }

        // Update weights and biases for input -> hidden
        for i in 0..self.input_size {
            for j in 0..self.hidden_size {
                self.weights_ih[i][j] += learning_rate * hidden_gradients[j] * input[i];
            }
        }
        for j in 0..self.hidden_size {
            self.biases_h[j] += learning_rate * hidden_gradients[j];
        }
    }
}
```
# Step 6: Training and Testing the Network
Now, let’s create a function to train the network using a dataset and a simple loop to iterate over it.
```rust
fn main() {
    // Create the neural network (2 inputs, 3 hidden nodes, 1 output)
    let mut nn = NeuralNetwork::new(2, 3, 1);

    // Training data (XOR problem)
    let data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    // Train the network
    for epoch in 0..5000 {
        let mut total_loss = 0.0;
        for (input, target) in &data {
            nn.train(input.clone(), target.clone(), 0.1);
            let output = nn.forward(input.clone()).1;
            total_loss += mse_loss(&output, target);
        }
        if epoch % 1000 == 0 {
            println!("Epoch {}: Loss = {}", epoch, total_loss / data.len() as f64);
        }
    }

    // Test the network
    for (input, _) in &data {
        let output = nn.forward(input.clone()).1;
        println!("Input: {:?}, Output: {:?}", input, output);
    }
}
```
# Running the Code
You can now run your neural network with:
```bash
cargo run
```
# Explanation of the Training:
- The network is trained on the XOR problem, a standard problem for basic neural networks.
- We use forward propagation to compute the output.
- Backpropagation adjusts the weights and biases using gradient descent to minimize the error.
- The training loop runs for a specified number of epochs, printing the loss periodically.
# Summary:
- Input Layer: Takes two inputs (representing the XOR inputs).
- Hidden Layer: Contains three neurons using ReLU activation.
- Output Layer: Contains one
