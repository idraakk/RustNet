# RustNet: Neural Networks Library ( WORK IN PROGRESS... )
## ( SEMESTER 5 DSA PROJECT )
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
