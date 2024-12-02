// This module defines the loss function used to evaluate the performance of the neural network.
// Loss functions measure the difference between the predicted outputs and the actual target values.

use ndarray::Array2; // Import for handling 2D arrays.

/// Computes the Mean Squared Error (MSE) between predictions and targets.
/// MSE is a common loss function for regression tasks, calculated as the average squared difference.
/// - `predictions`: The model's output for the given inputs.
/// - `targets`: The actual labels or ground truth values.
/// Returns the MSE as a single value.
pub fn mean_squared_error(predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
    let diff = predictions - targets; // Compute the element-wise difference between predictions and targets.
    diff.mapv(|x| x.powi(2)).mean().unwrap() // Square each difference, take the mean, and return the result.
}
