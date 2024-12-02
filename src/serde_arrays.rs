// This module provides utilities for serializing and deserializing 2D ndarrays (Array2).
// It ensures that ndarray types can be saved and loaded from files in JSON format.

use ndarray::Array2;                // 2D array support.
use serde::{Deserialize, Deserializer, Serialize, Serializer}; // Serialization and deserialization support.

/// Custom serialization function for Array2.
/// Converts the array into a serializable form by flattening it.
pub fn serialize<S>(array: &Array2<f64>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let shape = array.shape().to_vec(); // Get the shape of the array as a vector.
    let data = array.iter().cloned().collect::<Vec<f64>>(); // Flatten the array into a 1D vector.
    (&shape, &data).serialize(serializer) // Serialize the shape and data as a tuple.
}

/// Custom deserialization function for Array2.
/// Reconstructs the array from its shape and flattened data.
pub fn deserialize<'de, D>(deserializer: D) -> Result<Array2<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    let (shape, data): (Vec<usize>, Vec<f64>) = Deserialize::deserialize(deserializer)?; // Deserialize the shape and data.
    Array2::from_shape_vec((shape[0], shape[1]), data).map_err(serde::de::Error::custom) // Reconstruct the array.
}
