// This module provides utilities to load datasets from CSV files into ndarrays (multi-dimensional arrays).
// CSV is a common format for storing structured datasets.

use csv::Reader;        // Used to read data from CSV files.
use ndarray::Array2;    // Represents 2D arrays used for training and predictions.
use std::error::Error;  // Handles errors during file reading or parsing.

/// Loads a dataset from a CSV file and converts it into a 2D ndarray.
/// Each row in the CSV file corresponds to a data sample, and each column corresponds to a feature or label.
/// Why ndarray? It is optimized for numerical computations and is easy to manipulate.
pub fn load_csv(file_path: &str) -> Result<Array2<f64>, Box<dyn Error>> {
    // Initialize the CSV reader to read the file from the provided path.
    let mut reader = Reader::from_path(file_path)?;
    let mut records = Vec::new(); // To store rows as vectors of floats.

    // Iterate over each record (row) in the CSV file.
    for result in reader.records() {
        let record = result?; // Handle errors during record reading.
        let row: Vec<f64> = record
            .iter()
            .map(|field| field.parse::<f64>().unwrap()) // Convert string values to floats.
            .collect();
        records.push(row); // Add the parsed row to the records.
    }

    // Determine the number of rows and columns in the dataset.
    let rows = records.len();
    let cols = records[0].len();

    // Flatten the 2D vector into a single vector for ndarray creation.
    let flattened: Vec<f64> = records.into_iter().flatten().collect();

    // Convert the flattened vector into a 2D ndarray with the specified shape.
    Ok(Array2::from_shape_vec((rows, cols), flattened)?)
}
