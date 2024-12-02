import numpy as np  # NumPy is used for numerical computations
import pandas as pd  # Pandas is used for data manipulation and saving as CSV

# Step 1: Simulate the frequency range for RF waves
# We are generating 1000 points equally spaced between 0 and 100
frequency = np.linspace(0, 100, 1000)

# Step 2: Simulate amplitude values
# The amplitude is modeled as a sine wave with some random noise added
# - np.sin(frequency): Generates the sine wave
# - np.random.normal(0, 0.1, size=1000): Adds random noise with mean 0 and standard deviation 0.1
amplitude = np.sin(frequency) + np.random.normal(0, 0.1, size=1000)

# Step 3: Create labels based on the amplitude values
# If the amplitude is positive, we assign the label 1, otherwise 0
labels = (amplitude > 0).astype(int)

# Step 4: Combine frequency, amplitude, and labels into a single DataFrame
# A DataFrame is a table-like structure provided by Pandas
df = pd.DataFrame({
    'Frequency': frequency,  # First column: frequency values
    'Amplitude': amplitude,  # Second column: amplitude values (sine wave with noise)
    'Label': labels          # Third column: binary labels (1 if amplitude > 0, else 0)
})

# Step 5: Save the DataFrame to a CSV file
# The file will be named `rf_data.csv`, and the index is excluded for simplicity
df.to_csv('rf_data.csv', index=False)

# Print a message to confirm that the dataset has been saved successfully
print("Dataset saved as rf_data.csv")
