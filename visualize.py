
import pandas as pd
import matplotlib.pyplot as plt

# Load predictions and actual values
data = pd.read_csv("predictions.csv", names=["Prediction", "Actual"])

# Plot the data
plt.plot(data["Prediction"], label="Prediction")
plt.plot(data["Actual"], label="Actual")
plt.legend()
plt.title("Predictions vs Actual Values")
plt.xlabel("Sample")
plt.ylabel("Value")
plt.show()