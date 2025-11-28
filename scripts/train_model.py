from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
print("Loading dataset...")
X, y = fetch_california_housing(return_X_y=True)

# Split data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training model...")
model = LinearRegression()
model.fit(X_train, y_train)

import os

# Save trained model to file
print("Saving model to house_model.pkl...")
output_path = os.path.join(os.path.dirname(__file__), "../app/house_model.pkl")
joblib.dump(model, output_path)
print("Done!")
