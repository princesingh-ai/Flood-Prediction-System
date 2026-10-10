import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "Data" / "datasetFlood.csv"

data = pd.read_csv(file_path)

print("Dataset loaded successfully:", data.shape)
print("Dataset columns:", data.columns.tolist())

print("Dataset loaded:", data.shape)

# Separate input features and target
X = data.drop("FloodProbability", axis=1)
y = data["FloodProbability"]

# Convert categorical columns if present
X = pd.get_dummies(X, drop_first=True)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Create and train the model
model = RandomForestRegressor(
    n_estimators=20,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n===== MODEL EVALUATION =====")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")

# Save model and feature columns
os.makedirs("backend/models", exist_ok=True)

# Save the trained model and feature names

MODEL_PATH = BASE_DIR / "models" / "flood_model_small.pkl"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

model_data = {
    "model": model,
    "features": X.columns.tolist()
}

model_path = MODEL_DIR / "flood_model_small.pkl"

joblib.dump(model_data, model_path)

print(f"\nModel saved successfully to: {model_path}")

