#!/usr/bin/env python3
"""Train the Random Forest model used by the flood prediction API.

Run this script directly to rebuild the serialized model. All paths are
resolved relative to this file, so it works from any working directory.

Authors: Ruchi Suman, Shruti Singh, Hunny Pandey, Pranjal K. Sonawane
Copyright (c) 2026 Team Solvix. All rights reserved.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "Data" / "datasetFlood.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "flood_model_small.pkl"

# Load the dataset from the repository so every training run starts
# from the same source file.
data = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully:", data.shape)
print("Dataset columns:", data.columns.tolist())

# FloodProbability is the target; the remaining columns are the inputs.
X = data.drop("FloodProbability", axis=1)
y = data["FloodProbability"]

# The current dataset is fully numeric, but this step keeps the pipeline
# ready for categorical columns.
X = pd.get_dummies(X, drop_first=True)

# A fixed random state keeps the train/test split reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# A relatively small forest keeps the serialized model light enough to
# ship with the application.
model = RandomForestRegressor(
    n_estimators=20,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

print("\nTraining model...")
model.fit(X_train, y_train)

# Evaluate on the held-out split so the reported metrics reflect data
# the model did not see during training.
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel evaluation:")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")

# The API needs the feature order as well as the model, so both are
# stored in the same joblib file.
MODEL_DIR.mkdir(parents=True, exist_ok=True)

model_data = {
    "model": model,
    "features": X.columns.tolist(),
}

joblib.dump(model_data, MODEL_PATH)

print(f"\nModel saved successfully to: {MODEL_PATH}")
