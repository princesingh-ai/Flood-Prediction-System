#!/usr/bin/env python3

"""FastAPI service for the AI-Based Flood Prediction System.

The trained model is loaded once at startup. The API exposes a health
check, the expected feature list, and a prediction endpoint.

Authors: Ruchi Suman, Shruti Singh, Hunny Pandey, Pranjal K. Sonawane
Copyright (c) 2026 Team Solvix. All rights reserved.
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="AI-Based Flood Prediction System",
    description="API for estimating flood probability from supplied dataset features",
    version="1.0",
)

# CORS is open because the frontend is often served from a different
# origin during local development, such as a Live Server on port 5500.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve the model path relative to this file so the API works no matter
# which directory uvicorn is started from.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "flood_model_small.pkl"

# The small model is the default deployment artifact; the full model is
# kept as a fallback for older checkouts.
if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR / "models" / "flood_model.pkl"

saved_data = joblib.load(MODEL_PATH)
model = saved_data["model"]
features = saved_data["features"]


class FloodInput(BaseModel):
    """Feature values keyed by the exact names used during training."""

    features: dict[str, float]


@app.get("/")
@app.get("/health")
@app.get("/api")
def home():
    """Health check used by the frontend and the deployment platform."""
    return {
        "message": "Flood Prediction API is running",
        "model_loaded": True,
    }


@app.get("/features")
def get_features():
    """Return the feature names the model expects, in training order."""
    return {
        "required_features": features,
    }


@app.post("/predict")
def predict_flood(data: FloodInput):
    """Estimate flood probability from a complete set of feature values."""
    try:
        input_data = data.features

        # The model expects a fixed column order, so requests with missing
        # or unknown features are rejected.
        missing = [f for f in features if f not in input_data]
        extra = [f for f in input_data if f not in features]

        if missing or extra:
            raise HTTPException(
                status_code=400,
                detail={
                    "missing_features": missing,
                    "unexpected_features": extra,
                },
            )

        values = pd.DataFrame(
            [[input_data[f] for f in features]],
            columns=features,
        )

        prediction = float(model.predict(values)[0])

        return {
            "estimated_flood_probability": prediction,
            "note": (
                "This is a model estimate based on the training dataset, "
                "not a validated real-world probability for an arbitrary location."
            ),
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
