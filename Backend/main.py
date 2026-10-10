from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(
    title="AI-Based Flood Prediction System",
    description="API for estimating flood probability from supplied dataset features",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "https://sumanruchi410-ui.github.io"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "flood_model.pkl"

saved_data = joblib.load(MODEL_PATH)
model = saved_data["model"]
features = saved_data["features"]


class FloodInput(BaseModel):
    features: dict[str, float]


@app.get("/")
def home():
    return {
        "message": "Flood Prediction API is running",
        "model_loaded": True
    }


@app.get("/features")
def get_features():
    return {
        "required_features": features
    }


@app.post("/predict")
def predict_flood(data: FloodInput):
    try:
        input_data = data.features

        missing = [f for f in features if f not in input_data]
        extra = [f for f in input_data if f not in features]

        if missing or extra:
            raise HTTPException(
                status_code=400,
                detail={
                    "missing_features": missing,
                    "unexpected_features": extra
                }
            )

        values = pd.DataFrame(
            [[input_data[f] for f in features]],
            columns=features
        )

        prediction = float(model.predict(values)[0])

        return {
            "estimated_flood_probability": prediction,
            "note": (
                "This is a model estimate based on the training dataset, "
                "not a validated real-world probability for an arbitrary location."
            )
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )