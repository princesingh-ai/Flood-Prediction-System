Copyright (c) 2026 Team Solvix. All rights reserved.

Authors: Ruchi Suman, Shruti Singh, Hunny Pandey, Pranjal K. Sonawane

# AI-Based Flood Prediction System

A full-stack machine learning application that estimates flood probability from 20 environmental and infrastructure indicators. The project combines a FastAPI backend, a trained Random Forest regression model, and a browser-based frontend with interactive location search and map visualization.

> **Important:** This is an academic demonstration. The model output is a statistical estimate based on the training dataset, not a validated real-world flood probability. It must not be used for emergency response, safety, or infrastructure decisions.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model](#model)
- [API Reference](#api-reference)
- [Local Setup](#local-setup)
- [Running the Application](#running-the-application)
- [Deployment](#deployment)
- [Limitations](#limitations)
- [Authors](#authors)
- [License](#license)

## Overview

The system is made up of two independent parts:

1. **Backend:** A FastAPI service that loads a serialized scikit-learn model and exposes endpoints for feature discovery and flood probability prediction.
2. **Frontend:** A static HTML, CSS, and JavaScript application that lets users search for a place on an OpenStreetMap map and enter the 20 model features manually.

Location search and prediction are deliberately separate. Selecting a place only identifies its coordinates; the application does not retrieve weather, elevation, river, or drainage data for that place. The prediction is produced entirely from the feature values entered in the form.

## Key Features

- Flood probability estimation from 20 numerical features.
- Random Forest regression model trained on 50,000 records.
- FastAPI REST API with automatic Swagger documentation.
- Feature validation against the training schema.
- Interactive Leaflet map with Nominatim place search.
- Responsive frontend with no build step or JavaScript framework.
- Deployment configuration for Vercel with separate frontend and backend services.

## How It Works

The prediction flow is:

```text
Browser
  |
  |  POST /predict  { "features": { ... } }
  v
FastAPI backend
  |
  |  Validate feature names
  |  Build a DataFrame in training order
  v
Random Forest model (flood_model_small.pkl)
  |
  |  Predicted FloodProbability
  v
JSON response
  |
  v
Result panel in the browser
```

1. The user enters a value for each of the 20 features.
2. The frontend validates that every field contains a number and sends the values to `POST /predict`.
3. The backend checks that the request contains exactly the features used during training, in the expected schema.
4. The backend builds a one-row DataFrame in the original training column order.
5. The model returns an estimated `FloodProbability`.
6. The frontend displays the value and a proportional bar. The bar is a visual aid only, not a calibrated risk score.

The location search flow is independent:

1. The user enters a place name.
2. The frontend sends a request to the public Nominatim geocoding service.
3. The map centers on the returned coordinates and displays a marker.
4. The selected place name, latitude, and longitude are shown in the location panel.

## Technology Stack

| Layer | Technologies |
| --- | --- |
| Backend | Python, FastAPI, Pydantic, pandas, joblib, Uvicorn |
| Dependency management | uv, pyproject.toml, uv.lock |
| Machine learning | scikit-learn, Random Forest regression |
| Frontend | HTML, CSS, vanilla JavaScript |
| Mapping | Leaflet 1.9.4, OpenStreetMap, Nominatim |
| Deployment | Vercel services, GitHub Pages (static frontend) |

## Project Structure

```text
Flood-Prediction-System/
├── Backend/
│   ├── Data/
│   │   └── datasetFlood.csv
│   ├── models/
│   │   └── flood_model_small.pkl
│   ├── .python-version
│   ├── __init__.py
│   ├── main.py
│   ├── pyproject.toml
│   ├── train_model.py
│   └── uv.lock
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
├── index.html
├── script.js
├── style.css
├── vercel.json
└── README.md
```

- `Backend/main.py` contains the FastAPI application and prediction endpoint.
- `Backend/train_model.py` trains and serializes the model.
- `Backend/pyproject.toml` declares the backend dependencies, and `Backend/uv.lock` pins their exact versions.
- `Backend/.python-version` selects the Python version used by uv.
- `Backend/models/flood_model_small.pkl` stores the model and its feature order.
- `frontend/` is the frontend root used by the Vercel deployment.
- The files at the repository root are a static copy of the frontend for hosting environments such as GitHub Pages. They are currently identical to the files in `frontend/`, so changes should be made in both places.
- `vercel.json` defines the frontend and backend services and the request rewrites.

## Dataset

The training data is stored at `Backend/Data/datasetFlood.csv`.

- 50,000 records.
- 20 numerical input features.
- One target column: `FloodProbability`.
- Feature values in the supplied dataset range from 0 to 22.
- Target values range from 0.285 to 0.725, with a mean of approximately 0.5.

| Feature | Interpretation |
| --- | --- |
| `MonsoonIntensity` | Intensity of monsoon rainfall |
| `TopographyDrainage` | Drainage capacity of the terrain |
| `RiverManagement` | Quality of river management |
| `Deforestation` | Deforestation pressure |
| `Urbanization` | Urban development pressure |
| `ClimateChange` | Climate change impact |
| `DamsQuality` | Condition and quality of dams |
| `Siltation` | Silt accumulation |
| `AgriculturalPractices` | Impact of agricultural practices |
| `Encroachments` | Encroachment on floodplains or water bodies |
| `IneffectiveDisasterPreparedness` | Gaps in disaster preparedness |
| `DrainageSystems` | Condition of drainage infrastructure |
| `CoastalVulnerability` | Exposure of coastal areas |
| `Landslides` | Landslide susceptibility |
| `Watersheds` | Condition of watersheds |
| `DeterioratingInfrastructure` | Decay of infrastructure |
| `PopulationScore` | Population pressure |
| `WetlandLoss` | Loss of wetlands |
| `InadequatePlanning` | Gaps in urban and land-use planning |
| `PoliticalFactors` | Political and institutional factors |

The interpretations above describe the intended meaning of each column. The model itself only learns numerical relationships from the supplied values.

## Model

The model is a `RandomForestRegressor` trained with the following configuration:

| Parameter | Value |
| --- | --- |
| `n_estimators` | 20 |
| `max_depth` | 12 |
| `min_samples_leaf` | 2 |
| `random_state` | 42 |
| `n_jobs` | -1 |

Training details:

- The target column is `FloodProbability`; all other columns are used as inputs.
- The data is split into 80 percent training and 20 percent testing sets with a fixed random state.
- `pd.get_dummies(..., drop_first=True)` is applied so the pipeline can handle categorical columns if they are added later. The current dataset is fully numeric, so this step does not change the feature set.
- The trained model and the ordered feature list are saved together in `Backend/models/flood_model_small.pkl` using joblib.
- The API loads this file once at startup and reuses the model for every request.
- scikit-learn is pinned to `1.9.1` in `Backend/pyproject.toml` because the serialized model was trained with that version. If you retrain the model with a different scikit-learn version, update the pin and run `uv lock` from the `Backend` directory.

Reported performance on the current test split:

| Metric | Value |
| --- | --- |
| R2 score | approximately 0.73 |
| Mean Absolute Error | approximately 0.0205 |
| Mean Squared Error | approximately 0.0007 |

These values describe performance on this dataset only. They do not establish accuracy for real locations or future flood events.

## API Reference

Local base URL: `http://127.0.0.1:8000`

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Health check |
| `GET` | `/health` | Health check |
| `GET` | `/api` | Health check |
| `GET` | `/features` | List the required feature names |
| `POST` | `/predict` | Estimate flood probability |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc API documentation |

### GET /features

The response is a JSON object with a `required_features` array containing all 20 feature names in the order used during training.

### POST /predict

The request body must contain exactly the 20 required features:

```json
{
  "features": {
    "MonsoonIntensity": 3,
    "TopographyDrainage": 8,
    "RiverManagement": 6,
    "Deforestation": 6,
    "Urbanization": 4,
    "ClimateChange": 4,
    "DamsQuality": 6,
    "Siltation": 2,
    "AgriculturalPractices": 3,
    "Encroachments": 2,
    "IneffectiveDisasterPreparedness": 5,
    "DrainageSystems": 10,
    "CoastalVulnerability": 7,
    "Landslides": 4,
    "Watersheds": 2,
    "DeterioratingInfrastructure": 3,
    "PopulationScore": 4,
    "WetlandLoss": 3,
    "InadequatePlanning": 2,
    "PoliticalFactors": 6
  }
}
```

Illustrative response:

```json
{
  "estimated_flood_probability": 0.45,
  "note": "This is a model estimate based on the training dataset, not a validated real-world probability for an arbitrary location."
}
```

If features are missing or unexpected feature names are supplied, the API returns HTTP 400 with the offending names.

Save the request body above as `request.json`, then run:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @request.json
```

## Local Setup

### Prerequisites

- Python 3.12 or newer. uv installs the version pinned in `Backend/.python-version` if it is not already available.
- [uv](https://docs.astral.sh/uv/) for Python dependency management.
- A modern web browser.
- No Node.js installation is required.

### Install the backend dependencies

```bash
cd Backend
uv sync
```

This creates a `.venv` directory and installs the dependencies pinned in `uv.lock`. There is no need to activate the environment manually. If another virtual environment is active, deactivate it first; uv uses `Backend/.venv` as the project environment.

Dependencies are declared in `Backend/pyproject.toml`. From the `Backend` directory, use `uv add <package>` and `uv remove <package>` to change them; uv updates both `pyproject.toml` and `uv.lock`.

### Model file

The repository includes `Backend/models/flood_model_small.pkl`, so the API can start without retraining.

To rebuild the model from the dataset:

```bash
cd Backend
uv run python train_model.py
```

The script reads `Backend/Data/datasetFlood.csv` and overwrites `Backend/models/flood_model_small.pkl`.

## Running the Application

Start the backend from the `Backend` directory:

```bash
uv run uvicorn main:app --reload
```

The API is available at `http://127.0.0.1:8000`. Interactive documentation is available at `http://127.0.0.1:8000/docs`.

Serve the frontend from the `frontend` directory:

```bash
cd frontend
python3 -m http.server 5500
```

Open `http://localhost:5500`.

When the page is served from `localhost` or `127.0.0.1` on a port other than 8000, the frontend automatically sends API requests to `http://127.0.0.1:8000`. In production, it uses same-origin paths.

## Deployment

### Vercel

The repository includes a `vercel.json` configuration with two services:

- `app`: the static frontend in `frontend/`.
- `backend`: the FastAPI application in `Backend/`, with `main.py` as the entry point.

The configuration rewrites `/predict`, `/features`, `/health`, and `/api/*` to the backend service. All other requests are served by the frontend.

Vercel uses uv automatically when it finds the backend's `pyproject.toml` and `uv.lock`, and reads `Backend/.python-version` for the Python version. The deployment therefore installs the same locked dependencies as local development.

The serialized model must be present in the deployment. If the model is retrained, make sure the updated `.pkl` file is included in the deployed revision.

### GitHub Pages

The files at the repository root can be served as a static site by GitHub Pages. GitHub Pages cannot host the FastAPI backend, so prediction requests require a separately hosted API. The frontend must also be pointed at that API by updating `API_URL` in `script.js`.

## Limitations

- The application does not use live weather, river level, elevation, satellite, or sensor data.
- Location search only geocodes a place name. It does not collect environmental data for the selected coordinates.
- Predictions depend entirely on the 20 feature values entered by the user.
- The model is trained on a fixed dataset and may not generalize to other regions or future conditions.
- The output is not a calibrated probability and should not be treated as a flood warning.
- The API has no authentication or rate limiting and allows all origins for demonstration purposes.
- The root frontend copy and the `frontend/` copy must be kept in sync manually.

## Authors

- Ruchi Suman
- Shruti Singh
- Hunny Pandey
- Pranjal K. Sonawane

**Team:** Solvix

## License

This project is provided for academic and demonstration purposes. Redistribution or commercial use requires permission from Team Solvix.

---

Copyright (c) 2026 Team Solvix. All rights reserved.

Authors: Ruchi Suman, Shruti Singh, Hunny Pandey, Pranjal K. Sonawane
