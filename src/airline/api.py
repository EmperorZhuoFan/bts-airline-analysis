
"""FastAPI application for BTS airline departure-delay prediction."""

from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote
import logging

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_NAME = "BTS_Airline_GradientBoosting"
MODEL_VERSION = "2"
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

logger = logging.getLogger(__name__)

NUMERICAL_FEATURES = [
    "YEAR",
    "QUARTER",
    "MONTH",
    "DAY_OF_WEEK",
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "DISTANCE",
    "DEP_HOUR",
]

CATEGORICAL_FEATURES = [
    "OP_UNIQUE_CARRIER",
    "TAIL_NUM",
    "ORIGIN",
    "ORIGIN_CITY_NAME",
    "ORIGIN_STATE_ABR",
    "DEST",
    "DEST_CITY_NAME",
    "DEST_STATE_ABR",
    "TIME_OF_DAY",
    "ROUTE",
    "DISTANCE_CATEGORY",
]

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# ---------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------

class FlightFeatures(BaseModel):
    """Input features required by the trained model."""

    model_config = ConfigDict(extra="forbid")

    YEAR: int = Field(..., ge=1900)
    QUARTER: int = Field(..., ge=1, le=4)
    MONTH: int = Field(..., ge=1, le=12)
    DAY_OF_WEEK: int = Field(..., ge=1, le=7)

    CRS_DEP_TIME: float = Field(..., ge=0)
    CRS_ARR_TIME: float = Field(..., ge=0)
    DISTANCE: float = Field(..., ge=0)
    DEP_HOUR: int = Field(..., ge=0, le=23)

    OP_UNIQUE_CARRIER: str
    TAIL_NUM: str
    ORIGIN: str
    ORIGIN_CITY_NAME: str
    ORIGIN_STATE_ABR: str
    DEST: str
    DEST_CITY_NAME: str
    DEST_STATE_ABR: str
    TIME_OF_DAY: str
    ROUTE: str
    DISTANCE_CATEGORY: str


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

model = None


def get_tracking_uri() -> str:
    """Build the MLflow SQLite tracking URI from the project root."""

    database_path = Path(__file__).resolve().parents[2] / "mlflow.db"

    return f"sqlite:///{quote(database_path.as_posix(), safe='/:')}"


def load_model() -> None:
    """Load the registered model once when the API starts."""

    global model

    mlflow.set_tracking_uri(get_tracking_uri())

    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
        logger.info("Loaded model: %s", MODEL_URI)

    except Exception:
        logger.exception("Failed to load registered MLflow model.")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    load_model()
    yield


app = FastAPI(
    title="BTS Airline Delay Prediction API",
    description=(
        "Predicts whether a flight will experience a departure delay "
        "using the registered BTS airline Gradient Boosting model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------

@app.get("/")
def home():
    """Return basic API information."""

    return {
        "message": "BTS Airline Delay Prediction API",
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "status": "running",
    }


@app.get("/health")
def health_check():
    """Check whether the API and model are ready."""

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
    }



@app.post("/predict")
def predict(flight: FlightFeatures):
    """Predict the departure-delay class for one flight."""

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is not available.",
        )

    try:
        input_data = pd.DataFrame(
            [flight.model_dump()],
            columns=FEATURE_COLUMNS,
        )

        prediction = int(model.predict(input_data)[0])

        prediction_labels = {
            0: "Undelayed",
            1: "Delayed",
        }

        if prediction not in prediction_labels:
            raise ValueError(f"Unexpected prediction class: {prediction}")

        return {
            "prediction": prediction,
            "prediction_label": prediction_labels[prediction],
            "model": MODEL_NAME,
            "model_version": MODEL_VERSION,
        }

    except Exception:
        logger.exception("Prediction failed.")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server logs.",
        )
