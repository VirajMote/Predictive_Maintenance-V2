"""
api.py
──────
FastAPI inference server.

Endpoints:
  POST /predict          — single prediction from raw sensor readings
  POST /predict/batch    — batch predictions
  GET  /health           — server + model health check
  GET  /model/info       — loaded model metadata

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from config import MODEL_PKG_PATH, SENSOR_COLS
from inference import PredictiveMaintenanceInference, PredictionResult

logger = logging.getLogger(__name__)

# ── App State ─────────────────────────────────────────────────────────────────

engine: PredictiveMaintenanceInference | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    global engine
    if MODEL_PKG_PATH.exists():
        engine = PredictiveMaintenanceInference.from_package(MODEL_PKG_PATH)
        logger.info(f"Model loaded: {engine.pkg.mode}")
    else:
        logger.warning(
            f"Model package not found at {MODEL_PKG_PATH}. "
            "Run train.py first. /predict will return 503."
        )
    yield
    engine = None


app = FastAPI(
    title="Predictive Maintenance API",
    description="Failure prediction, RUL estimation, and health scoring for industrial machinery.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────

class SensorReading(BaseModel):
    """A single timestep of sensor data."""
    type_code:            str   = Field(..., description="Machine type: L, M, or H")
    air_temperature_K:    float = Field(..., ge=290, le=320, description="Air temperature [K]")
    process_temperature_K: float = Field(..., ge=300, le=320, description="Process temperature [K]")
    rotational_speed_rpm: float = Field(..., ge=1000, le=3000, description="Rotational speed [rpm]")
    torque_Nm:            float = Field(..., ge=0, le=80, description="Torque [Nm]")
    tool_wear_min:        float = Field(..., ge=0, le=300, description="Tool wear [min]")

    @field_validator("type_code")
    @classmethod
    def validate_type(cls, v: str) -> str:
        v = v.upper()
        if v not in ("L", "M", "H"):
            raise ValueError("type_code must be L, M, or H")
        return v

    def to_dataframe_row(self) -> dict:
        return {
            "Type":                       self.type_code,
            "Air temperature [K]":        self.air_temperature_K,
            "Process temperature [K]":    self.process_temperature_K,
            "Rotational speed [rpm]":     self.rotational_speed_rpm,
            "Torque [Nm]":                self.torque_Nm,
            "Tool wear [min]":            self.tool_wear_min,
        }


class PredictRequest(BaseModel):
    """
    Single-point or window prediction request.
    For LSTM models, pass a list of readings (window).
    For RF/XGB, a single reading is sufficient.
    """
    readings:               list[SensorReading] = Field(..., min_length=1)
    cycle_duration_seconds: float | None = Field(
        None, description="If provided, converts RUL cycles to hours."
    )


class BatchPredictRequest(BaseModel):
    """Batch of independent prediction requests."""
    requests: list[PredictRequest] = Field(..., min_length=1, max_length=100)


class PredictResponse(BaseModel):
    failure_probabilities: dict[str, float]
    failure_predictions:   dict[str, int]
    rul_cycles:            float
    rul_hours:             float | None
    health_score:          float
    health_label:          str
    alerts:                list[str]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _readings_to_df(readings: list[SensorReading]) -> pd.DataFrame:
    rows = [r.to_dataframe_row() for r in readings]
    return pd.DataFrame(rows)


def _check_engine():
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train.py to generate the model package.",
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest):
    """
    Predict failure probability, RUL, and health score for a machine.

    For RF/XGB: send a single reading (or recent history — last row is used).
    For LSTM:   send at least 30 readings (one full sequence window).
    """
    _check_engine()
    try:
        df     = _readings_to_df(request.readings)
        result = engine.predict(df, request.cycle_duration_seconds)
        return PredictResponse(**result.to_dict())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


@app.post("/predict/batch", response_model=list[PredictResponse], tags=["Inference"])
async def predict_batch(request: BatchPredictRequest):
    """Batch inference — up to 100 machines per call."""
    _check_engine()
    results = []
    for req in request.requests:
        try:
            df     = _readings_to_df(req.readings)
            result = engine.predict(df, req.cycle_duration_seconds)
            results.append(PredictResponse(**result.to_dict()))
        except Exception as e:
            logger.warning(f"Batch item failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    return results


@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status":       "ok" if engine is not None else "model_not_loaded",
        "model_loaded": engine is not None,
        "model_type":   engine.pkg.mode if engine else None,
    }


@app.get("/model/info", tags=["System"])
async def model_info():
    _check_engine()
    return {
        "mode":           engine.pkg.mode,
        "n_features":     len(engine.pkg.feature_cols),
        "feature_sample": engine.pkg.feature_cols[:10],
        "labels":         list(engine.pkg.classifiers.keys())
                          if isinstance(engine.pkg.classifiers, dict) else ["LSTM multi-label"],
        "thresholds":     engine.pkg.thresholds,
    }
