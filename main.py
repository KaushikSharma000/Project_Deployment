from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import pandas as pd
import numpy as np
import os

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Machine Failure Prediction API",
    description="Predicts whether a machine will fail based on sensor readings.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "machine_failure_model.pkl")
model = joblib.load(MODEL_PATH)

# ── Schemas ────────────────────────────────────────────────────────────────────
class MachineInput(BaseModel):
    time_cycle_count: Optional[float] = Field(None, example=6151.12, description="Time or cycle count of the machine")
    temperature_c: Optional[float]    = Field(None, example=74.13,   description="Temperature in Celsius")
    pressure_bar: Optional[float]     = Field(None, example=165.51,  description="Pressure in bar")
    vibration_mm_s: Optional[float]   = Field(None, example=2.21,    description="Vibration in mm/s")
    speed_rpm: Optional[float]        = Field(None, example=1489.87, description="Speed in RPM")
    torque_nm: Optional[float]        = Field(None, example=208.19,  description="Torque in Nm")
    operational_mode: str             = Field(...,  example="Normal", description="One of: Normal, Idle, Overload, Maintenance")

class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    failure_probability: float
    safe_probability: float
    confidence: str

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Machine Failure Prediction API is running."}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: MachineInput):
    try:
        input_df = pd.DataFrame([{
            "Time/Cycle Count":  data.time_cycle_count,
            "Temperature (°C)":  data.temperature_c,
            "Pressure (bar)":    data.pressure_bar,
            "Vibration (mm/s)":  data.vibration_mm_s,
            "Speed (RPM)":       data.speed_rpm,
            "Torque (Nm)":       data.torque_nm,
            "Operational Mode":  data.operational_mode,
        }])

        prediction  = int(model.predict(input_df)[0])
        probability = model.predict_proba(input_df)[0]
        fail_prob   = round(float(probability[1]), 4)
        safe_prob   = round(float(probability[0]), 4)

        if fail_prob >= 0.75:
            confidence = "High"
        elif fail_prob >= 0.50:
            confidence = "Medium"
        else:
            confidence = "Low"

        return PredictionResponse(
            prediction=prediction,
            prediction_label="Machine Failure" if prediction == 1 else "No Failure",
            failure_probability=fail_prob,
            safe_probability=safe_prob,
            confidence=confidence,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(records: list[MachineInput]):
    if len(records) > 500:
        raise HTTPException(status_code=400, detail="Batch size must be ≤ 500.")
    return [predict(r) for r in records]
