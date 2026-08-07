import os
import pickle
import joblib
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/predict", tags=["Mental Health Score Predictor"])

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "mental_health_model.pkl")
model = None

try:
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
        except Exception:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
        print("✅ [Mental Health] Model loaded successfully!")
    else:
        print(f"⚠️ [Mental Health] Model not found at {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ [Mental Health] Model loading error: {e}")

class MentalHealthInput(BaseModel):
    sleep_hours: float = Field(..., ge=0, le=24)
    work_hours: float = Field(..., ge=0, le=24)
    physical_activity_hours: float = Field(..., ge=0, le=24)
    stress_level: int = Field(..., ge=1, le=10)

@router.post("/mental-health")
def predict_mental_health(data: MentalHealthInput):
    global model
    
    score = (data.sleep_hours * 5) + (data.physical_activity_hours * 4) - (data.work_hours * 2) - (data.stress_level * 4) + 40
    score = max(5.0, min(98.0, score))
    
    if score >= 70:
        status = "EXCELLENT WELLBEING 🟢"
        risk = "LOW RISK"
    elif score >= 45:
        status = "MODERATE STRESS 🟡"
        risk = "MEDIUM RISK"
    else:
        status = "HIGH BURNOUT RISK 🔴"
        risk = "HIGH RISK"

    return {
        "wellbeing_score": round(score, 1),
        "status": status,
        "risk_level": risk,
        "recommendation": "Maintain sleep routine and regular physical breaks" if score >= 60 else "Consider taking time off to decompress and reduce stress"
    }
