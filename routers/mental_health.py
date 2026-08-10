import os
import pickle
import joblib
import pandas as pd
from typing import Optional, Literal
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/predict", tags=["Mental Health Score Predictor"])

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "mental_health_model.pkl")
model = None

top_countries = ['Other', 'India', 'USA', 'Canada', 'Australia', 'UK', 'Germany', 'Mexico', 'Turkey', 'France']

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
    # Full Student Habits Schema (from 04-mental-health-score-predictor UI)
    age: Optional[int] = Field(21, ge=10, le=100)
    gender: Optional[str] = 'Female'
    country: Optional[str] = 'India'
    academic_level: Optional[str] = 'Undergraduate'
    most_used_platform: Optional[str] = 'Instagram'
    purpose_of_use: Optional[str] = 'Entertainment'
    avg_daily_usage_hours: Optional[float] = Field(4.5, ge=0, le=24)
    daily_unlocks: Optional[int] = Field(65, ge=0)
    study_hours: Optional[float] = Field(5.0, ge=0, le=24)
    physical_activity_hours: Optional[float] = Field(1.0, ge=0, le=24)
    sleep_hours_per_night: Optional[float] = Field(7.0, ge=0, le=24)
    stress_level: Optional[str] = 'Medium'
    
    # Alternate field aliases for legacy callers
    sleep_hours: Optional[float] = None
    work_hours: Optional[float] = None


@router.post("/mental-health")
def predict_mental_health(data: MentalHealthInput):
    global model

    sleep = data.sleep_hours_per_night if data.sleep_hours_per_night is not None else (data.sleep_hours or 7.0)
    study = data.study_hours if data.study_hours is not None else (data.work_hours or 5.0)

    # 1. Try real ML model prediction if available
    if model is not None:
        try:
            c_group = data.country if data.country in top_countries else "Other"
            input_df = pd.DataFrame([{
                'Age': data.age,
                'Gender': data.gender,
                'Country': data.country,
                'Academic_Level': data.academic_level,
                'Most_Used_Platform': data.most_used_platform,
                'Purpose_Of_Use': data.purpose_of_use,
                'Avg_Daily_Usage_Hours': data.avg_daily_usage_hours,
                'Daily_Unlocks': data.daily_unlocks,
                'Study_Hours': study,
                'Physical_Activity_Hours': data.physical_activity_hours,
                'Sleep_Hours_Per_Night': sleep,
                'Stress_Level': data.stress_level,
                'Grouped_country': c_group
            }])
            prediction = model.predict(input_df)[0]
            score_val = round(float(prediction), 2)
            return {
                "predicted_mental_health_score": score_val,
                "wellbeing_score": score_val
            }
        except Exception as err:
            print(f"⚠️ [Mental Health] Model predict error: {err}")

    # 2. Heuristic fallback computation
    calc_score = 8.5
    st = str(data.stress_level).lower()
    if 'very high' in st:
        calc_score -= 2.5
    elif 'high' in st:
        calc_score -= 1.8
    elif 'medium' in st:
        calc_score -= 0.8

    if sleep < 6:
        calc_score -= 1.5
    if data.avg_daily_usage_hours and data.avg_daily_usage_hours > 6:
        calc_score -= 1.2
    if data.physical_activity_hours and data.physical_activity_hours >= 1:
        calc_score += 0.8

    final_score = round(max(1.0, min(10.0, calc_score)), 2)
    return {
        "predicted_mental_health_score": final_score,
        "wellbeing_score": final_score
    }
