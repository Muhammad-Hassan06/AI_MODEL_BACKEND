import os
import pickle
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/predict", tags=["House Price Prediction"])

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "house_model.pkl")
model = None

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("✅ [House Price] Model loaded successfully!")
    else:
        print(f"⚠️ [House Price] Model not found at {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ [House Price] Model loading error: {e}")

class HouseInput(BaseModel):
    MedInc: float = Field(..., example=5.2)
    HouseAge: float = Field(..., example=25.0)
    AveRooms: float = Field(..., example=5.8)
    AveBedrms: float = Field(..., example=1.1)
    Population: float = Field(default=1200.0)
    AveOccup: float = Field(default=3.0)
    Latitude: float = Field(default=34.2)
    Longitude: float = Field(default=-118.4)

@router.post("/house-price")
def predict_house(data: HouseInput):
    global model
    
    if model is not None:
        try:
            arr = np.array([[
                data.MedInc, data.HouseAge, data.AveRooms, data.AveBedrms,
                data.Population, data.AveOccup, data.Latitude, data.Longitude
            ]])
            pred = float(model.predict(arr)[0])
            usd = max(50000.0, pred * 100000.0)
            return {
                "status": "success",
                "predicted_price_usd": round(usd, 2),
                "predicted_value_100k": round(pred, 4),
                "algorithm": "XGBoost Regressor"
            }
        except Exception:
            pass
            
    base_val = (data.MedInc * 0.45) + (data.AveRooms * 0.09) - (data.AveBedrms * 0.06) + (data.HouseAge * 0.006) + 0.35
    usd = round(max(base_val, 0.5) * 100000, 2)
    return {
        "status": "fallback",
        "predicted_price_usd": usd,
        "predicted_value_100k": round(usd / 100000, 4),
        "algorithm": "Linear Approximation"
    }
