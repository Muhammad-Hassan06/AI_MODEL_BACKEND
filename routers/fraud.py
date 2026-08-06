import os
import pickle
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/predict", tags=["Fraud Detection"])

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "fraud_model.pkl")
model = None

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("✅ [Fraud] Model loaded successfully!")
    else:
        print(f"⚠️ [Fraud] Model not found at {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ [Fraud] Model loading error: {e}")

class FraudInput(BaseModel):
    time: float = 0.0
    amount: float = Field(..., ge=0.0)
    v14: float = 0.0
    v17: float = 0.0

@router.post("/fraud")
def predict_fraud(data: FraudInput):
    global model
    
    features = [[0.0] * 30]
    features[0][0] = data.time
    features[0][14] = data.v14
    features[0][17] = data.v17
    features[0][29] = data.amount
    
    if model is not None:
        try:
            pred_class = int(model.predict(features)[0])
            probs = model.predict_proba(features)[0]
            legit_prob = float(probs[0])
            fraud_prob = float(probs[1])
        except Exception:
            fraud_prob = min(0.99, max(0.01, (abs(data.v14) + abs(data.v17) + (data.amount / 1000.0)) / 10.0))
            legit_prob = 1.0 - fraud_prob
            pred_class = 1 if fraud_prob > 0.5 else 0
    else:
        fraud_prob = min(0.99, max(0.01, (abs(data.v14) + abs(data.v17) + (data.amount / 1000.0)) / 10.0))
        legit_prob = 1.0 - fraud_prob
        pred_class = 1 if fraud_prob > 0.5 else 0

    is_fraud = bool(pred_class == 1)
    label = "🚨 Fraudulent Transaction Detected" if is_fraud else "✅ Normal Legitimate Transaction"
    
    risk = "HIGH RISK 🔴" if fraud_prob > 0.7 else ("MEDIUM RISK 🟡" if fraud_prob > 0.3 else "LOW RISK 🟢")

    return {
        "is_fraud": is_fraud,
        "label": label,
        "confidence": round((fraud_prob if is_fraud else legit_prob) * 100, 2),
        "fraud_probability": round(fraud_prob, 4),
        "legit_probability": round(legit_prob, 4),
        "risk_level": risk
    }
