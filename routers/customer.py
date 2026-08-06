import os
import pickle
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/predict", tags=["Customer Segmentation"])

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "segmentation_model.pkl")
model = None

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("✅ [Customer Segmentation] Model loaded successfully!")
    else:
        print(f"⚠️ [Customer Segmentation] Model not found at {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ [Customer Segmentation] Model loading error: {e}")

class CustomerInput(BaseModel):
    annual_income: float = Field(..., ge=10, le=150, description="Annual Income in $k")
    spending_score: int = Field(..., ge=1, le=100, description="Spending Score (1-100)")

PERSONAS = {
    0: {"name": "Careful Customers", "strategy": "Target with high-value quality guarantees"},
    1: {"name": "Standard Regulars", "strategy": "Engage with loyalty rewards & seasonal offers"},
    2: {"name": "Target Premium VIPs", "strategy": "VIP concierge, early access & exclusive perks"},
    3: {"name": "Careless Big Spenders", "strategy": "Promote flash sales & trendy impulse buys"},
    4: {"name": "Sensible Savers", "strategy": "Highlight discount bundles & budget deals"}
}

@router.post("/customer-segmentation")
def predict_segment(data: CustomerInput):
    global model
    
    if model is not None:
        try:
            cluster_id = int(model.predict([[data.annual_income, data.spending_score]])[0])
        except Exception:
            cluster_id = assign_heuristic_cluster(data.annual_income, data.spending_score)
    else:
        cluster_id = assign_heuristic_cluster(data.annual_income, data.spending_score)

    persona = PERSONAS.get(cluster_id, PERSONAS[1])

    return {
        "cluster_id": cluster_id,
        "persona_name": persona["name"],
        "marketing_strategy": persona["strategy"],
        "annual_income_k": data.annual_income,
        "spending_score": data.spending_score
    }

def assign_heuristic_cluster(inc, score):
    if inc > 70 and score > 70: return 2
    if inc > 70 and score <= 40: return 0
    if inc <= 40 and score > 70: return 3
    if inc <= 40 and score <= 40: return 4
    return 1
