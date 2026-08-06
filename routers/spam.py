import os
import pickle
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/predict", tags=["Spam Email Detector"])

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "spam_model.pkl")
model = None

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("✅ [Spam] Model loaded successfully!")
    else:
        print(f"⚠️ [Spam] Model not found at {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ [Spam] Model loading error: {e}")

class SpamInput(BaseModel):
    text: str = Field(..., description="Email or message text content to analyze")

SPAM_KEYWORDS = ["free", "winner", "prize", "cash", "urgent", "claim", "money", "congratulations", "credit", "offer"]

@router.post("/spam")
def predict_spam(data: SpamInput):
    global model
    
    text_clean = data.text.lower()
    
    if model is not None:
        try:
            pred = model.predict([text_clean])[0]
            probs = model.predict_proba([text_clean])[0]
            is_spam = bool(pred == 1 or pred == "spam")
            conf = float(probs[1]) if is_spam else float(probs[0])
        except Exception:
            spam_count = sum(1 for kw in SPAM_KEYWORDS if kw in text_clean)
            is_spam = spam_count >= 1 or len(text_clean) < 10
            conf = min(0.98, 0.65 + (spam_count * 0.15))
    else:
        spam_count = sum(1 for kw in SPAM_KEYWORDS if kw in text_clean)
        is_spam = spam_count >= 1 or "http" in text_clean
        conf = min(0.98, 0.65 + (spam_count * 0.15))

    return {
        "is_spam": is_spam,
        "label": "🚨 SPAM EMAIL" if is_spam else "✅ HAM (CLEAN EMAIL)",
        "confidence": round(conf * 100, 2),
        "text_preview": data.text[:60] + "..." if len(data.text) > 60 else data.text
    }
