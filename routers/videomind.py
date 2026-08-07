import os
import uuid
import shutil
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from genai.videomind.pipeline import run_pipeline
from genai.videomind.vector_store import ask_question

limiter = Limiter(key_func=get_remote_address)
router = APIRouter(prefix="", tags=["VideoMind AI | YouTube & Audio Intelligence"])

os.makedirs("uploads", exist_ok=True)
os.makedirs("downloads", exist_ok=True)

# In-memory storage for active video intelligence sessions
SESSIONS = {}

class URLProcessRequest(BaseModel):
    url: str
    language: Optional[str] = "auto"

class ChatRequest(BaseModel):
    session_id: str
    question: str

@router.get("/api/videomind/health")
def videomind_health_check():
    groq_key = os.getenv("GROQ_API_KEY") or os.getenv("Groq_API_KEY")
    return {
        "status": "healthy",
        "service": "VideoMind AI",
        "llm_provider": "Groq Llama-3.3 70B",
        "transcriber": "Groq Whisper / Local Whisper ASR",
        "vector_db": "Chroma / Simple Vector Store",
        "rate_limits": "5 video processes & 20 chat messages per hour per IP",
        "groq_key_configured": bool(groq_key)
    }

@router.post("/api/process-url")
@limiter.limit("5/hour")
def process_url(request: Request, req: URLProcessRequest):
    """Process YouTube URL (Max 5/hour per IP, Max 10 minutes video duration)."""
    if not req.url or not req.url.strip():
        raise HTTPException(status_code=400, detail="YouTube URL is required.")
    if len(req.url.strip()) > 500:
        raise HTTPException(status_code=400, detail="URL length exceeds maximum limit.")
    
    try:
        session_id = uuid.uuid4().hex[:12]
        result = run_pipeline(req.url.strip(), language=req.language)
        
        SESSIONS[session_id] = result
        
        return {
            "success": True,
            "session_id": session_id,
            "transcript": result["transcript"],
            "summary": result["summary"],
            "actions": result["actions"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.post("/api/process-file")
@limiter.limit("5/hour")
async def process_file(request: Request, file: UploadFile = File(...), language: str = Form("auto")):
    """Process uploaded Audio/Video file (Max 5/hour per IP, Max 10 minutes audio duration)."""
    try:
        session_id = uuid.uuid4().hex[:12]
        file_ext = os.path.splitext(file.filename)[1]
        temp_path = os.path.join("uploads", f"{session_id}{file_ext}")
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        result = run_pipeline(temp_path, language=language)
        SESSIONS[session_id] = result
        
        return {
            "success": True,
            "session_id": session_id,
            "transcript": result["transcript"],
            "summary": result["summary"],
            "actions": result["actions"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

@router.post("/api/chat")
@limiter.limit("20/hour")
def chat_with_video(request: Request, req: ChatRequest):
    """Interactive RAG Chatbot over the video transcript context (Max 20/hour per IP)."""
    if not req.session_id or req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Active video session not found. Please process a video first.")
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(req.question.strip()) > 300:
        raise HTTPException(status_code=400, detail="Question length exceeds maximum limit of 300 characters.")
    
    try:
        vector_store = SESSIONS[req.session_id]["vector_store"]
        answer = ask_question(vector_store, req.question.strip())
        return {"success": True, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
