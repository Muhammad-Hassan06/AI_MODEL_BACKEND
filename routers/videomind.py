import os
import uuid
import shutil
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from genai.videomind.pipeline import run_pipeline
from genai.videomind.vector_store import ask_question

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
        "groq_key_configured": bool(groq_key)
    }

@router.post("/api/process-url")
def process_url(req: URLProcessRequest):
    """Process YouTube URL: Download, transcribe, summarize, extract tasks & index vector DB."""
    if not req.url or not req.url.strip():
        raise HTTPException(status_code=400, detail="YouTube URL is required.")
    
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
async def process_file(file: UploadFile = File(...), language: str = Form("auto")):
    """Process uploaded Audio/Video file."""
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
def chat_with_video(req: ChatRequest):
    """Interactive RAG Chatbot over the video transcript context."""
    if not req.session_id or req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Active video session not found. Please process a video first.")
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        vector_store = SESSIONS[req.session_id]["vector_store"]
        answer = ask_question(vector_store, req.question.strip())
        return {"success": True, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
