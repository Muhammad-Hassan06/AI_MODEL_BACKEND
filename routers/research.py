import os
import asyncio
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from genai.research.pipeline import run_research_pipeline, run_research_pipeline_stream

limiter = Limiter(key_func=get_remote_address)
router = APIRouter(prefix="", tags=["Autonomous AI Research Agent"])

class ResearchRequest(BaseModel):
    topic: str

@router.get("/api/research/health")
def research_health_check():
    groq_key = os.getenv("GROQ_API_KEY") or os.getenv("Groq_API_KEY") or os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    return {
        "status": "healthy",
        "service": "Autonomous AI Research Agent",
        "llm_provider": "Groq (llama-3.3-70b-versatile)",
        "rate_limit": "3 requests per hour per IP",
        "groq_key_configured": bool(groq_key),
        "tavily_key_configured": bool(tavily_key)
    }

@router.post("/api/research")
@limiter.limit("3/hour")
def research_endpoint(request: Request, req: ResearchRequest):
    """Run multi-agent research pipeline synchronously (Max 3/hour per IP address)."""
    if not req.topic or not req.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")
    if len(req.topic.strip()) > 200:
        raise HTTPException(status_code=400, detail="Topic length exceeds maximum limit of 200 characters.")
        
    try:
        results = run_research_pipeline(req.topic.strip())
        return {"success": True, "topic": req.topic, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/research/stream")
@limiter.limit("3/hour")
async def research_stream_endpoint(request: Request, topic: str):
    """Stream multi-agent pipeline progress step-by-step using SSE (Max 3/hour per IP address)."""
    if not topic or not topic.strip():
        raise HTTPException(status_code=400, detail="Topic parameter is required.")
    if len(topic.strip()) > 200:
        raise HTTPException(status_code=400, detail="Topic length exceeds maximum limit of 200 characters.")
    
    async def event_generator():
        try:
            loop = asyncio.get_event_loop()
            gen = run_research_pipeline_stream(topic.strip())
            
            while True:
                item = await loop.run_in_executor(None, next, gen, None)
                if item is None:
                    break
                yield f"data: {item}\n\n"
                await asyncio.sleep(0.1)
        except Exception as e:
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
