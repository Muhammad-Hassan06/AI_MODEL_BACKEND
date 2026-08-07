import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import all ML/DL project routers
from routers.fraud import router as fraud_router
from routers.house import router as house_router
from routers.airbnb import router as airbnb_router
from routers.mental_health import router as mental_health_router
from routers.spam import router as spam_router
from routers.customer import router as customer_router
from routers.movie import router as movie_router
from routers.dogcat import router as dogcat_router
from routers.mnist import router as mnist_router

# Import Gen-AI project routers
from routers.research import router as research_router
from routers.videomind import router as videomind_router

app = FastAPI(
    title="AI, Machine Learning & Generative AI Microservice Gateway API",
    description="Unified Production API hosting 11 Machine Learning, Deep Learning & Generative AI microservices for Cloud Deployment.",
    version="2.0.0"
)

# Enable CORS for all frontends (Vercel, GitHub Pages, Render, localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all ML & Deep Learning routers
app.include_router(fraud_router)
app.include_router(house_router)
app.include_router(airbnb_router)
app.include_router(mental_health_router)
app.include_router(spam_router)
app.include_router(customer_router)
app.include_router(movie_router)
app.include_router(dogcat_router)
app.include_router(mnist_router)

# Include all Generative AI routers
app.include_router(research_router)
app.include_router(videomind_router)

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Unified AI, ML & Gen-AI Portfolio Microservice Gateway",
        "active_models_and_services": 11,
        "docs_url": "/docs",
        "endpoints": [
            "POST /predict/fraud",
            "POST /predict/house-price",
            "POST /predict/airbnb",
            "POST /predict/mental-health",
            "POST /predict/spam",
            "POST /predict/customer-segmentation",
            "POST /predict/movie",
            "POST /predict/dog-cat",
            "POST /predict/mnist",
            "POST /api/research",
            "GET /api/research/stream",
            "POST /api/process-url",
            "POST /api/process-file",
            "POST /api/chat"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
