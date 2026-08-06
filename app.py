import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import all project routers
from routers.fraud import router as fraud_router
from routers.house import router as house_router
from routers.airbnb import router as airbnb_router
from routers.mental_health import router as mental_health_router
from routers.spam import router as spam_router
from routers.customer import router as customer_router
from routers.movie import router as movie_router
from routers.dogcat import router as dogcat_router
from routers.mnist import router as mnist_router

app = FastAPI(
    title="AI & Machine Learning Microservice Gateway API",
    description="Unified Production API hosting 9 Machine & Deep Learning models for Render Cloud Deployment.",
    version="1.0.0"
)

# Enable CORS for all frontends (Vercel, GitHub Pages, Render, localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all endpoint routers
app.include_router(fraud_router)
app.include_router(house_router)
app.include_router(airbnb_router)
app.include_router(mental_health_router)
app.include_router(spam_router)
app.include_router(customer_router)
app.include_router(movie_router)
app.include_router(dogcat_router)
app.include_router(mnist_router)

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Unified AI/ML Portfolio Microservice Gateway",
        "active_models": 9,
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
            "POST /predict/mnist"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
