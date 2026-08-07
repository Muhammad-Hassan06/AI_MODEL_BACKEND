# 🚀 AI-Backend — Unified Machine, Deep Learning & Generative AI Microservice API Gateway

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Render](https://img.shields.io/badge/Deploy-Render-46E3B7?style=flat-square&logo=render)](https://render.com/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/Orchestration-LangChain-1C3C3C?style=flat-square&logo=chainlink)](https://python.langchain.com/)
[![Groq](https://img.shields.io/badge/Inference-Groq_Llama_3.3_70B-orange?style=flat-square)](https://groq.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

A single production-grade FastAPI microservice repository hosting **11 Machine Learning, Deep Learning, and Generative AI applications** under one roof. Designed specifically for cloud deployment (Render / Railway / Heroku) so you only need **1 central backend service** for your entire AI portfolio!

---

## 🏗️ Architecture Overview

```
                      React App 1 (Fraud Detection)
                                   │
                      React App 2 (House Price)
                                   │
                      React App 3 (Airbnb Room Type)
                                   │
                      React App 4 (Mental Health)
                                   │
                      React App 5 (Spam Detector)
                                   │
                      React App 6 (Customer Segmentation)
                                   │
                      React App 7 (Movie Recommendation)
                                   │
                      React App 8 (Dog vs Cat Vision)
                                   │
                      React App 10 (MNIST Digit AI)
                                   │
                      Gen-AI Project 1 (Autonomous AI Research Agent)
                                   │
                      Gen-AI Project 2 (VideoMind AI Platform)
                                   │
                                   ▼
                      https://ai-backend.onrender.com
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       │                           │                           │
/predict/fraud              /api/research               /api/process-url
       │                           │                           │
       └───────────────────────────┼───────────────────────────┘
                                   │
                            FastAPI Backend
                                   │
                  Loads 9 ML models & 2 Gen-AI Pipelines
```

---

## 📁 Repository Structure

```
AI-Backend/
├── app.py                   # Central FastAPI application & CORS configuration
├── requirements.txt         # Production Python dependencies (ML + Deep Learning + Gen-AI)
├── setup_models.py          # Script to collect & organize model binary pickles
├── models/                  # Serialized ML/DL models
│   ├── fraud_model.pkl
│   ├── house_model.pkl
│   ├── airbnb_model.pkl
│   ├── mental_health_model.pkl
│   ├── spam_model.pkl
│   ├── segmentation_model.pkl
│   ├── movie_df.pkl
│   ├── dogcat_model.pkl
│   └── mnist_model.pkl
├── genai/                   # Modular Generative AI core engines & multi-agent swarms
│   ├── research/            # Autonomous AI Research Agent (LangGraph + Tavily + Groq)
│   │   ├── agents.py
│   │   ├── pipeline.py
│   │   └── tools.py
│   └── videomind/           # VideoMind AI (YouTube / Audio Intelligence, Whisper, RAG Chatbot)
│       ├── audio_processor.py
│       ├── extractor.py
│       ├── pipeline.py
│       ├── summarize.py
│       ├── transcriber.py
│       └── vector_store.py
└── routers/                 # Modular API routers per project domain
    ├── fraud.py             # POST /predict/fraud
    ├── house.py             # POST /predict/house-price
    ├── airbnb.py            # POST /predict/airbnb
    ├── mental_health.py     # POST /predict/mental-health
    ├── spam.py              # POST /predict/spam
    ├── customer.py          # POST /predict/customer-segmentation
    ├── movie.py             # POST /predict/movie
    ├── dogcat.py            # POST /predict/dog-cat & /predict/dog-cat-url
    ├── mnist.py             # POST /predict/mnist & /predict/mnist-url
    ├── research.py          # POST /api/research & GET /api/research/stream (SSE)
    └── videomind.py         # POST /api/process-url, /api/process-file & /api/chat
```

---

## 🔌 API Endpoints Summary

| # | Project Domain | Endpoint Path | Method | Description / Input |
| :-: | :--- | :--- | :--- | :--- |
| **01** | Credit Card Fraud | `/predict/fraud` | `POST` | JSON (`{ "time": 0, "amount": 150.0, "v14": -2.3, ... }`) |
| **02** | House Price | `/predict/house-price` | `POST` | JSON (`{ "MedInc": 5.2, "HouseAge": 25, ... }`) |
| **03** | Airbnb Room Type | `/predict/airbnb` | `POST` | JSON (`{ "price": 150, "minimum_nights": 2, ... }`) |
| **04** | Mental Health Index | `/predict/mental-health` | `POST` | JSON (`{ "sleep_hours": 7.5, "work_hours": 8, ... }`) |
| **05** | Spam Email Detector | `/predict/spam` | `POST` | JSON (`{ "text": "CONGRATULATIONS! You won cash!" }`) |
| **06** | Customer Segmentation | `/predict/customer-segmentation` | `POST` | JSON (`{ "annual_income": 75, "spending_score": 85 }`) |
| **07** | Movie Recommendation | `/predict/movie` | `POST` | JSON (`{ "movie_title": "The Dark Knight" }`) |
| **08** | Dog vs Cat Vision | `/predict/dog-cat` | `POST` | `multipart/form-data` (file) or `/predict/dog-cat-url` |
| **10** | MNIST Handwritten Digit | `/predict/mnist` | `POST` | `multipart/form-data` (file) or `/predict/mnist-url` |
| **Gen-AI 1** | Autonomous AI Research Agent | `/api/research` | `POST` | Multi-Agent Swarm report generation |
| **Gen-AI 1** | AI Research Agent Stream | `/api/research/stream` | `GET` | Real-time SSE step-by-step stream |
| **Gen-AI 2** | VideoMind AI Process URL | `/api/process-url` | `POST` | YouTube video transcription, summary & action items |
| **Gen-AI 2** | VideoMind AI Process File | `/api/process-file` | `POST` | Audio/Video file upload processing |
| **Gen-AI 2** | VideoMind AI Chatbot | `/api/chat` | `POST` | Interactive RAG Chatbot over transcript context |

---

## 🔑 Environment Variables

To activate all Gen-AI capabilities (Research Agent & VideoMind AI), set the following environment variables:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

---

## ☁️ Deploying to Render

1. Commit and push this `AI-Backend` directory to your GitHub account as `AI-Backend`.
2. Go to [Render Dashboard](https://dashboard.render.com/) -> **New + -> Web Service**.
3. Connect your `AI-Backend` repository.
4. Set build & start settings:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn -w 2 -k uvicorn.workers.UvicornWorker app:app` (or `uvicorn app:app --host 0.0.0.0 --port $PORT`)
5. Add `GROQ_API_KEY` & `TAVILY_API_KEY` under Environment Variables.
6. Click **Create Web Service**!

---

## 💻 Local Execution

To run locally:

```bash
python app.py
```
- Interactive Swagger UI Documentation: **[http://localhost:8000/docs](http://localhost:8000/docs)**
