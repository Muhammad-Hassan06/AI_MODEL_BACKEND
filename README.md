# 🚀 AI-Backend — Unified Machine & Deep Learning Microservice API Gateway

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Render](https://img.shields.io/badge/Deploy-Render-46E3B7?style=flat-square&logo=render)](https://render.com/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

A single production-grade FastAPI microservice repository hosting **9 Machine Learning and Deep Learning models** under one roof. Designed specifically for **Render / Railway / Heroku deployment** so you only need **1 Render Service** for your entire AI portfolio!

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
                                   ▼
                      https://ai-backend.onrender.com
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       │                           │                           │
/predict/fraud              /predict/house-price         /predict/dog-cat
       │                           │                           │
       └───────────────────────────┼───────────────────────────┘
                                   │
                            FastAPI Backend
                                   │
                  Loads all 9 ML models on startup
```

---

## 📁 Repository Structure

```
AI-Backend/
├── app.py                   # Central FastAPI application & CORS configuration
├── requirements.txt         # Production Python dependencies
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
└── routers/                 # Modular API endpoints per ML domain
    ├── fraud.py             # POST /predict/fraud
    ├── house.py             # POST /predict/house-price
    ├── airbnb.py            # POST /predict/airbnb
    ├── mental_health.py     # POST /predict/mental-health
    ├── spam.py              # POST /predict/spam
    ├── customer.py          # POST /predict/customer-segmentation
    ├── movie.py             # POST /predict/movie
    ├── dogcat.py            # POST /predict/dog-cat & /predict/dog-cat-url
    └── mnist.py             # POST /predict/mnist & /predict/mnist-url
```

---

## 🔌 API Endpoints Summary

| # | Project Domain | Endpoint Path | Method | Input Format |
| :-: | :--- | :--- | :--- | :--- |
| **01** | Credit Card Fraud | `/predict/fraud` | `POST` | JSON (`{ "time": 0, "amount": 150.0, "v14": -2.3, "v17": 1.1 }`) |
| **02** | House Price | `/predict/house-price` | `POST` | JSON (`{ "MedInc": 5.2, "HouseAge": 25, "AveRooms": 5.8, ... }`) |
| **03** | Airbnb Room Type | `/predict/airbnb` | `POST` | JSON (`{ "price": 150, "minimum_nights": 2, "availability_365": 180 }`) |
| **04** | Mental Health Index | `/predict/mental-health` | `POST` | JSON (`{ "sleep_hours": 7.5, "work_hours": 8, "stress_level": 4 }`) |
| **05** | Spam Email Detector | `/predict/spam` | `POST` | JSON (`{ "text": "CONGRATULATIONS! You won $1000 cash!" }`) |
| **06** | Customer Segmentation | `/predict/customer-segmentation` | `POST` | JSON (`{ "annual_income": 75, "spending_score": 85 }`) |
| **07** | Movie Recommendation | `/predict/movie` | `POST` | JSON (`{ "movie_title": "The Dark Knight" }`) |
| **08** | Dog vs Cat Vision | `/predict/dog-cat` | `POST` | `multipart/form-data` (file) or `/predict/dog-cat-url` (JSON url) |
| **10** | MNIST Handwritten Digit | `/predict/mnist` | `POST` | `multipart/form-data` (file) or `/predict/mnist-url` (JSON url) |

---

## ☁️ Deploying to Render (Step-by-Step)

1. Push this `AI-Backend` directory to GitHub as a standalone repository (e.g., `github.com/your-username/AI-Backend`).
2. Log in to [Render Dashboard](https://dashboard.render.com/) and click **New + -> Web Service**.
3. Connect your `AI-Backend` GitHub repository.
4. Set the following build options:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn -w 2 -k uvicorn.workers.UvicornWorker app:app` (or `uvicorn app:app --host 0.0.0.0 --port $PORT`)
5. Click **Create Web Service**! Render will give you a live URL:
   `https://ai-backend.onrender.com`

---

## 💻 Local Execution

To run the API gateway locally:

```bash
cd AI-Backend
python app.py
```
- Interactive Swagger UI Docs: **[http://localhost:8000/docs](http://localhost:8000/docs)**
