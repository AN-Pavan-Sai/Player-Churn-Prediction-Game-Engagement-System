# ChurnGuard — Player Churn Prediction & Game Engagement System

An end-to-end **Machine Learning** system that predicts player churn risk in online games and delivers personalised engagement strategies.

**[Google Colab Notebook with End-to-End Pipeline & Visuals](https://colab.research.google.com/drive/1wLqkjDsmnHh4MHkAxNykjoNR0Qv2xg6K#scrollTo=whNd12qAoKCL)**

Built with **FastAPI** (backend), **Next.js 14** (frontend), and **scikit-learn** (ML).

---

## Features

| Feature | Description |
|---------|-------------|
| **ML Prediction** | Logistic Regression model with **88.7% accuracy** and **0.93 ROC AUC** |
| **Visual Dashboard** | Interactive gauge chart, risk badges, and animated UI |
| **Smart Recommendations** | Actionable retention strategies based on risk level |
| **Real-time API** | FastAPI backend with < 50 ms prediction latency |
| **Modern UI** | Dark gaming theme with Tailwind CSS & Framer Motion |
| **Responsive** | Fully responsive across desktop, tablet, and mobile |

---

## Architecture

```
┌──────────────┐       POST /predict       ┌──────────────────┐
│              │  ───────────────────────►  │                  │
│   Next.js    │                           │   FastAPI + ML   │
│   Frontend   │  ◄───────────────────────  │   Backend        │
│   (port 3000)│    { probability, risk,   │   (port 8000)    │
│              │      recommendations }    │                  │
└──────────────┘                           └──────────────────┘
                                                    │
                                           ┌────────┴────────┐
                                            │    Logistic     │
                                            │   Regression    │
                                            │  (15 features)  │
                                            └─────────────────┘
```

---

## Project Structure

```
├── backend/
│   ├── main.py                  # FastAPI application
│   ├── requirements.txt         # Python dependencies
│   ├── agent/
│   │   └── workflow.py          # LangGraph agentic AI workflow
│   ├── ml/
│   │   ├── preprocess.py        # Data loading & encoding
│   │   ├── feature_engineering.py # 5 derived features
│   │   ├── train.py             # Training pipeline
│   │   └── predict.py           # Single-player prediction
│   └── models/                  # Saved model artifacts (.pkl)
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx       # Root layout + Navbar + Toaster
│   │   │   ├── page.tsx         # Prediction form + results
│   │   │   ├── about/page.tsx   # About, metrics, tech stack
│   │   │   └── globals.css      # Tailwind + custom styles
│   │   ├── components/
│   │   │   ├── Navbar.tsx       # Responsive navigation
│   │   │   ├── PredictionForm.tsx # 11-field input form
│   │   │   ├── ResultsDisplay.tsx # Risk badge + recommendations
│   │   │   └── GaugeChart.tsx   # SVG semicircular gauge
│   │   └── lib/
│   │       ├── api.ts           # Axios API client
│   │       └── types.ts         # TypeScript interfaces
│   ├── package.json
│   └── tailwind.config.ts
├── data/
│   └── online_gaming_behavior_dataset.csv
├── tests/                       # Unit tests
└── README.md
```

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **npm 8+**

### 1. Clone the repository

```bash
git clone https://github.com/Parth10P/Player-Churn-Prediction-Game-Engagement-System.git
cd Player-Churn-Prediction-Game-Engagement-System
```

### 2. Backend setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate          # Windows

# Install dependencies
pip install -r backend/requirements.txt

# Train the model (first time only)
python -m backend.ml.train

# Start the API
uvicorn backend.main:app --reload --port 8000
```

The API will be live at **http://localhost:8000**
Interactive docs at **http://localhost:8000/docs**

### 3. Frontend setup

```bash
cd frontend
npm install
npm run dev
```

The app will be live at **http://localhost:3000**
##### or
### Start directly
```bash
  ./setup.sh
```
---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check + model status |
| `GET` | `/model/info` | Model metadata & feature names |
| `GET` | `/model/compare` | Model evaluation metrics |
| `GET` | `/model/feature-importance` | Feature importance rankings |
| `GET` | `/model/weights` | Signed logistic regression weights |
| `POST` | `/predict` | Predict churn for a single player |
| `POST` | `/agent/ask` | Ask the AI agent for engagement insights |

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 25,
    "Gender": "Male",
    "Location": "USA",
    "GameGenre": "Action",
    "PlayTimeHours": 10.5,
    "InGamePurchases": 1,
    "GameDifficulty": "Medium",
    "SessionsPerWeek": 5,
    "AvgSessionDurationMinutes": 90,
    "PlayerLevel": 30,
    "AchievementsUnlocked": 15
  }'
```

### Example response

```json
{
  "churn_probability": 0.0813,
  "will_churn": false,
  "risk_level": "LOW",
  "recommendations": [
    "Player is healthy — maintain current experience",
    "Recognise loyalty with a milestone reward",
    "Invite to beta-test new content or features"
  ]
}
```

---

## Model Details

| Property | Value |
|----------|-------|
| Algorithm | Logistic Regression |
| Accuracy | 88.7% |
| ROC AUC | 0.93 |
| Dataset | 40,034 player records |
| Features | 15 (11 original + 4 engineered) |

### Engineered Features

| Feature | Formula |
|---------|---------|
| `EngagementScore` | SessionsPerWeek × AvgSessionDurationMinutes |
| `ProgressionRate` | PlayerLevel / (PlayTimeHours + 1) |
| `IsInactive` | 1 if SessionsPerWeek ≤ 2, else 0 |
| `SessionConsistency` | 1 if SessionsPerWeek > 3, else 0 |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML** | scikit-learn, pandas, NumPy, joblib |
| **Backend** | FastAPI, Pydantic, Uvicorn |
| **Frontend** | Next.js 14, React 18, TypeScript |
| **Styling** | Tailwind CSS, Framer Motion |
| **Charts** | Recharts, custom SVG gauge |
| **Notifications** | Sonner toast library |
| **Icons** | Lucide React |

---
