"""
FastAPI Backend for Player Churn Prediction System.
Provides REST endpoints for the Next.js frontend.
"""

import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Path setup — so we can import the existing ML modules
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from backend.agent.workflow import create_agent_workflow
from backend.ml.feature_engineering import run_feature_engineering
from backend.ml.preprocess import MODELS_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Player Churn Prediction API",
    description="ML-powered API to predict player churn risk in online games",
    version="1.0.0",
)

# CORS — allow Next.js dev server & production origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load model artifacts once at startup
# ---------------------------------------------------------------------------
model = None
scaler = None
label_encoders = None
feature_names = None


def _ensure_model_loaded():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — run training first")


def _build_weight_rows(names, coefficients):
    rows = []
    for name, coef in zip(names, coefficients):
        rows.append(
            {
                "feature": name,
                "coefficient": round(float(coef), 4),
                "importance": round(float(abs(coef)), 4),
                "effect": "increases churn risk" if coef > 0 else "reduces churn risk",
                "odds_multiplier": round(float(np.exp(coef)), 4),
            }
        )
    return sorted(rows, key=lambda item: item["importance"], reverse=True)


def apply_purchase_calibration(probability: float, data: dict) -> float:
    """
    Apply a small business-rule calibration on top of the model output.

    Paying users usually show slightly stronger retention than otherwise
    identical non-paying users, so we nudge the score downward for them.
    """
    adjusted = float(probability)
    if data.get("InGamePurchases", 0) == 1:
        adjusted = max(0.0, adjusted - 0.05)
    return min(1.0, adjusted)


@app.on_event("startup")
def load_artifacts():
    """Load ML artifacts into memory when the server starts."""
    global model, scaler, label_encoders, feature_names, agent
    try:
        model = joblib.load(os.path.join(MODELS_DIR, "churn_model.pkl"))
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        label_encoders = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))
        feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
        print(f"✅ Model loaded — {len(feature_names)} features")
    except Exception as e:
        print(f"⚠️  Could not load model artifacts: {e}")

    try:
        agent = create_agent_workflow()
        logger.info("✅ Agent workflow initialized")
    except Exception as e:
        agent = None
        logger.warning("⚠️ Could not initialize agent workflow: %s", e)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class PlayerInput(BaseModel):
    """Raw player data sent from the frontend (6 user-facing fields)."""
    Age: int = Field(..., ge=15, le=65, description="Player age")
    Gender: str = Field(..., description="Male / Female")
    Location: str = Field(..., description="USA / Europe / Asia / Other")
    GameGenre: str = Field(..., description="Action / RPG / Strategy / Sports / Simulation")
    PlayTimeHours: float = Field(..., ge=0, le=24, description="Daily play time in hours")
    InGamePurchases: int = Field(..., ge=0, le=1, description="0 = No, 1 = Yes")
    GameDifficulty: str = Field(..., description="Easy / Medium / Hard")
    SessionsPerWeek: int = Field(..., ge=0, le=20, description="Sessions per week")
    AvgSessionDurationMinutes: int = Field(..., ge=10, le=180, description="Avg session duration (min)")
    PlayerLevel: int = Field(..., ge=1, le=100, description="Current player level")
    AchievementsUnlocked: int = Field(..., ge=0, le=50, description="Achievements unlocked")

    model_config = {"json_schema_extra": {"examples": [{
        "Age": 25, "Gender": "Male", "Location": "USA",
        "GameGenre": "Action", "PlayTimeHours": 10.5,
        "InGamePurchases": 1, "GameDifficulty": "Medium",
        "SessionsPerWeek": 5, "AvgSessionDurationMinutes": 90,
        "PlayerLevel": 30, "AchievementsUnlocked": 15,
    }]}}


class PredictionResponse(BaseModel):
    churn_probability: float
    will_churn: bool
    risk_level: str
    recommendations: list[str]
    agent_query: str | None = None
    agent_answer: str | None = None
    agent_strategies: list[str] = []


class PredictInput(PlayerInput):
    query: str | None = Field(
        default=None,
        description="Optional question for the AI agent using the same /predict endpoint",
    )


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------
def get_recommendations(risk_level: str, data: dict) -> list[str]:
    """Return actionable recommendations based on risk level & player data."""
    recs: list[str] = []

    if risk_level == "HIGH":
        recs.append("Send a personalised retention offer immediately")
        recs.append("Offer exclusive in-game rewards or limited-time items")
        recs.append("Assign to priority support for proactive outreach")
        if data.get("SessionsPerWeek", 0) <= 2:
            recs.append("Send re-engagement push notifications")
        if data.get("InGamePurchases", 0) == 0:
            recs.append("Offer a first-purchase discount or starter pack")

    elif risk_level == "MEDIUM":
        recs.append("Monitor engagement trends over the next 7 days")
        recs.append("Introduce achievement-based challenges to boost activity")
        recs.append("Suggest social features like guilds or team events")
        if data.get("PlayerLevel", 0) < 20:
            recs.append("Provide a guided progression quest to level 20")

    else:  # LOW
        recs.append("Player is healthy — maintain current experience")
        recs.append("Recognise loyalty with a milestone reward")
        recs.append("Invite to beta-test new content or features")

    return recs


def get_enhanced_recommendations(risk_level: str, data: dict) -> list[str]:
    """
    Keep the /predict response shape stable while upgrading its recommendation
    quality with the internal agent workflow when available.
    """
    if agent is None:
        return get_recommendations(risk_level, data)

    try:
        result = agent.invoke({"player_data": data, "user_query": None})
        report = result.get("final_report", {})
        strategies = report.get("personalized_strategies", [])
        if isinstance(strategies, list) and strategies:
            return [str(item) for item in strategies][:5]
    except Exception as exc:
        logger.warning("Enhanced recommendation generation failed: %s", exc)

    return get_recommendations(risk_level, data)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "message": "Player Churn Prediction API is running",
    }


@app.get("/model/info")
def model_info():
    """Return model metadata."""
    _ensure_model_loaded()

    # Get intercept if available (LogisticRegression has it, RandomForest does not)
    intercept = None
    if hasattr(model, "intercept_") and len(model.intercept_) > 0:
        intercept = round(float(model.intercept_[0]), 4)

    return {
        "model_type": type(model).__name__,
        "n_features": len(feature_names),
        "features": feature_names,
        "intercept": intercept,
        "categorical_mappings": {
            col: list(le.classes_) for col, le in label_encoders.items()
        },
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(player: PredictInput):
    """Predict churn risk for a single player."""
    _ensure_model_loaded()

    try:
        payload = player.model_dump()
        user_query = payload.pop("query", None)
        data = payload
        df = pd.DataFrame([data])

        # Encode categorical columns
        categorical_cols = ["Gender", "Location", "GameGenre", "GameDifficulty"]
        for col in categorical_cols:
            df[col] = label_encoders[col].transform(df[col])

        # Apply feature engineering (adds EngagementScore, ProgressionRate, etc.)
        df = run_feature_engineering(df)

        # Select features in training order & scale
        df = df[feature_names]
        df_scaled = pd.DataFrame(scaler.transform(df), columns=feature_names)

        # Predict
        prediction = int(model.predict(df_scaled)[0])
        probability = float(model.predict_proba(df_scaled)[0][1])
        probability = apply_purchase_calibration(probability, data)

        # Risk level
        if probability >= 0.7:
            risk_level = "HIGH"
        elif probability >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        recommendations = get_enhanced_recommendations(risk_level, data)

        agent_answer = None
        agent_strategies: list[str] = []
        if user_query and agent is not None:
            try:
                result = agent.invoke({"player_data": data, "user_query": user_query})
                report = result.get("final_report", {})
                answer_parts = [
                    report.get("executive_summary", ""),
                    report.get("engagement_analysis", ""),
                ]
                combined = " ".join(part.strip() for part in answer_parts if part)
                agent_answer = combined or None
                strategies = report.get("personalized_strategies", [])
                if isinstance(strategies, list):
                    agent_strategies = [str(item) for item in strategies][:5]
            except Exception as exc:
                logger.warning("Agent follow-up generation failed: %s", exc)

        return PredictionResponse(
            churn_probability=round(probability, 4),
            will_churn=bool(prediction),
            risk_level=risk_level,
            recommendations=recommendations,
            agent_query=user_query,
            agent_answer=agent_answer,
            agent_strategies=agent_strategies,
        )

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))



# ---------------------------------------------------------------------------
# Agent / LLM Endpoint - Called separately when user clicks "Ask Agent"
# ---------------------------------------------------------------------------
class AgentQueryInput(BaseModel):
    player_data: dict
    query: Optional[str] = Field(default=None)


class AgentQueryResponse(BaseModel):
    agent_answer: str
    agent_strategies: list[str]
    confidence_level: str


@app.post("/agent/ask", response_model=AgentQueryResponse)
def ask_agent(query_input: AgentQueryInput):
    """
    Dedicated endpoint for LLM agent queries.
    This is called when user clicks 'Ask Agent' to ensure LLM is invoked.
    """
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="Agent workflow not initialized. Check GROQ_API_KEY and dependencies."
        )

    try:
        result = agent.invoke({
            "player_data": query_input.player_data,
            "user_query": query_input.query
        })

        report = result.get("final_report", {})

        # Build answer from report sections
        direct_answer = report.get("direct_answer_to_user", "")
        if direct_answer:
            combined = direct_answer
        else:
            answer_parts = [
                report.get("executive_summary", ""),
                report.get("engagement_analysis", ""),
            ]
            combined = " ".join(part.strip() for part in answer_parts if part)

        strategies = report.get("personalized_strategies", [])
        if not isinstance(strategies, list):
            strategies = []

        return AgentQueryResponse(
            agent_answer=combined or "No analysis available.",
            agent_strategies=[str(item) for item in strategies][:5],
            confidence_level=report.get("confidence_level", "medium"),
        )

    except Exception as exc:
        logger.error("Agent query failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Agent query failed: {exc}")


# ---------------------------------------------------------------------------
# Model comparison & feature importance helpers
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(BASE_DIR, "models")


def _parse_results_file(path: str) -> dict:
    """Parse a results.txt file into a {metric: value} dict."""
    metrics = {}
    if not os.path.exists(path):
        return metrics
    with open(path) as f:
        for line in f:
            if ":" in line and any(
                k in line for k in ("Accuracy", "Precision", "Recall", "F1", "ROC")
            ):
                key, val = line.strip().split(":")
                metrics[key.strip()] = round(float(val.strip()), 4)
    return metrics


@app.get("/model/compare")
def model_compare():
    """Return Logistic Regression metrics."""
    logistic_path = os.path.join(RESULTS_DIR, "logistic_results.txt")
    logistic_metrics = _parse_results_file(logistic_path)

    if not logistic_metrics:
        raise HTTPException(status_code=404, detail="No results files found")

    return {
        "logistic_regression": logistic_metrics,
    }


@app.get("/model/feature-importance")
def feature_importance():
    """Return feature importances sorted by importance."""
    _ensure_model_loaded()

    # RandomForest uses feature_importances_, LogisticRegression uses coef_
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = model.coef_[0]
    else:
        importances = np.zeros(len(feature_names))

    result = _build_weight_rows(feature_names, importances)
    return {"feature_importance": result}


@app.get("/model/weights")
def model_weights():
    """Return model weights/feature importances."""
    _ensure_model_loaded()

    # Get intercept if available (LogisticRegression has it, RandomForest does not)
    intercept = None
    if hasattr(model, "intercept_") and len(model.intercept_) > 0:
        intercept = round(float(model.intercept_[0]), 4)

    # RandomForest uses feature_importances_, LogisticRegression uses coef_
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = model.coef_[0]
    else:
        importances = np.zeros(len(feature_names))

    return {
        "model_type": type(model).__name__,
        "intercept": intercept,
        "weights": _build_weight_rows(feature_names, importances),
    }


# ---------------------------------------------------------------------------
# Run with: uvicorn backend.main:app --reload --port 8000
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
