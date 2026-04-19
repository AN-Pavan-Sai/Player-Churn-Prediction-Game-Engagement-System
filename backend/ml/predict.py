"""
Prediction module for Player Churn Prediction.
Loads trained model and makes predictions on new data.
"""

import pandas as pd
import numpy as np
import joblib
import os

from backend.ml.feature_engineering import run_feature_engineering
from backend.ml.preprocess import MODELS_DIR

# Module-level cache so model artifacts are loaded once, not on every call
_cached_artifacts = None


def load_model():
    """Load the trained model and artifacts (cached after first call)."""
    global _cached_artifacts
    if _cached_artifacts is not None:
        return _cached_artifacts
    model = joblib.load(os.path.join(MODELS_DIR, "churn_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    label_encoders = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    _cached_artifacts = (model, scaler, label_encoders, feature_names)
    return _cached_artifacts


def predict_single(player_data: dict) -> dict:
    """
    Predict churn for a single player.

    Args:
        player_data: dict with keys matching dataset columns, e.g.:
            {
                "Age": 25, "Gender": "Male", "Location": "USA",
                "GameGenre": "Action", "PlayTimeHours": 10.5,
                "InGamePurchases": 1, "GameDifficulty": "Medium",
                "SessionsPerWeek": 5, "AvgSessionDurationMinutes": 90,
                "PlayerLevel": 30, "AchievementsUnlocked": 15
            }

    Returns:
        dict with prediction, probability, and risk level.
    """
    model, scaler, label_encoders, feature_names = load_model()

    df = pd.DataFrame([player_data])

    # Encode categoricals
    categorical_cols = ["Gender", "Location", "GameGenre", "GameDifficulty"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = label_encoders[col].transform(df[col])

    # Feature engineering
    df = run_feature_engineering(df)

    # Ensure correct feature order
    df = df[feature_names]

    # Scale
    df_scaled = pd.DataFrame(scaler.transform(df), columns=feature_names)

    # Predict
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    # Risk level (UPPERCASE to match workflow.py convention)
    if probability >= 0.7:
        risk_level = "HIGH"
    elif probability >= 0.4:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "churned": int(prediction),
        "churn_probability": round(float(probability), 4),
        "risk_level": risk_level,
    }


if __name__ == "__main__":
    sample = {
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
        "AchievementsUnlocked": 15,
    }
    result = predict_single(sample)
    print(f"Prediction: {'Churned' if result['churned'] else 'Active'}")
    print(f"Churn Probability: {result['churn_probability']:.2%}")
    print(f"Risk Level: {result['risk_level']}")
