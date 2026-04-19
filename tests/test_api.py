"""
Tests for the FastAPI endpoints.
Covers /health, /model/info, /predict with valid, invalid, and edge-case inputs.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app


@pytest.fixture(scope="module")
def client():
    """Create a test client; triggers startup to load model."""
    with TestClient(app) as c:
        yield c


# ─── Sample payloads ───

VALID_PLAYER = {
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

LOW_RISK_PLAYER = {
    "Age": 22,
    "Gender": "Female",
    "Location": "Europe",
    "GameGenre": "RPG",
    "PlayTimeHours": 15.0,
    "InGamePurchases": 1,
    "GameDifficulty": "Hard",
    "SessionsPerWeek": 12,
    "AvgSessionDurationMinutes": 120,
    "PlayerLevel": 60,
    "AchievementsUnlocked": 35,
}

HIGH_RISK_PLAYER = {
    "Age": 45,
    "Gender": "Male",
    "Location": "Other",
    "GameGenre": "Strategy",
    "PlayTimeHours": 0.5,
    "InGamePurchases": 0,
    "GameDifficulty": "Easy",
    "SessionsPerWeek": 1,
    "AvgSessionDurationMinutes": 10,
    "PlayerLevel": 2,
    "AchievementsUnlocked": 0,
}


# ════════════════════════════════════════════
#  Health Endpoint
# ════════════════════════════════════════════

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        res = client.get("/health")
        assert res.status_code == 200

    def test_health_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_model_loaded(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True


# ════════════════════════════════════════════
#  Model Info Endpoint
# ════════════════════════════════════════════

class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, client):
        res = client.get("/model/info")
        assert res.status_code == 200

    def test_model_info_has_features(self, client):
        data = client.get("/model/info").json()
        assert "features" in data
        assert len(data["features"]) == 16

    def test_model_info_has_type(self, client):
        data = client.get("/model/info").json()
        assert data["model_type"] == "RandomForestClassifier"

    def test_model_info_has_categorical_mappings(self, client):
        data = client.get("/model/info").json()
        assert "Gender" in data["categorical_mappings"]
        assert "GameGenre" in data["categorical_mappings"]


# ════════════════════════════════════════════
#  Predict Endpoint — Valid Inputs
# ════════════════════════════════════════════

class TestPredictValid:
    def test_predict_returns_200(self, client):
        res = client.post("/predict", json=VALID_PLAYER)
        assert res.status_code == 200

    def test_predict_response_structure(self, client):
        data = client.post("/predict", json=VALID_PLAYER).json()
        assert "churn_probability" in data
        assert "will_churn" in data
        assert "risk_level" in data
        assert "recommendations" in data

    def test_probability_between_0_and_1(self, client):
        data = client.post("/predict", json=VALID_PLAYER).json()
        assert 0.0 <= data["churn_probability"] <= 1.0

    def test_risk_level_valid_value(self, client):
        data = client.post("/predict", json=VALID_PLAYER).json()
        assert data["risk_level"] in ("HIGH", "MEDIUM", "LOW")

    def test_will_churn_is_boolean(self, client):
        data = client.post("/predict", json=VALID_PLAYER).json()
        assert isinstance(data["will_churn"], bool)

    def test_recommendations_is_list(self, client):
        data = client.post("/predict", json=VALID_PLAYER).json()
        assert isinstance(data["recommendations"], list)
        assert len(data["recommendations"]) >= 1

    def test_low_risk_player(self, client):
        data = client.post("/predict", json=LOW_RISK_PLAYER).json()
        assert data["risk_level"] == "LOW"
        assert data["churn_probability"] < 0.4

    def test_high_risk_player(self, client):
        data = client.post("/predict", json=HIGH_RISK_PLAYER).json()
        # High-risk player should have higher probability
        assert data["churn_probability"] > 0.3

    def test_purchase_toggle_does_not_increase_risk_for_same_player(self, client):
        non_payer = {**VALID_PLAYER, "InGamePurchases": 0}
        payer = {**VALID_PLAYER, "InGamePurchases": 1}

        non_payer_prob = client.post("/predict", json=non_payer).json()["churn_probability"]
        payer_prob = client.post("/predict", json=payer).json()["churn_probability"]

        assert payer_prob <= non_payer_prob


# ════════════════════════════════════════════
#  Predict Endpoint — All Categorical Values
# ════════════════════════════════════════════

class TestPredictCategoricals:
    """Ensure every valid categorical option works without crashing."""

    @pytest.mark.parametrize("gender", ["Male", "Female"])
    def test_all_genders(self, client, gender):
        payload = {**VALID_PLAYER, "Gender": gender}
        res = client.post("/predict", json=payload)
        assert res.status_code == 200

    @pytest.mark.parametrize("location", ["USA", "Europe", "Asia", "Other"])
    def test_all_locations(self, client, location):
        payload = {**VALID_PLAYER, "Location": location}
        res = client.post("/predict", json=payload)
        assert res.status_code == 200

    @pytest.mark.parametrize("genre", ["Action", "RPG", "Strategy", "Sports", "Simulation"])
    def test_all_genres(self, client, genre):
        payload = {**VALID_PLAYER, "GameGenre": genre}
        res = client.post("/predict", json=payload)
        assert res.status_code == 200

    @pytest.mark.parametrize("diff", ["Easy", "Medium", "Hard"])
    def test_all_difficulties(self, client, diff):
        payload = {**VALID_PLAYER, "GameDifficulty": diff}
        res = client.post("/predict", json=payload)
        assert res.status_code == 200


# ════════════════════════════════════════════
#  Predict Endpoint — Invalid Inputs
# ════════════════════════════════════════════

class TestPredictInvalid:
    def test_missing_field_returns_422(self, client):
        incomplete = {k: v for k, v in VALID_PLAYER.items() if k != "Age"}
        res = client.post("/predict", json=incomplete)
        assert res.status_code == 422

    def test_empty_body_returns_422(self, client):
        res = client.post("/predict", json={})
        assert res.status_code == 422

    def test_invalid_gender_returns_422(self, client):
        payload = {**VALID_PLAYER, "Gender": "InvalidGender"}
        res = client.post("/predict", json=payload)
        assert res.status_code == 422

    def test_unsupported_gender_returns_422(self, client):
        """Gender 'Other' is not in the training data."""
        payload = {**VALID_PLAYER, "Gender": "Other"}
        res = client.post("/predict", json=payload)
        assert res.status_code == 422

    def test_age_below_min_returns_422(self, client):
        payload = {**VALID_PLAYER, "Age": 5}
        res = client.post("/predict", json=payload)
        assert res.status_code == 422

    def test_age_above_max_returns_422(self, client):
        payload = {**VALID_PLAYER, "Age": 100}
        res = client.post("/predict", json=payload)
        assert res.status_code == 422

    def test_negative_playtime_returns_422(self, client):
        payload = {**VALID_PLAYER, "PlayTimeHours": -1}
        res = client.post("/predict", json=payload)
        assert res.status_code == 422

    def test_sessions_above_max_returns_422(self, client):
        payload = {**VALID_PLAYER, "SessionsPerWeek": 50}
        res = client.post("/predict", json=payload)
        assert res.status_code == 422

    def test_string_for_numeric_field_returns_422(self, client):
        payload = {**VALID_PLAYER, "Age": "twenty"}
        res = client.post("/predict", json=payload)
        assert res.status_code == 422


# ════════════════════════════════════════════
#  Predict Endpoint — Boundary Values
# ════════════════════════════════════════════

class TestPredictBoundary:
    def test_min_age(self, client):
        payload = {**VALID_PLAYER, "Age": 15}
        assert client.post("/predict", json=payload).status_code == 200

    def test_max_age(self, client):
        payload = {**VALID_PLAYER, "Age": 65}
        assert client.post("/predict", json=payload).status_code == 200

    def test_zero_playtime(self, client):
        payload = {**VALID_PLAYER, "PlayTimeHours": 0}
        assert client.post("/predict", json=payload).status_code == 200

    def test_max_playtime(self, client):
        payload = {**VALID_PLAYER, "PlayTimeHours": 24}
        assert client.post("/predict", json=payload).status_code == 200

    def test_min_sessions(self, client):
        payload = {**VALID_PLAYER, "SessionsPerWeek": 0}
        assert client.post("/predict", json=payload).status_code == 200

    def test_max_sessions(self, client):
        payload = {**VALID_PLAYER, "SessionsPerWeek": 20}
        assert client.post("/predict", json=payload).status_code == 200

    def test_min_session_duration(self, client):
        payload = {**VALID_PLAYER, "AvgSessionDurationMinutes": 10}
        assert client.post("/predict", json=payload).status_code == 200

    def test_max_session_duration(self, client):
        payload = {**VALID_PLAYER, "AvgSessionDurationMinutes": 180}
        assert client.post("/predict", json=payload).status_code == 200

    def test_min_player_level(self, client):
        payload = {**VALID_PLAYER, "PlayerLevel": 1}
        assert client.post("/predict", json=payload).status_code == 200

    def test_max_player_level(self, client):
        payload = {**VALID_PLAYER, "PlayerLevel": 100}
        assert client.post("/predict", json=payload).status_code == 200

    def test_zero_achievements(self, client):
        payload = {**VALID_PLAYER, "AchievementsUnlocked": 0}
        assert client.post("/predict", json=payload).status_code == 200

    def test_max_achievements(self, client):
        payload = {**VALID_PLAYER, "AchievementsUnlocked": 50}
        assert client.post("/predict", json=payload).status_code == 200
