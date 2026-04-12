"""
Tests for the Milestone 2 backend agent workflow and endpoint.
"""

from backend.agent.workflow import create_agent_workflow
from backend.main import AgenticPredictionInput, load_artifacts, predict_agentic


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


def test_agent_workflow_generates_structured_report():
    agent = create_agent_workflow()

    result = agent.invoke({"player_data": VALID_PLAYER})

    assert "ml_prediction" in result
    assert "final_report" in result
    assert "engagement_analysis" in result

    report = result["final_report"]
    assert "executive_summary" in report
    assert "key_risk_factors" in report
    assert "personalized_strategies" in report
    assert "industry_best_practices" in report
    assert "sources" in report
    assert report["confidence_level"] in ("high", "medium", "low")


def test_agent_workflow_uses_fallbacks_without_keys():
    agent = create_agent_workflow()

    result = agent.invoke({"player_data": VALID_PLAYER})

    assert "warnings" in result
    assert isinstance(result["warnings"], list)
    assert len(result["warnings"]) >= 1


def test_predict_agentic_endpoint_returns_structured_response():
    load_artifacts()
    response = predict_agentic(
        AgenticPredictionInput(
            **{
                **VALID_PLAYER,
                "query": "Why is this player likely to churn and how should we retain them?",
            }
        )
    )
    payload = response.model_dump()

    assert payload["query"]
    assert "ml_prediction" in payload
    assert "report" in payload
    assert "warnings" in payload

    report = payload["report"]
    assert "executive_summary" in report
    assert isinstance(report["key_risk_factors"], list)
    assert isinstance(report["personalized_strategies"], list)
    assert isinstance(report["industry_best_practices"], list)
    assert isinstance(report["sources"], list)
