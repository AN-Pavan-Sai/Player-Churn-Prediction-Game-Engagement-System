"""
LangGraph-based agent workflow for Milestone 2.

The workflow extends the existing ML prediction flow with:
- LLM-backed explanation
- local best-practice generation
- final structured report generation

The module is intentionally resilient:
- if LangGraph is unavailable, it falls back to a sequential runner
- if GROQ is unavailable, it falls back to local heuristics
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, TypedDict

import pandas as pd

from backend.ml.feature_engineering import run_feature_engineering
from backend.ml.predict import predict_single

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency at runtime
    def load_dotenv(*args, **kwargs):
        return False


try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover - optional dependency at runtime
    END = "__END__"
    StateGraph = None


try:
    from langchain_groq import ChatGroq
except ImportError:  # pragma: no cover - optional dependency at runtime
    ChatGroq = None


load_dotenv()

logger = logging.getLogger(__name__)

# Keywords that indicate a gaming / churn / engagement related question
_RELEVANT_KEYWORDS = {
    "churn", "player", "game", "gaming", "engagement", "retention", "session",
    "play", "level", "achievement", "purchase", "risk", "score", "strategy",
    "recommend", "inactive", "active", "quit", "leave", "return", "comeback",
    "reward", "offer", "notification", "progression", "difficulty", "genre",
    "duration", "frequency", "behavior", "behaviour", "predict", "analysis",
    "why", "how", "what", "factor", "reason", "cause", "improve", "reduce",
    "increase", "decrease", "save", "lose", "losing", "engaged", "disengaged",
    "monetization", "spending", "revenue", "ltv", "lifetime", "loyalty",
}

_OFF_TOPIC_RESPONSE = (
    "I'm specialized in player churn analysis and game engagement strategies. "
    "Your question doesn't appear to be related to this player's gaming behavior. "
    "Please ask about churn risk factors, engagement patterns, or retention strategies "
    "for this player profile."
)


def _is_relevant_query(query: str | None) -> bool:
    """Return True if the query is related to gaming / churn / engagement."""
    if not query or not query.strip():
        return True  # empty query = use default dynamic query
    words = set(query.lower().split())
    return bool(words & _RELEVANT_KEYWORDS)


def get_dynamic_query(risk_level: str) -> str:
    if risk_level == "HIGH":
        return "This player is at high risk of churning. What are the critical warning signs, and what immediate, personalized actions can we take to save them?"
    elif risk_level == "MEDIUM":
        return "This player shows some signs of disengagement. Why might they be losing momentum, and how can we proactively re-engage them?"
    return "This player is currently engaged. What are their strongest retention drivers, and how can we reward their loyalty?"

class AgentState(TypedDict, total=False):
    player_data: dict[str, Any]
    user_query: str
    ml_prediction: dict[str, Any]
    engagement_analysis: str
    key_risk_factors: list[str]
    industry_best_practices: list[str]
    personalized_strategies: list[str]
    sources: list[str]
    confidence_level: str
    final_report: dict[str, Any]
    warnings: list[str]


class SequentialWorkflow:
    """Fallback workflow used when LangGraph is unavailable."""

    def __init__(self, steps: list):
        self.steps = steps

    def invoke(self, state: AgentState) -> AgentState:
        current_state = dict(state)
        for step in self.steps:
            updates = step(current_state)
            if updates:
                current_state.update(updates)
        return current_state


def _safe_json_loads(raw_text: str) -> dict[str, Any] | list[Any] | None:
    if not raw_text:
        return None

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw_text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _normalize_risk_level(risk_level: str) -> str:
    return (risk_level or "MEDIUM").upper()


def _normalize_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    churn_probability = float(prediction.get("churn_probability", 0.0))
    risk_level = _normalize_risk_level(prediction.get("risk_level", "MEDIUM"))

    return {
        "churn_probability": round(churn_probability, 4),
        "will_churn": bool(prediction.get("churned", churn_probability >= 0.5)),
        "risk_level": risk_level,
    }


def _build_feature_snapshot(player_data: dict[str, Any]) -> dict[str, float]:
    engineered = run_feature_engineering(pd.DataFrame([player_data]))
    row = engineered.iloc[0]
    return {
        "EngagementScore": round(float(row["EngagementScore"]), 2),
        "ProgressionRate": round(float(row["ProgressionRate"]), 2),
        "PurchaseFrequency": round(float(row["PurchaseFrequency"]), 2),
        "IsInactive": int(row["IsInactive"]),
        "SessionConsistency": int(row["SessionConsistency"]),
    }


def _derive_risk_factors(player_data: dict[str, Any], prediction: dict[str, Any]) -> list[str]:
    factors: list[str] = []
    engineered = _build_feature_snapshot(player_data)

    if player_data.get("SessionsPerWeek", 0) <= 2:
        factors.append("Player is highly inactive with two or fewer sessions per week.")
    elif player_data.get("SessionsPerWeek", 0) <= 4:
        factors.append("Session frequency is below the healthy engagement range.")

    if player_data.get("AvgSessionDurationMinutes", 0) < 35:
        factors.append("Average session duration is short, which suggests weak gameplay stickiness.")

    if player_data.get("PlayerLevel", 0) < 15 and player_data.get("PlayTimeHours", 0) >= 5:
        factors.append("Progression is slow relative to play time, which may indicate friction or boredom.")

    if player_data.get("AchievementsUnlocked", 0) <= 3:
        factors.append("Achievement activity is low, suggesting limited motivation or milestone completion.")

    if player_data.get("InGamePurchases", 0) == 0:
        factors.append("No purchase activity is a weak monetization and commitment signal.")

    if engineered["IsInactive"] == 1:
        factors.append("Engineered inactivity flag is triggered by the current behavior pattern.")

    if engineered["SessionConsistency"] == 0:
        factors.append("Session consistency is low, which often appears before churn.")

    if prediction["risk_level"] == "HIGH" and not factors:
        factors.append("The model predicts high churn risk based on the overall feature pattern.")
    elif not factors:
        factors.append(
            "No major churn signals are present right now, but retention still depends on maintaining session consistency and progression momentum."
        )

    return factors[:5]


def _fallback_analysis(player_data: dict[str, Any], prediction: dict[str, Any]) -> tuple[str, list[str], str]:
    factors = _derive_risk_factors(player_data, prediction)
    risk_level = prediction["risk_level"].lower()
    
    if prediction["risk_level"] in ("HIGH", "MEDIUM"):
        momentum_text = "These indicators suggest the player may be losing momentum unless the game provides a short-term reason to return."
    else:
        momentum_text = "Their overall engagement is relatively stable, though continued monitoring is recommended."

    factors_summary = " ".join(factors) if factors else "No major warning signs detected."

    analysis = (
        f"This player is currently in the {risk_level} churn-risk segment. "
        f"{factors_summary} "
        f"{momentum_text}"
    )

    confidence = "high" if prediction["risk_level"] == "HIGH" else "medium"
    return analysis, factors, confidence


def _local_best_practices(player_data: dict[str, Any], prediction: dict[str, Any]) -> list[str]:
    practices: list[str] = []
    genre = str(player_data.get("GameGenre", "game")).lower()

    if prediction["risk_level"] == "HIGH":
        practices.append(
            "High-risk players respond best to fast re-engagement loops such as comeback rewards and short-term goals."
        )
    if player_data.get("SessionsPerWeek", 0) <= 2:
        practices.append(
            "Low-frequency players are easier to recover when the next session offers immediate progress with minimal friction."
        )
    if player_data.get("AchievementsUnlocked", 0) <= 3:
        practices.append(
            "Visible milestone systems and easy wins help rebuild momentum when achievement activity is low."
        )
    if player_data.get("InGamePurchases", 0) == 0:
        practices.append(
            "Non-paying players should see value-first offers or gameplay benefits before any strong monetization push."
        )
    if genre in {"strategy", "rpg"}:
        practices.append(
            f"For {genre} players, guided progression and clearer medium-term goals usually outperform generic promotional messaging."
        )

    if not practices:
        practices.append(
            "Healthy players are usually retained through consistent progression, fresh content, and timely milestone rewards."
        )

    return practices[:5]


def _fallback_personalized_strategies(
    player_data: dict[str, Any],
    prediction: dict[str, Any],
    risk_factors: list[str],
) -> list[str]:
    strategies: list[str] = []

    if player_data.get("SessionsPerWeek", 0) <= 2:
        strategies.append("Send a comeback notification with a time-limited reward in the next 24 to 48 hours.")

    if player_data.get("PlayerLevel", 0) < 15:
        strategies.append("Create a short progression mission that helps the player reach the next meaningful level quickly.")

    if player_data.get("AchievementsUnlocked", 0) <= 3:
        strategies.append("Surface easy-to-complete achievements so the player gets a fast sense of progress.")

    if player_data.get("InGamePurchases", 0) == 0:
        strategies.append("Offer a starter bundle or beginner-friendly value pack instead of a generic store promotion.")

    if player_data.get("AvgSessionDurationMinutes", 0) < 35:
        strategies.append("Reduce early-session friction with a focused mission, bonus XP, or guided challenge.")

    if not strategies:
        strategies.append("Keep the player engaged with fresh content, milestone rewards, and social invitations.")

    if prediction["risk_level"] == "HIGH":
        strategies.append("Prioritize this player for immediate re-engagement because the model flags a high churn likelihood.")

    return strategies[:5]


def _get_disclaimers() -> list[str]:
    """Return ethical and user-experience disclaimers for the report."""
    return [
        "This analysis is generated by an AI model and should be used as a decision-support tool, not as a definitive player assessment.",
        "Player behavior predictions are based on historical patterns and may not reflect individual intent or circumstances.",
        "All engagement strategies should be reviewed by a human game designer before deployment to players.",
        "Player data should be handled in accordance with applicable privacy regulations (GDPR, CCPA).",
        "Retention interventions should respect player autonomy and avoid manipulative dark patterns.",
    ]


def _get_sources() -> list[str]:
    """Return supporting references for the report."""
    return [
        "Dataset: https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset",
        "scikit-learn Logistic Regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
        "Player Retention Best Practices — Game Developer Conference Archives",
    ]


def _build_query_focused_answer(user_query: str | None, player_data: dict[str, Any], prediction: dict[str, Any]) -> str:
    """Generate a focused answer that directly addresses the user's question using heuristics."""
    if not user_query or not user_query.strip():
        return ""

    query_lower = user_query.lower()
    risk_level = prediction["risk_level"]
    prob = prediction["churn_probability"]

    # Detect common question intents and produce tailored answers
    if any(w in query_lower for w in ["why", "reason", "cause", "factor"]):
        factors = _derive_risk_factors(player_data, prediction)
        factors_text = " ".join(factors)
        return (
            f"Based on the player profile, the model predicts a {prob:.1%} churn probability ({risk_level} risk). "
            f"The main contributing factors are: {factors_text}"
        )

    if any(w in query_lower for w in ["how", "save", "retain", "keep", "prevent", "reduce", "improve", "strategy", "action", "recommend"]):
        strategies = _fallback_personalized_strategies(player_data, prediction, [])
        strategies_text = " ".join(f"({i+1}) {s}" for i, s in enumerate(strategies))
        return (
            f"To address this player's {risk_level.lower()} churn risk ({prob:.1%} probability), "
            f"here are the recommended actions: {strategies_text}"
        )

    if any(w in query_lower for w in ["engagement", "session", "active", "inactive", "behavior", "behaviour", "pattern"]):
        snapshot = _build_feature_snapshot(player_data)
        return (
            f"This player's engagement profile shows: Engagement Score = {snapshot['EngagementScore']}, "
            f"Progression Rate = {snapshot['ProgressionRate']}, Session Consistency = {'Yes' if snapshot['SessionConsistency'] else 'No'}, "
            f"Inactive Flag = {'Yes' if snapshot['IsInactive'] else 'No'}. "
            f"Overall churn risk is {risk_level} at {prob:.1%} probability."
        )

    if any(w in query_lower for w in ["purchase", "spend", "money", "monetiz", "revenue", "pay"]):
        purchases = player_data.get("InGamePurchases", 0)
        status = "has made in-game purchases" if purchases else "has not made any in-game purchases"
        return (
            f"This player {status}. "
            f"{'Paying players typically show stronger retention signals.' if purchases else 'Non-paying players are generally at higher churn risk. Consider offering value-first promotions.'} "
            f"Current churn probability: {prob:.1%} ({risk_level} risk)."
        )

    # Generic fallback that still incorporates the query
    factors = _derive_risk_factors(player_data, prediction)
    return (
        f"Regarding your question about this player: The model assigns a {prob:.1%} churn probability ({risk_level} risk). "
        f"Key observations: {' '.join(factors[:3])}"
    )



def _fallback_report(state: AgentState) -> dict[str, Any]:
    best_practices = state.get("industry_best_practices") or _local_best_practices(
        state["player_data"], state["ml_prediction"]
    )

    risk_level = state["ml_prediction"]["risk_level"].lower()
    prob = state["ml_prediction"]["churn_probability"]

    if state["ml_prediction"]["risk_level"] == "HIGH":
        action_text = "immediate proactive retention action is highly recommended."
    elif state["ml_prediction"]["risk_level"] == "MEDIUM":
        action_text = "proactive retention action is appropriate."
    else:
        action_text = "regular engagement monitoring is advised."

    # Build a query-focused direct answer when a user question is present
    user_query = state.get("user_query")
    direct_answer = _build_query_focused_answer(
        user_query, state["player_data"], state["ml_prediction"]
    )

    return {
        "direct_answer_to_user": direct_answer,
        "executive_summary": (
            f"Player shows {risk_level} churn risk with "
            f"{prob:.1%} predicted probability. "
            f"The current engagement pattern suggests {action_text}"
        ),
        "engagement_analysis": state["engagement_analysis"],
        "key_risk_factors": state["key_risk_factors"],
        "personalized_strategies": state["personalized_strategies"],
        "industry_best_practices": best_practices,
        "sources": _get_sources(),
        "disclaimers": _get_disclaimers(),
        "confidence_level": state["confidence_level"],
    }


def _build_analysis_prompt(player_data: dict[str, Any], prediction: dict[str, Any]) -> str:
    return f"""
You are a gaming retention analyst.
Given the player data and ML prediction below, explain why this player may churn.

Player data:
{json.dumps(player_data, indent=2)}

ML prediction:
{json.dumps(prediction, indent=2)}

Return valid JSON with this exact shape:
{{
  "engagement_analysis": "2-4 sentence explanation in clear language",
  "key_risk_factors": ["factor 1", "factor 2", "factor 3"],
  "confidence_level": "high" | "medium" | "low"
}}

Rules:
- Be specific to the data.
- Do not invent features not present in the player data.
- Keep factors actionable and easy to understand.
"""


def _build_report_prompt(state: AgentState) -> str:
    return f"""
You are creating a structured churn-risk report for a gaming analytics product.

User question:
{state.get("user_query") or get_dynamic_query(state.get("ml_prediction", {}).get("risk_level", "MEDIUM"))}

Player data:
{json.dumps(state["player_data"], indent=2)}

ML prediction:
{json.dumps(state["ml_prediction"], indent=2)}

Analysis:
{json.dumps({
    "engagement_analysis": state["engagement_analysis"],
    "key_risk_factors": state["key_risk_factors"],
    "confidence_level": state["confidence_level"],
}, indent=2)}

Industry research:
{json.dumps(state.get("industry_best_practices", []), indent=2)}

Return valid JSON with this exact shape:
{{
  "direct_answer_to_user": "Clear, direct paragraph directly answering the User question",
  "executive_summary": "2-3 sentence overview",
  "engagement_analysis": "clear explanation",
  "key_risk_factors": ["factor 1", "factor 2"],
  "personalized_strategies": ["strategy 1", "strategy 2", "strategy 3"],
  "industry_best_practices": ["practice 1", "practice 2"],
  "sources": ["https://..."],
  "disclaimers": ["ethical disclaimer 1", "UX disclaimer 2"],
  "confidence_level": "high" | "medium" | "low"
}}

Rules:
- Keep recommendations grounded in the player profile.
- Do not invent URLs or citations.
- Include ethical disclaimers about AI-generated predictions and player data privacy.
- If the user's question is NOT related to gaming, player engagement, churn prediction, or retention strategies, respond with: "I'm specialized in player churn analysis and game engagement strategies. Please ask a question related to this player's gaming behavior or retention."
"""


class ChurnAgent:
    def __init__(self, llm: Any | None = None):
        self.llm = llm
        self.app = self._compile_workflow()

    def invoke(self, state: AgentState) -> AgentState:
        user_query = state.get("user_query")

        # Guard: reject off-topic questions before running the full pipeline
        if user_query and not _is_relevant_query(user_query):
            return {
                "player_data": state["player_data"],
                "user_query": user_query,
                "ml_prediction": {"churn_probability": 0.0, "will_churn": False, "risk_level": "LOW"},
                "engagement_analysis": "",
                "key_risk_factors": [],
                "personalized_strategies": [],
                "confidence_level": "low",
                "warnings": ["Query is not related to gaming or player engagement."],
                "final_report": {
                    "direct_answer_to_user": _OFF_TOPIC_RESPONSE,
                    "executive_summary": _OFF_TOPIC_RESPONSE,
                    "engagement_analysis": "",
                    "key_risk_factors": [],
                    "personalized_strategies": [],
                    "industry_best_practices": [],
                    "sources": [],
                    "disclaimers": _get_disclaimers(),
                    "confidence_level": "low",
                },
            }

        initial_state: AgentState = {
            "player_data": state["player_data"],
            "user_query": user_query,
            "warnings": list(state.get("warnings", [])),
        }
        return self.app.invoke(initial_state)

    def _compile_workflow(self):
        if StateGraph is None:
            logger.warning("LangGraph is not installed. Falling back to sequential workflow.")
            return SequentialWorkflow(
                [
                    self.predict_node,
                    self.analyze_node,
                    self.research_node,
                    self.generate_report_node,
                ]
            )

        workflow = StateGraph(AgentState)
        workflow.add_node("predict", self.predict_node)
        workflow.add_node("analyze", self.analyze_node)
        workflow.add_node("research", self.research_node)
        workflow.add_node("generate_report", self.generate_report_node)
        workflow.set_entry_point("predict")
        workflow.add_edge("predict", "analyze")
        workflow.add_edge("analyze", "research")
        workflow.add_edge("research", "generate_report")
        workflow.add_edge("generate_report", END)
        return workflow.compile()

    def predict_node(self, state: AgentState) -> AgentState:
        logger.info("Agent step: predict")
        prediction = predict_single(state["player_data"])
        normalized = _normalize_prediction(prediction)
        return {"ml_prediction": normalized}

    def analyze_node(self, state: AgentState) -> AgentState:
        logger.info("Agent step: analyze")

        if self.llm is None:
            analysis, factors, confidence = _fallback_analysis(
                state["player_data"], state["ml_prediction"]
            )
            warnings = list(state.get("warnings", []))
            warnings.append("LLM unavailable or API key missing. Used fallback explanation.")
            return {
                "engagement_analysis": analysis,
                "key_risk_factors": factors,
                "confidence_level": confidence,
                "warnings": warnings,
            }

        try:
            response = self.llm.invoke(
                _build_analysis_prompt(state["player_data"], state["ml_prediction"])
            )
            payload = _safe_json_loads(getattr(response, "content", "")) or {}
            analysis = payload.get("engagement_analysis")
            factors = payload.get("key_risk_factors")
            confidence = payload.get("confidence_level")

            if not analysis or not isinstance(factors, list):
                raise ValueError("LLM returned incomplete analysis payload")

            return {
                "engagement_analysis": analysis,
                "key_risk_factors": [str(item) for item in factors][:5],
                "confidence_level": str(confidence or "medium").lower(),
            }
        except Exception as exc:  # pragma: no cover - exercised in integration runtime
            logger.warning("LLM analysis failed: %s", exc)
            analysis, factors, confidence = _fallback_analysis(
                state["player_data"], state["ml_prediction"]
            )
            warnings = list(state.get("warnings", []))
            warnings.append("LLM analysis failed. Used fallback explanation.")
            return {
                "engagement_analysis": analysis,
                "key_risk_factors": factors,
                "confidence_level": confidence,
                "warnings": warnings,
            }

    def research_node(self, state: AgentState) -> AgentState:
        logger.info("Agent step: research")
        return {
            "industry_best_practices": _local_best_practices(
                state["player_data"], state["ml_prediction"]
            ),
            "sources": [],
        }

    def generate_report_node(self, state: AgentState) -> AgentState:
        logger.info("Agent step: generate_report")
        personalized_strategies = _fallback_personalized_strategies(
            state["player_data"],
            state["ml_prediction"],
            state.get("key_risk_factors", []),
        )

        if self.llm is None:
            return {
                "personalized_strategies": personalized_strategies,
                "final_report": _fallback_report(
                    {
                        **state,
                        "personalized_strategies": personalized_strategies,
                    }
                ),
            }

        try:
            response = self.llm.invoke(_build_report_prompt(state))
            payload = _safe_json_loads(getattr(response, "content", "")) or {}
            if not isinstance(payload, dict) or "executive_summary" not in payload:
                raise ValueError("LLM returned incomplete report payload")

            payload["direct_answer_to_user"] = str(payload.get("direct_answer_to_user", ""))
            payload["personalized_strategies"] = [
                str(item) for item in payload.get("personalized_strategies", personalized_strategies)
            ][:5]
            payload["industry_best_practices"] = [
                str(item) for item in payload.get(
                    "industry_best_practices",
                    state.get("industry_best_practices", []),
                )
            ][:5]
            payload["key_risk_factors"] = [
                str(item) for item in payload.get("key_risk_factors", state.get("key_risk_factors", []))
            ][:5]
            payload["engagement_analysis"] = str(
                payload.get("engagement_analysis", state.get("engagement_analysis", ""))
            )
            payload["confidence_level"] = str(
                payload.get("confidence_level", state.get("confidence_level", "medium"))
            ).lower()
            payload["sources"] = _get_sources()
            payload["disclaimers"] = _get_disclaimers()

            return {
                "personalized_strategies": payload["personalized_strategies"],
                "final_report": payload,
            }
        except Exception as exc:  # pragma: no cover - exercised in integration runtime
            logger.warning("Final report generation failed: %s", exc)
            warnings = list(state.get("warnings", []))
            warnings.append("Final LLM report generation failed. Used fallback report.")
            return {
                "personalized_strategies": personalized_strategies,
                "warnings": warnings,
                "final_report": _fallback_report(
                    {
                        **state,
                        "personalized_strategies": personalized_strategies,
                    }
                ),
            }


def _build_llm_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or ChatGroq is None:
        return None

    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    try:
        return ChatGroq(
            model=model_name,
            temperature=0.2,
            api_key=api_key,
        )
    except Exception as exc:  # pragma: no cover - runtime/environment dependent
        logger.warning("Unable to initialize Groq client: %s", exc)
        return None


def create_agent_workflow() -> ChurnAgent:
    """Create the Milestone 2 backend agent with safe fallbacks."""
    return ChurnAgent(llm=_build_llm_client())
