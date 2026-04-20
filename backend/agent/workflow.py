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

from backend.agent.prompts import ANALYSIS_PROMPT_TEMPLATE, REPORT_PROMPT_TEMPLATE
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


# Guardrail: topics clearly outside the app's scope
_OFF_TOPIC_KEYWORDS = [
    "weather", "capital", "president", "recipe", "cook", "bake", "sports score",
    "movie", "music", "history", "politics", "stock", "crypto", "forex",
    "translate", "language", "poem", "story", "write me", "essay",
    "joke", "math", "equation", "solve", "code", "program", "debug",
    "python", "javascript", "sql", "hospital", "doctor", "medicine",
]

_OFF_TOPIC_REPLY = (
    "I am an AI specialized in player churn analysis and game engagement strategies "
    "for this application. I cannot answer questions outside this scope. "
    "Please ask me about this player's churn risk, engagement patterns, or retention strategies."
)


def _build_query_focused_answer(
    user_query: str | None,
    player_data: dict[str, Any],
    prediction: dict[str, Any],
) -> str:
    """Generate a dynamic, focused answer that directly addresses the user's question."""
    if not user_query or not user_query.strip():
        return ""

    q = user_query.lower()
    risk_level = prediction["risk_level"]
    prob = prediction["churn_probability"]
    lvl = player_data.get("PlayerLevel", "N/A")
    sessions = player_data.get("SessionsPerWeek", "N/A")
    duration = player_data.get("AvgSessionDurationMinutes", "N/A")
    achievements = player_data.get("AchievementsUnlocked", "N/A")
    playtime = player_data.get("PlayTimeHours", "N/A")
    genre = player_data.get("GameGenre", "N/A")
    difficulty = player_data.get("GameDifficulty", "N/A")
    purchases = player_data.get("InGamePurchases", 0)
    age = player_data.get("Age", "N/A")
    location = player_data.get("Location", "N/A")

    # --- Off-topic guardrail ---
    if any(kw in q for kw in _OFF_TOPIC_KEYWORDS):
        return _OFF_TOPIC_REPLY

    # --- "About me / Tell me about this player" ---
    if any(w in q for w in ["about me", "about myself", "who am i", "my stats", "my profile",
                             "tell me about", "summarize", "overview", "profile"]):
        snapshot = _build_feature_snapshot(player_data)
        purchase_status = "has made in-game purchases" if purchases else "has not made any purchases"
        return (
            f"Here is your gaming profile: You are a {age}-year-old {genre} player from {location}, "
            f"currently at Level {lvl} with {achievements} achievements unlocked. "
            f"You play {sessions} sessions per week, averaging {duration} minutes per session "
            f"(total {playtime} hours). You {purchase_status}. "
            f"Your engagement score is {snapshot['EngagementScore']} and progression rate is {snapshot['ProgressionRate']}. "
            f"The ML model assigns you a {prob:.1%} churn probability, placing you in the {risk_level} risk category."
        )

    # --- Churn risk / probability ---
    if any(w in q for w in ["churn", "risk", "probability", "likelihood", "chance", "will i leave",
                             "will they leave", "leaving"]):
        factors = _derive_risk_factors(player_data, prediction)
        return (
            f"This player has a {prob:.1%} churn probability and is classified as {risk_level} risk. "
            f"The key factors driving this prediction are: {' '.join(factors[:3])}"
        )

    # --- Why / reasons / causes ---
    if any(w in q for w in ["why", "reason", "cause", "what makes", "factor", "explain"]):
        factors = _derive_risk_factors(player_data, prediction)
        return (
            f"Based on the player's data, the {prob:.1%} churn probability ({risk_level} risk) is driven by: "
            + " ".join(f"({i+1}) {f}" for i, f in enumerate(factors))
        )

    # --- Retention / how to keep / strategies ---
    if any(w in q for w in ["how", "save", "retain", "keep", "prevent", "reduce", "improve",
                             "strategy", "strategies", "action", "recommend", "suggestion",
                             "what should", "what can", "what to do"]):
        strategies = _fallback_personalized_strategies(player_data, prediction, [])
        return (
            f"To retain this {risk_level.lower()}-risk player ({prob:.1%} churn probability), "
            f"here are specific recommended actions: "
            + " ".join(f"({i+1}) {s}" for i, s in enumerate(strategies))
        )

    # --- Session / engagement / activity ---
    if any(w in q for w in ["session", "engagement", "active", "inactive", "activity",
                             "behavior", "behaviour", "pattern", "play time", "playtime"]):
        snapshot = _build_feature_snapshot(player_data)
        inactive_flag = "Yes" if snapshot["IsInactive"] else "No"
        consistency = "Yes" if snapshot["SessionConsistency"] else "No"
        return (
            f"This player logs {sessions} sessions/week, averaging {duration} minutes each "
            f"(total {playtime} hours). Their Engagement Score is {snapshot['EngagementScore']}, "
            f"Progression Rate is {snapshot['ProgressionRate']}, "
            f"Session Consistency: {consistency}, Inactive Flag: {inactive_flag}. "
            f"Overall churn risk: {risk_level} ({prob:.1%})."
        )

    # --- Level / progression ---
    if any(w in q for w in ["level", "progress", "progression", "advance", "grow"]):
        snapshot = _build_feature_snapshot(player_data)
        return (
            f"This player is at Level {lvl} with {playtime} total hours played. "
            f"Their Progression Rate is {snapshot['ProgressionRate']} and they have unlocked {achievements} achievements. "
            f"{'Progression is slow relative to playtime, which can signal friction or boredom.' if snapshot['ProgressionRate'] < 1.0 else 'Progression appears healthy for their playtime.'} "
            f"Churn risk: {risk_level} ({prob:.1%})."
        )

    # --- Purchase / spending / monetization ---
    if any(w in q for w in ["purchase", "spend", "money", "monetiz", "revenue", "pay",
                             "transaction", "buy", "store"]):
        status = "has made in-game purchases" if purchases else "has NOT made any in-game purchases"
        tip = (
            "Paying players tend to show stronger retention signals."
            if purchases
            else "Non-paying players are at higher churn risk — consider a value-first starter bundle."
        )
        return (
            f"This player {status}. {tip} "
            f"Churn probability: {prob:.1%} ({risk_level} risk)."
        )

    # --- Achievements ---
    if any(w in q for w in ["achievement", "unlock", "badge", "trophy", "milestone"]):
        low = achievements <= 3
        return (
            f"This player has unlocked {achievements} achievements. "
            f"{'This is very low and suggests limited motivation or exposure to milestone systems.' if low else 'Achievement activity appears reasonable.'} "
            f"Churn risk: {risk_level} ({prob:.1%})."
        )

    # --- Genre / difficulty ---
    if any(w in q for w in ["genre", "difficulty", "game type", "mode"]):
        return (
            f"This player plays {genre} games on {difficulty} difficulty. "
            f"Their churn risk is {risk_level} ({prob:.1%}). "
            f"{'Strategy and RPG players often need clearer progression goals to stay engaged.' if genre.lower() in ('strategy', 'rpg') else 'Keeping content fresh and difficulty balanced is key to retention.'}"
        )

    # --- Generic on-topic fallback (still uses real data) ---
    factors = _derive_risk_factors(player_data, prediction)
    return (
        f"Based on your question about this player: They have a {prob:.1%} churn probability ({risk_level} risk). "
        f"They are Level {lvl}, play {sessions} sessions/week averaging {duration} min each, "
        f"and have {achievements} achievements. "
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


class ChurnAgent:
    def __init__(self, llm: Any | None = None):
        self.llm = llm
        self.app = self._compile_workflow()

    def invoke(self, state: AgentState) -> AgentState:
        initial_state: AgentState = {
            "player_data": state["player_data"],
            "user_query": state.get("user_query"),
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
            prompt_str = ANALYSIS_PROMPT_TEMPLATE.format(
                user_query=state.get("user_query") or "Analyze this player's churn risk",
                player_data=json.dumps(state["player_data"], indent=2),
                prediction=json.dumps(state["ml_prediction"], indent=2),
            )
            response = self.llm.invoke(prompt_str)
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
            # Use the ACTUAL user query — never substitute a generic fallback
            # so the LLM always answers what the user literally asked
            query_to_use = (state.get("user_query") or "").strip()
            if not query_to_use:
                query_to_use = "Provide a full churn risk analysis and retention recommendations for this player."
            analysis_dict = {
                "engagement_analysis": state["engagement_analysis"],
                "key_risk_factors": state["key_risk_factors"],
                "confidence_level": state["confidence_level"],
            }
            prompt_str = REPORT_PROMPT_TEMPLATE.format(
                user_query=query_to_use,
                player_data=json.dumps(state["player_data"], indent=2),
                prediction=json.dumps(state["ml_prediction"], indent=2),
                analysis=json.dumps(analysis_dict, indent=2),
                industry_best_practices=json.dumps(state.get("industry_best_practices", []), indent=2),
            )
            response = self.llm.invoke(prompt_str)
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
