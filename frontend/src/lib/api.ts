import axios from "axios";
import type {
  PlayerInput,
  PredictionResponse,
  HealthResponse,
  ModelInfoResponse,
  ModelCompareResponse,
  FeatureImportanceResponse,
  ModelWeightsResponse,
  AgentAskResponse,
} from "./types";

const API = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  headers: { "Content-Type": "application/json" },
  timeout: 30000,  // Increased timeout for LLM calls
});

/** Predict churn for a single player */
export async function predictChurn(data: PlayerInput): Promise<PredictionResponse> {
  const res = await API.post<PredictionResponse>("/predict", data);
  return res.data;
}

/** Ask the AI agent for insights - calls LLM dynamically */
export async function askAgent(
  playerData: PlayerInput,
  query: string
): Promise<AgentAskResponse> {
  const res = await API.post<AgentAskResponse>("/agent/ask", {
    player_data: playerData,
    query,
  });
  return res.data;
}

/** Check if the API is alive */
export async function checkHealth(): Promise<HealthResponse> {
  const res = await API.get<HealthResponse>("/health");
  return res.data;
}

/** Get model metadata */
export async function getModelInfo(): Promise<ModelInfoResponse> {
  const res = await API.get<ModelInfoResponse>("/model/info");
  return res.data;
}

/** Get Logistic Regression metrics */
export async function getModelComparison(): Promise<ModelCompareResponse> {
  const res = await API.get<ModelCompareResponse>("/model/compare");
  return res.data;
}

/** Get Logistic Regression feature importances */
export async function getFeatureImportance(): Promise<FeatureImportanceResponse> {
  const res = await API.get<FeatureImportanceResponse>("/model/feature-importance");
  return res.data;
}

/** Get signed Logistic Regression weights */
export async function getModelWeights(): Promise<ModelWeightsResponse> {
  const res = await API.get<ModelWeightsResponse>("/model/weights");
  return res.data;
}
