/* ─── API Request / Response Types ─── */

export interface PlayerInput {
  Age: number;
  Gender: string;
  Location: string;
  GameGenre: string;
  PlayTimeHours: number;
  InGamePurchases: number;
  GameDifficulty: string;
  SessionsPerWeek: number;
  AvgSessionDurationMinutes: number;
  PlayerLevel: number;
  AchievementsUnlocked: number;
}

export interface PredictionResponse {
  churn_probability: number;
  will_churn: boolean;
  risk_level: "HIGH" | "MEDIUM" | "LOW";
  recommendations: string[];
  agent_query?: string | null;
  agent_answer?: string | null;
  agent_strategies?: string[];
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  message: string;
}

export interface ModelInfoResponse {
  model_type: string;
  n_features: number;
  features: string[];
  intercept: number;
  categorical_mappings: Record<string, string[]>;
}

export interface ModelCompareResponse {
  logistic_regression: Record<string, number>;
  random_forest: Record<string, number>;
}

export interface FeatureImportanceItem {
  feature: string;
  coefficient?: number;
  importance: number;
  effect?: string;
  odds_multiplier?: number;
}

export interface FeatureImportanceResponse {
  feature_importance: FeatureImportanceItem[];
}

export interface ModelWeightsResponse {
  model_type: string;
  intercept: number;
  weights: FeatureImportanceItem[];
}

/* ─── Form option helpers ─── */

export const GENDER_OPTIONS = ["Male", "Female"] as const;
export const LOCATION_OPTIONS = ["USA", "Europe", "Asia", "Other"] as const;
export const GENRE_OPTIONS = ["Action", "RPG", "Strategy", "Sports", "Simulation"] as const;
export const DIFFICULTY_OPTIONS = ["Easy", "Medium", "Hard"] as const;

export const DEFAULT_PLAYER: PlayerInput = {
  Age: 25,
  Gender: "Male",
  Location: "USA",
  GameGenre: "Action",
  PlayTimeHours: 10,
  InGamePurchases: 0,
  GameDifficulty: "Medium",
  SessionsPerWeek: 5,
  AvgSessionDurationMinutes: 60,
  PlayerLevel: 30,
  AchievementsUnlocked: 10,
};
