export type PageKey =
  | "overview"
  | "analyzer"
  | "investigation"
  | "behavior"
  | "performance"
  | "explainable";

export type RiskLevel = "low" | "medium" | "high";

export type Factor = {
  feature: string;
  contribution: number;
  direction: "toward_fraud" | "reduces_risk";
  display_value: string;
};

export type Prediction = {
  prediction_id: string;
  timestamp: string;
  amount: number;
  fraud_probability: number;
  risk_score: number;
  risk_level: RiskLevel;
  decision: "normal" | "fraud";
  model_version: string;
  threshold: number;
  investigation_status?: InvestigationStatus;
};

export type InvestigationStatus = "new" | "reviewing" | "resolved";

export type Explanation = Prediction & {
  top_factors: Factor[];
  top_positive_contributors: Factor[];
  top_negative_contributors: Factor[];
  explanation_note: string;
};

export type ModelInfo = {
  model_name: string;
  model_version: string;
  feature_count: number;
  threshold: number;
};

export type TransactionPayload = Record<string, number | string>;

export type PredictionListResponse = { items: Prediction[] };

export type TransactionDetail = Prediction & {
  behavioral_summary: Record<string, number | string>;
};

export type ApiErrorShape = {
  message: string;
  status?: number;
  details?: unknown;
};

export type TransactionRow = {
  id: string;
  time: string;
  amount: number;
  probability: number;
  risk: RiskLevel;
  decision: "normal" | "fraud";
  channel: string;
  topSignal: string;
  reviewed: boolean;
};

export type BehaviorPoint = {
  label: string;
  baseline: number;
  observed: number;
};
