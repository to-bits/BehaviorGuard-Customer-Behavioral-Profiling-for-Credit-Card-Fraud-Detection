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
  fraud_probability: number;
  risk_score: number;
  risk_level: RiskLevel;
  decision: "normal" | "fraud";
  model_version: string;
};

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
