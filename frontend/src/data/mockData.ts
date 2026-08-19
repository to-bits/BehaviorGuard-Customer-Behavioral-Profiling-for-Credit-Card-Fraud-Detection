import type { BehaviorPoint, Factor, TransactionRow } from "../types";

// Mock fallback data is isolated here and can be replaced by API adapters later.
export const mockTransactions: TransactionRow[] = [
  { id: "TX-84291", time: "Today, 09:42", amount: 1240.5, probability: 0.92, risk: "high", decision: "fraud", channel: "E-commerce", topSignal: "V14", reviewed: false },
  { id: "TX-84288", time: "Today, 09:18", amount: 86.2, probability: 0.08, risk: "low", decision: "normal", channel: "POS terminal", topSignal: "Amount_Log", reviewed: true },
  { id: "TX-84271", time: "Today, 08:51", amount: 430.0, probability: 0.67, risk: "high", decision: "fraud", channel: "Card not present", topSignal: "V10", reviewed: false },
  { id: "TX-84265", time: "Today, 08:14", amount: 64.9, probability: 0.14, risk: "low", decision: "normal", channel: "POS terminal", topSignal: "V4", reviewed: true },
  { id: "TX-84242", time: "Yesterday, 23:47", amount: 210.0, probability: 0.38, risk: "medium", decision: "normal", channel: "Mobile wallet", topSignal: "V12", reviewed: false },
  { id: "TX-84213", time: "Yesterday, 22:32", amount: 792.15, probability: 0.81, risk: "high", decision: "fraud", channel: "Card not present", topSignal: "V14", reviewed: false },
];

export const riskTrend = [
  { label: "00:00", value: 12 }, { label: "04:00", value: 18 }, { label: "08:00", value: 14 },
  { label: "12:00", value: 26 }, { label: "16:00", value: 22 }, { label: "20:00", value: 35 }, { label: "Now", value: 29 },
];

export const precisionRecall = [
  { threshold: "0.20", precision: 0.42, recall: 0.96 },
  { threshold: "0.35", precision: 0.68, recall: 0.88 },
  { threshold: "0.50", precision: 0.93, recall: 0.79 },
  { threshold: "0.54", precision: 0.93, recall: 0.79 },
  { threshold: "0.70", precision: 0.97, recall: 0.65 },
];

export const behaviorSeries: BehaviorPoint[] = [
  { label: "Amount", baseline: 42, observed: 68 },
  { label: "Velocity", baseline: 38, observed: 61 },
  { label: "Hour", baseline: 55, observed: 49 },
  { label: "Category", baseline: 44, observed: 72 },
  { label: "V-features", baseline: 48, observed: 81 },
];

export const globalFactors: Factor[] = [
  { feature: "V14", contribution: 0.894, direction: "toward_fraud", display_value: "model importance 0.894" },
  { feature: "V10", contribution: 0.821, direction: "toward_fraud", display_value: "model importance 0.821" },
  { feature: "V12", contribution: 0.819, direction: "toward_fraud", display_value: "model importance 0.819" },
  { feature: "V4", contribution: 0.812, direction: "toward_fraud", display_value: "model importance 0.812" },
  { feature: "V8", contribution: 0.784, direction: "toward_fraud", display_value: "model importance 0.784" },
  { feature: "Transactions_Last_1H", contribution: 0.154, direction: "toward_fraud", display_value: "model importance 0.154" },
];

export const mockExplanation = {
  fraud_probability: 0.92,
  risk_score: 92,
  risk_level: "high" as const,
  decision: "fraud" as const,
  model_version: "xgboost-final-v1",
  top_factors: [
    { feature: "V14", contribution: 2.31, direction: "toward_fraud" as const, display_value: "-4.218" },
    { feature: "Transactions_Last_1H", contribution: 0.84, direction: "toward_fraud" as const, display_value: "18 transactions" },
    { feature: "Amount_Log", contribution: 0.48, direction: "toward_fraud" as const, display_value: "7.12" },
  ],
  top_positive_contributors: [],
  top_negative_contributors: [
    { feature: "V4", contribution: -0.34, direction: "reduces_risk" as const, display_value: "0.82" },
  ],
  explanation_note: "Contributions describe how the model influenced this prediction; they do not prove causality.",
};
