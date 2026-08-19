import type { BehaviorPoint, Factor } from "../types";

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

