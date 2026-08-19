import type { Explanation, ModelInfo, Prediction } from "../types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export async function getModelInfo(): Promise<ModelInfo> {
  const response = await fetch(`${API_BASE}/model-info`);
  if (!response.ok) throw new Error("Model metadata is unavailable");
  return response.json() as Promise<ModelInfo>;
}

export async function predictTransaction(payload: Record<string, unknown>): Promise<Prediction> {
  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error("Prediction request failed");
  return response.json() as Promise<Prediction>;
}

export async function explainTransaction(payload: Record<string, unknown>): Promise<Explanation> {
  const response = await fetch(`${API_BASE}/explain?top_n=5`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error("Explanation request failed");
  return response.json() as Promise<Explanation>;
}
