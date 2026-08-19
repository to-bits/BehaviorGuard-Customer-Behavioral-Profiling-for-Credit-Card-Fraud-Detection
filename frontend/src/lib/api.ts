import type { ApiErrorShape, Explanation, ModelInfo, Prediction, PredictionListResponse, TransactionDetail, TransactionPayload } from "../types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
const REQUEST_TIMEOUT_MS = 12_000;
let modelInfoCache: ModelInfo | null = null;

export class ApiError extends Error implements ApiErrorShape {
  status?: number;
  details?: unknown;
  constructor(message: string, status?: number, details?: unknown) { super(message); this.name = "ApiError"; this.status = status; this.details = details; }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(`${API_BASE}${path}`, { ...init, signal: controller.signal, headers: { Accept: "application/json", ...(init?.body ? { "Content-Type": "application/json" } : {}), ...init?.headers } });
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = typeof body?.detail === "string" ? body.detail : body?.detail ?? body;
      const message = response.status === 422 ? "The transaction fields are invalid." : response.status >= 500 ? "The fraud service is temporarily unavailable." : "The fraud service rejected the request.";
      throw new ApiError(message, response.status, detail);
    }
    return body as T;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error instanceof DOMException && error.name === "AbortError") throw new ApiError("The fraud service timed out. Try again.");
    throw new ApiError("Could not reach the fraud service. Check that the FastAPI server is running.");
  } finally { window.clearTimeout(timeout); }
}

export async function getModelInfo(forceRefresh = false): Promise<ModelInfo> { if (modelInfoCache && !forceRefresh) return modelInfoCache; modelInfoCache = await request<ModelInfo>("/model-info"); return modelInfoCache; }
export function predictTransaction(payload: TransactionPayload): Promise<Prediction> { return request<Prediction>("/predict", { method: "POST", body: JSON.stringify(payload) }); }
export function explainTransaction(payload: TransactionPayload, topN = 5): Promise<Explanation> { return request<Explanation>(`/explain?top_n=${topN}`, { method: "POST", body: JSON.stringify(payload) }); }
export function getRecentPredictions(limit = 20): Promise<PredictionListResponse> { return request<PredictionListResponse>(`/predictions?limit=${limit}`); }
export function getHighRiskAlerts(limit = 20): Promise<PredictionListResponse> { return request<PredictionListResponse>(`/alerts?limit=${limit}`); }
export function getTransactionDetail(predictionId: string): Promise<TransactionDetail> { return request<TransactionDetail>(`/transactions/${encodeURIComponent(predictionId)}`); }
export function updateAlertStatus(predictionId: string, status: "new" | "reviewing" | "resolved"): Promise<Prediction> { return request<Prediction>(`/alerts/${encodeURIComponent(predictionId)}`, { method: "PATCH", body: JSON.stringify({ status }) }); }
