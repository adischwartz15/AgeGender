import type { HealthResponse, ModelInfoResponse, PredictionResponse, QualityDiagnostics } from "./types";

// In dev, Vite proxies /api -> http://localhost:8000 (see vite.config.ts).
// In production (Docker), set VITE_API_BASE_URL at build time.
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api";

async function parseJsonOrThrow<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = body.detail;
    } catch {
      // ignore parse failure, use default message
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/health`);
  return parseJsonOrThrow<HealthResponse>(response);
}

export async function fetchModelInfo(): Promise<ModelInfoResponse> {
  const response = await fetch(`${API_BASE}/models`);
  return parseJsonOrThrow<ModelInfoResponse>(response);
}

export async function checkImageQuality(file: File): Promise<QualityDiagnostics> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE}/quality-check`, { method: "POST", body: formData });
  return parseJsonOrThrow<QualityDiagnostics>(response);
}

export interface PredictOptions {
  includeGradcam?: boolean;
  includeKnn?: boolean;
}

export async function predict(file: File, options: PredictOptions = {}): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const params = new URLSearchParams();
  if (options.includeGradcam) params.set("include_gradcam", "true");
  if (options.includeKnn) params.set("include_knn", "true");
  const query = params.toString() ? `?${params.toString()}` : "";
  const response = await fetch(`${API_BASE}/predict${query}`, { method: "POST", body: formData });
  return parseJsonOrThrow<PredictionResponse>(response);
}
