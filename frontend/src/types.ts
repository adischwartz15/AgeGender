// Mirrors src/api/schemas.py. Keep in sync with the backend response shape.

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_version: string;
  checkpoint_name: string | null;
  disclaimer: string;
}

export interface ModelInfoResponse {
  model_version: string;
  checkpoint_name: string | null;
  architecture: string | null;
  gender_label_names: string[];
  gender_confidence_threshold: number;
  age_min: number;
  age_max: number;
  calibration_available: boolean;
  knn_available: boolean;
  disclaimer: string;
}

export interface QualityDiagnostics {
  width: number;
  height: number;
  brightness: number;
  contrast: number;
  blur_score: number;
  file_type: string;
  file_size_bytes: number;
  warnings: string[];
}

export interface AgeUncertaintyMetadata {
  method: string;
  calibrated: boolean;
  note: string;
}

export interface AgePrediction {
  q10: number;
  q50: number;
  q90: number;
  q10_calibrated: number | null;
  q90_calibrated: number | null;
  is_calibrated: boolean;
  uncertainty: AgeUncertaintyMetadata;
}

export interface GenderLabelPrediction {
  probabilities: Record<string, number>;
  predicted_label: string | null;
  confidence: number;
  abstained: boolean;
  display_label: string;
}

export interface GradCamResult {
  age_attention_map_base64: string | null;
  gender_attention_map_base64: string | null;
  label: string;
  caveat: string;
}

export interface KNNComparison {
  age_q10: number;
  age_q50: number;
  age_q90: number;
  gender_probabilities: Record<string, number>;
  gender_predicted_label: string | null;
  gender_display_label: string;
  gender_abstained: boolean;
  mean_neighbor_distance: number;
}

export interface PredictionResponse {
  age: AgePrediction;
  gender: GenderLabelPrediction;
  quality: QualityDiagnostics;
  gradcam: GradCamResult | null;
  knn_comparison: KNNComparison | null;
  model_version: string;
  checkpoint_name: string | null;
  warnings: string[];
  latency_ms: number;
  disclaimer: string;
}

export interface ApiError {
  detail: string;
}
