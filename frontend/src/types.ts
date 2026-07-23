export type Sex = "M" | "F";

export interface BoneClassification {
  stage: string;
  probabilities: Record<string, number>;
  source: string;
}

export interface BoneDetection {
  box: [number, number, number, number];
  conf: number;
  source: string;
}

export interface GaussStageData {
  stage: string;
  mean: number;
  sd: number;
  probability: number;
}

export interface GaussBoneData {
  stages: GaussStageData[];
  detected_stage: string;
  chrono_age_months: number | null;
  label: string;
}

export interface AnalysisResult {
  bone_age_months: number;
  bone_age_years: number;
  confidence_interval: [number, number];
  rus_score: number;
  carpal_score: number;
  rus_age_months: number;
  carpal_age_months: number;
  sex: Sex;
  mm_per_px: number;
  stages: Record<string, string>;
  bone_scores: Record<string, number>;
  classifications: Record<string, BoneClassification>;
  detections: Record<string, BoneDetection>;
  annotated_image_b64: string;
  gaussian_data: Record<string, GaussBoneData>;
}
