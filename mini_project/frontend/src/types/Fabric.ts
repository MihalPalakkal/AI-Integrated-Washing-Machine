// API-1 returns dynamic material names, so no fixed union needed
export type FiberCategory = 'Natural' | 'Synthetic' | 'Semi-synthetic';

export interface FabricDetection {
  name: string;            // material_type from API-1 (e.g. "Cotton Twill")
  confidence: number;      // 0–1 confidence_score
  fiberCategory: string;   // Natural / Synthetic / Semi-synthetic
  dirtLevel: number;       // 1–5 soil level
  description: string;     // 3-line property summary
}

export interface FabricAnalysisResult {
  fabrics: FabricDetection[];
  recommendedCycle: string;
}

// Raw API-1 POST /predict response shape
export interface API1PredictResponse {
  total_images: number;
  results: API1Result[];
  hint: string;
}

export interface API1Result {
  filename: string;
  material_type: string | null;
  fiber_category: string | null;
  description: string | null;
  dirt_level: number | null;
  confidence_score: number | null;
  is_retry: boolean;
  previous_wrong_prediction: string | null;
  error: string | null;
}
