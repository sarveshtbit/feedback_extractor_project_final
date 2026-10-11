export interface FieldExtraction {
  raw_value?: string;
  normalized_value?: string;
  source: string;
  confidence: number;
  status: string;
}

export interface VerificationDecision {
  field_name: string;
  gemini_value?: string;
  cnn_value?: string;
  final_value?: string;
  decision: string;
  confidence: number;
  reason?: string;
}

export interface PageResult {
  page_id: number;
  document_id: number;
  page_number: number;
  status: string;
  original_image_path: string;
  aligned_image_path?: string;
  fields: Record<string, FieldExtraction>;
  verification_decisions: VerificationDecision[];
}

export interface DocumentItem {
  document_id: number;
  filename: string;
  file_type: string;
  total_pages: number;
  status: string;
  created_at: string;
}

export interface ModelStatus {
  status: 'READY' | 'NOT_TRAINED' | 'TRAINING';
  weights_exist: boolean;
  architecture: string;
  notice?: string;
}

export interface SystemMetrics {
  total_documents: number;
  total_pages: number;
  completed_pages: number;
  failed_pages: number;
  total_verification_decisions: number;
  agreements: number;
  conflicts: number;
  cnn_selected: number;
  gemini_selected: number;
  cnn_unavailable: number;
  agreement_rate: number;
  conflict_rate: number;
}
