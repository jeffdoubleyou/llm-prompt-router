const API_BASE = "";

export interface ModelEntry {
  id: string;
  display_name: string;
  provider: string;
  base_url: string | null;
  capabilities: string[];
  tags: string[];
  cost_per_1k_input: number;
  cost_per_1k_output: number;
  max_tokens: number;
  context_window: number;
  rpm_limit: number;
  tpm_limit: number;
  is_active: boolean;
  priority: number;
  timeout: number | null;
  estimated_parameters_billions: number | null;
  estimated_tokens_per_second: number | null;
  max_complexity_score: number | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface RequestLogEntry {
  id: string;
  request_id: string;
  model_id: string;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  latency_ms: number;
  cost: number;
  is_error: boolean;
  error_message: string | null;
  model_used: string | null;
  created_at: string | null;
}

export interface DashboardMetrics {
  total_requests: number;
  total_cost: number;
  total_tokens: number;
  error_count: number;
  error_rate: number;
  top_models: { model_id: string; count: number }[];
  hourly: {
    timestamp: string;
    requests: number;
    avg_latency_ms: number;
    cost: number;
  }[];
}

export interface ClassifierStatus {
  model_version: string | null;
  accuracy: number | null;
  training_data_count: number;
  last_trained_at: string | null;
  is_training: boolean;
  embedding_routing_enabled?: boolean;
  embedding_model_loaded?: boolean;
  embedding_exemplar_count?: number;
  embedding_model_name?: string | null;
}

export interface ClassifierSample {
  id: string;
  prompt_text: string;
  selected_model: string;
  features: Record<string, unknown>;
  confidence: number | null;
  is_correct: boolean | null;
  created_at: string | null;
}

export interface ClassifierSamplesResponse {
  samples: ClassifierSample[];
  total: number;
  page: number;
  page_size: number;
}

export async function fetchClassifierSamples(params?: {
  page?: number;
  page_size?: number;
  model?: string;
  correct?: boolean;
  q?: string;
}): Promise<ClassifierSamplesResponse> {
  const search = new URLSearchParams();
  if (params?.page) search.set("page", String(params.page));
  if (params?.page_size) search.set("page_size", String(params.page_size));
  if (params?.model) search.set("model", params.model);
  if (params?.correct !== undefined) search.set("correct", String(params.correct));
  if (params?.q) search.set("q", params.q);
  const qs = search.toString();
  return fetcher(`/api/v1/classifier/samples${qs ? `?${qs}` : ""}`);
}

export async function updateClassifierSample(
  sampleId: string,
  isCorrect: boolean
): Promise<ClassifierSample> {
  return fetcher(`/api/v1/classifier/samples/${encodeURIComponent(sampleId)}`, {
    method: "PATCH",
    body: JSON.stringify({ is_correct: isCorrect }),
  });
}

export interface QueueStatus {
  depth: number;
  workers_active: number;
  avg_processing_time_ms: number;
  consumed_total: number;
  failed_total: number;
}

export interface PromptDebugEntry {
  request_id: string;
  model_id: string | null;
  messages: Record<string, unknown>[];
  features: Record<string, unknown>;
  created_at: string;
}

export interface PromptDebugResponse {
  prompts: PromptDebugEntry[];
  count: number;
  max_stored: number;
}

export interface MetricsSummary {
  model_id: string;
  total_requests: number;
  total_tokens: number;
  total_cost: number;
  avg_latency_ms: number;
  error_count: number;
  period_seconds: number;
}

export interface TimeSeriesPoint {
  timestamp: string;
  model_id: string;
  requests: number;
  avg_latency_ms: number;
  cost: number;
  errors: number;
}

async function fetcher<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "Unknown error");
    throw new Error(`${res.status} ${res.statusText}: ${text.slice(0, 200)}`);
  }
  return res.json();
}

export async function fetchModels(): Promise<{ models: ModelEntry[]; total: number }> {
  return fetcher("/api/v1/models");
}

export async function createModel(data: Record<string, unknown>): Promise<ModelEntry> {
  return fetcher("/api/v1/models", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function updateModel(
  id: string,
  data: Record<string, unknown>
): Promise<ModelEntry> {
  return fetcher(`/api/v1/models/${encodeURIComponent(id)}`, {
    method: "PUT",
    body: JSON.stringify(data),
  });
}

export async function deleteModel(id: string): Promise<{ deleted: string }> {
  return fetcher(`/api/v1/models/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
}

export async function exportModels(): Promise<Blob> {
  const res = await fetch(`${API_BASE}/api/v1/models/export`, {
    headers: { "Accept": "application/json" },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "Unknown error");
    throw new Error(`${res.status} ${res.statusText}: ${text.slice(0, 200)}`);
  }
  return res.blob();
}

export interface ImportResult {
  imported: number;
  skipped: number;
  errors: Array<{ model: string; error: string }>;
}

export async function importModels(file: File): Promise<ImportResult> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/api/v1/models/import`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "Unknown error");
    throw new Error(`${res.status} ${res.statusText}: ${text.slice(0, 200)}`);
  }
  return res.json();
}

export async function fetchLogs(params?: {
  skip?: number;
  limit?: number;
  model_id?: string;
  is_error?: boolean;
}): Promise<{ logs: RequestLogEntry[]; total: number }> {
  const search = new URLSearchParams();
  if (params?.skip) search.set("skip", String(params.skip));
  if (params?.limit) search.set("limit", String(params.limit));
  if (params?.model_id) search.set("model_id", params.model_id);
  if (params?.is_error !== undefined) search.set("is_error", String(params.is_error));
  const qs = search.toString();
  return fetcher(`/api/v1/logs${qs ? `?${qs}` : ""}`);
}

export async function fetchDashboardMetrics(): Promise<DashboardMetrics> {
  return fetcher("/api/v1/metrics/dashboard");
}

export async function fetchMetricsSummary(
  periodMinutes = 60
): Promise<{ metrics: MetricsSummary[] }> {
  return fetcher(`/api/v1/metrics/summary?period_minutes=${periodMinutes}`);
}

export async function fetchTimeSeries(
  periodMinutes = 60,
  granularityMinutes = 5
): Promise<{ time_series: TimeSeriesPoint[] }> {
  return fetcher(
    `/api/v1/metrics/time-series?period_minutes=${periodMinutes}&granularity_minutes=${granularityMinutes}`
  );
}

export async function fetchClassifierStatus(): Promise<ClassifierStatus> {
  return fetcher("/api/v1/classifier");
}

export async function fetchQueueStatus(): Promise<QueueStatus> {
  return fetcher("/api/v1/queue");
}

export interface UpstreamQueueEntry {
  request_id: string;
  model_id: string;
  base_url: string;
  status: string;
  position: number;
  created_at: string;
}

export interface UpstreamQueueGroup {
  base_url: string;
  base_url_key: string;
  waiting_count: number;
  processing: UpstreamQueueEntry | null;
  waiting: UpstreamQueueEntry[];
}

export interface UpstreamQueueStatus {
  enabled: boolean;
  base_urls: UpstreamQueueGroup[];
  total_waiting: number;
  total_processing: number;
}

export interface ModelRoutingEvaluation {
  model_id: string;
  eligible: boolean;
  exclusion_reason: string | null;
  max_complexity_score: number | null;
  rule_score: number | null;
  selected: boolean;
}

export interface DebugComplexityResponse {
  model_id: string | null;
  routing_method: string;
  routing_confidence: number;
  routing_difficulty: number;
  would_enqueue_classifier: boolean;
  features: Record<string, unknown>;
  complexity_explanation: Record<string, unknown>;
  model_evaluations: ModelRoutingEvaluation[];
  complexity_candidate: string | null;
  rule_candidate: string | null;
}

export async function fetchUpstreamQueueStatus(): Promise<UpstreamQueueStatus> {
  return fetcher("/api/v1/upstream-queue");
}

export async function debugComplexity(messages: Record<string, unknown>[]): Promise<DebugComplexityResponse> {
  return fetcher("/api/v1/debug/complexity", {
    method: "POST",
    body: JSON.stringify({ messages }),
  });
}

export async function fetchDebugPrompts(limit = 20): Promise<PromptDebugResponse> {
  return fetcher(`/api/v1/debug/prompts?limit=${limit}`);
}

export function connectLiveMetrics(
  onMetric: (data: Record<string, unknown>) => void,
  onError?: (err: Event) => void
): () => void {
  const eventSource = new EventSource("/api/v1/metrics/live");
  eventSource.addEventListener("metric", (event) => {
    try {
      const data = JSON.parse(event.data);
      onMetric(data);
    } catch {
      // ignore parse errors
    }
  });
  eventSource.addEventListener("error", (event) => {
    onError?.(event);
  });
  return () => eventSource.close();
}
