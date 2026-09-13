export type Broker = {
  connected: boolean;
  company: string | null;
  server: string | null;
  read_only: boolean;
};
export type SystemStatus = {
  backend_online: boolean;
  source: string;
  broker: Broker;
  research_mode: string;
  strategy_count: number;
  dataset_count: number;
  partition_count: number;
  verified_test_count: number;
  integrity_status: string;
};
export type StrategySummary = {
  strategy_id: string;
  name: string;
  version: string;
  description: string;
  required_timeframes: string[];
  capabilities: string[];
  implementation_status: string;
  research_status: string;
};
export type Parameter = {
  name: string;
  type: string;
  material: boolean;
  required: boolean;
  default: unknown;
};
export type StrategyDetail = StrategySummary & {
  parameters: Parameter[];
  code_fingerprint: string;
  parameter_fingerprint: string;
};
export type ResearchStatus = {
  program_count: number;
  hypothesis_count: number;
  experiment_count: number;
  candidate_count: number;
  partition_count: number;
  discovery_status: string;
  validation_status: string;
  final_test_status: string;
};
export type Dataset = {
  dataset_id: string;
  broker: string;
  server: string;
  symbol: string;
  timeframe: string;
  start: string;
  end: string;
  row_count: number;
  checksum: string;
  schema_version: string;
};
export type ImportedDataset = {
  dataset_id: string;
  provider: string;
  symbol: string;
  data_type: string;
  price_type: string;
  date_range: string;
  row_count: number;
  certification: string;
  purpose: string;
  fingerprint: string;
};
export type DataStatus = {
  broker: Broker;
  datasets: Dataset[];
  imported_datasets: ImportedDataset[];
  dataset_count: number;
  external_import_gateway_available: boolean;
  external_production_dataset_count: number;
  research_partition_count: number;
};
export type ImportMetadata = {
  provider: string;
  symbol: string;
  data_type: string;
  price_type: string;
  source_timezone: string;
  purpose: string;
  timestamp_format: string;
  interpretation_reason?: string;
};
export type DiagnosticExample = {
  timestamp: string | null;
  values: Record<string, string | number | null>;
};
export type ImportFinding = {
  check_name: string;
  error_code: string;
  severity: string;
  count: number;
  description: string;
  first_affected_timestamp: string | null;
  last_affected_timestamp: string | null;
  examples: DiagnosticExample[];
};
export type ValidationSummary = {
  invalid_ohlc_count: number;
  invalid_timestamp_count: number;
  misaligned_timestamp_count: number;
  duplicate_timestamp_count: number;
  nan_inf_count: number;
  suspicious_gap_count: number;
  weekend_closure_gap_count: number;
  unexpected_gap_count: number;
  crossed_quote_count: number;
  schema_error_count: number;
  parse_error_count: number;
};
export type ImportResult = {
  outcome: string;
  import_status: string;
  certification: string;
  dataset_id: string | null;
  import_id: string | null;
  filename: string;
  file_size: number;
  provider: string;
  symbol: string;
  format: string;
  data_type: string;
  price_type: string;
  source_timezone: string;
  timestamp_format: string;
  purpose: string;
  row_count: number;
  earliest_timestamp: string | null;
  latest_timestamp: string | null;
  original_sha256: string | null;
  dataset_fingerprint: string | null;
  duplicates: number;
  suspicious_gaps: number;
  weekend_gaps: number;
  crossed_quotes: number;
  invalid_rows: number;
  provenance: string;
  message: string;
  validation_summary: ValidationSummary;
  quarantine_reasons: ImportFinding[];
  rejection_reasons: ImportFinding[];
};
export type EvidenceStatus = {
  available: boolean;
  report_count: number;
  status: string;
};
export type AuditStatus = {
  checks: { name: string; status: string }[];
  journal_integrity: string;
  access_ledger_integrity: string;
  final_test_vault_state: string;
  broker_execution: string;
  optimization: string;
  machine_learning: string;
};
export type MarketStateStatus = {
  available: boolean;
  status: string;
  schema_version: string | null;
  engine_version: string | null;
  authorized_snapshot_count: number;
  message: string;
};
const base = (
  import.meta.env.VITE_SANDBOX_API_URL || "http://127.0.0.1:8765"
).replace(/\/$/, "");
async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${base}${path}`, {
    headers: { Accept: "application/json" },
  });
  if (!response.ok)
    throw new Error(`BACKEND REQUEST FAILED (${response.status})`);
  return response.json() as Promise<T>;
}
async function importHistory(
  file: File,
  metadata: ImportMetadata,
): Promise<ImportResult> {
  const body = new FormData();
  body.append("file", file, file.name);
  Object.entries(metadata).forEach(([k, v]) => body.append(k, v));
  const response = await fetch(`${base}/api/data/import`, {
    method: "POST",
    body,
    headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    const detail = await response
      .json()
      .catch(() => ({ detail: "Import failed" }));
    throw new Error(
      typeof detail.detail === "string" ? detail.detail : "Import failed",
    );
  }
  return response.json();
}
export const api = {
  system: () => get<SystemStatus>("/api/system/status"),
  strategies: () => get<StrategySummary[]>("/api/strategies"),
  strategy: (id: string) =>
    get<StrategyDetail>(`/api/strategies/${encodeURIComponent(id)}`),
  research: () => get<ResearchStatus>("/api/research/status"),
  data: () => get<DataStatus>("/api/data/status"),
  importHistory,
  importDiagnostics: (id: string) =>
    get<ImportResult>(
      `/api/data/imports/${encodeURIComponent(id)}/diagnostics`,
    ),
  evidence: () => get<EvidenceStatus>("/api/evidence/status"),
  audit: () => get<AuditStatus>("/api/audit/status"),
  marketState: () => get<MarketStateStatus>("/api/market-state/status"),
};
