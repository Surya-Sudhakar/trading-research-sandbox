import React, { useEffect, useState } from "react";
import { api, DataStatus, ImportResult, StrategyDetail } from "./api";
const green = "#33ff33",
  dim = "#2b542b",
  amber = "#b5b500";
const Panel = ({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) => (
  <section
    style={{
      border: `1px solid ${dim}`,
      background: "#0a0f0a",
      padding: 12,
      marginBottom: 16,
    }}
  >
    <header
      className="glow"
      style={{
        borderBottom: `1px solid ${dim}`,
        paddingBottom: 4,
        marginBottom: 8,
      }}
    >
      [+] {title.toUpperCase()}
    </header>
    {children}
  </section>
);
const Row = ({ label, value }: { label: string; value: React.ReactNode }) => (
  <div
    style={{
      display: "flex",
      justifyContent: "space-between",
      gap: 18,
      padding: "3px 0",
      borderBottom: "1px dashed #152515",
    }}
  >
    <span style={{ color: dim }}>{label}</span>
    <strong style={{ textAlign: "right" }}>{value}</strong>
  </div>
);
const Empty = ({ children }: { children: React.ReactNode }) => (
  <div
    style={{
      padding: 32,
      textAlign: "center",
      border: `1px dashed ${dim}`,
      color: dim,
    }}
  >
    {children}
  </div>
);
const Badge = ({ children }: { children: React.ReactNode }) => (
  <span style={{ border: `1px solid ${green}`, padding: "1px 5px" }}>
    {children}
  </span>
);
function useLoad<T>(load: () => Promise<T>) {
  const [data, setData] = useState<T | null>(null),
    [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    load()
      .then((x) => active && setData(x))
      .catch(() => active && setError("BACKEND OFFLINE"));
    return () => {
      active = false;
    };
  }, [load]);
  return { data, error };
}
const State = ({
  error,
  data,
  children,
}: {
  error: string;
  data: unknown;
  children: React.ReactNode;
}) =>
  error ? (
    <Panel title="Connection Failure">
      <div style={{ color: "#ff3333", fontSize: 20 }}>BACKEND OFFLINE</div>
      <div style={{ color: dim }}>No mock data has been substituted.</div>
    </Panel>
  ) : !data ? (
    <div>[LOADING LIVE BACKEND STATE...]</div>
  ) : (
    <>{children}</>
  );
const Overview = () => {
  const x = useLoad(api.system);
  return (
    <State {...x}>
      <div className="divider">
        ================ LIVE BACKEND STATE ================
      </div>
      {x.data && (
        <>
          <Panel title="System">
            <Row label="BACKEND" value="ONLINE" />
            <Row
              label="BROKER"
              value={
                x.data.broker.connected
                  ? `${x.data.broker.company} / ${x.data.broker.server}`
                  : "DISCONNECTED"
              }
            />
            <Row
              label="BOUNDARY"
              value={x.data.broker.read_only ? "READ ONLY" : "INVALID"}
            />
            <Row label="INTEGRITY" value={x.data.integrity_status} />
            <Row
              label="VERIFIED TEST BASELINE"
              value={x.data.verified_test_count}
            />
          </Panel>
          <Panel title="Authoritative Counts">
            <Row label="REAL STRATEGIES" value={x.data.strategy_count} />
            <Row label="CATALOG DATASETS" value={x.data.dataset_count} />
            <Row label="RESEARCH PARTITIONS" value={x.data.partition_count} />
            <Row label="MODE" value={x.data.research_mode} />
          </Panel>
        </>
      )}
    </State>
  );
};
const Strategies = () => {
  const list = useLoad(api.strategies),
    [details, setDetails] = useState<StrategyDetail[]>([]);
  useEffect(() => {
    if (list.data)
      Promise.all(list.data.map((x) => api.strategy(x.strategy_id)))
        .then(setDetails)
        .catch(() => setDetails([]));
  }, [list.data]);
  return (
    <State error={list.error} data={list.data}>
      <div className="divider">
        ================ STRATEGY REGISTRY ================
      </div>
      {details.map((s) => (
        <Panel key={s.strategy_id} title={`${s.strategy_id} :: ${s.name}`}>
          <Row label="VERSION" value={s.version} />
          <Row
            label="STATUS"
            value={`${s.implementation_status} / ${s.research_status}`}
          />
          <Row label="TIMEFRAMES" value={s.required_timeframes.join(" + ")} />
          <p style={{ color: "#66ff66", margin: "8px 0" }}>{s.description}</p>
          {s.parameters.map((p) => (
            <Row
              key={p.name}
              label={`param.${p.name}`}
              value={String(p.default)}
            />
          ))}
          <div className="hash">CODE: {s.code_fingerprint}</div>
          <div className="hash">PARAMETERS: {s.parameter_fingerprint}</div>
        </Panel>
      ))}
      {list.data?.length === 0 && <Empty>NO STRATEGIES REGISTERED</Empty>}
    </State>
  );
};
const MarketState = () => {
  const x = useLoad(api.marketState);
  return (
    <State {...x}>
      {x.data && (
        <Panel title="Causal Market-State Framework">
          <Row
            label="AVAILABLE"
            value={String(x.data.available).toUpperCase()}
          />
          <Row label="STATUS" value={x.data.status} />
          <Row label="SCHEMA" value={x.data.schema_version || "UNAVAILABLE"} />
          <Row label="ENGINE" value={x.data.engine_version || "UNAVAILABLE"} />
          <Row
            label="AUTHORIZED SNAPSHOTS"
            value={x.data.authorized_snapshot_count}
          />
          <Empty>
            {x.data.message}
            <br />
            NO MARKET VALUES FABRICATED
          </Empty>
        </Panel>
      )}
    </State>
  );
};
const Research = () => {
  const x = useLoad(api.research);
  return (
    <State {...x}>
      {x.data && (
        <>
          <Panel title="Registered Research State">
            <Row label="PROGRAMS" value={x.data.program_count} />
            <Row label="HYPOTHESES" value={x.data.hypothesis_count} />
            <Row label="EXPERIMENTS" value={x.data.experiment_count} />
            <Row label="CANDIDATES" value={x.data.candidate_count} />
            <Row label="PARTITIONS" value={x.data.partition_count} />
          </Panel>
          <Panel title="Eligibility">
            <Row label="DISCOVERY" value={x.data.discovery_status} />
            <Row label="VALIDATION" value={x.data.validation_status} />
            <Row
              label="FINAL TEST"
              value={
                <span style={{ color: amber }}>{x.data.final_test_status}</span>
              }
            />
          </Panel>
        </>
      )}
    </State>
  );
};
const ImportDiagnostics = ({ result }: { result: ImportResult }) => {
  const s = result.validation_summary,
    findings = [...result.quarantine_reasons, ...result.rejection_reasons],
    examples = findings.flatMap((f) =>
      f.examples.map((x) => ({ finding: f, example: x })),
    ),
    eligibility =
      result.certification === "CERTIFIED"
        ? "ELIGIBLE FOR EXPLICIT STAGE 6 ASSIGNMENT"
        : result.certification === "DUPLICATE"
          ? "UNCHANGED — SEE EXISTING DATASET"
          : "NOT ELIGIBLE FOR RESEARCH";
  return (
    <>
      <Panel title="<<< Import Diagnostics >>>">
        <Row label="STATUS" value={result.certification} />
        <Row label="ROWS" value={result.row_count} />
        <Row
          label="DATE RANGE"
          value={`${result.earliest_timestamp || "N/A"} — ${result.latest_timestamp || "N/A"}`}
        />
        <Row
          label="TIMESTAMPS"
          value={`${s.invalid_timestamp_count} INVALID / ${result.timestamp_format}`}
        />
        <Row
          label="ALIGNMENT"
          value={`${s.misaligned_timestamp_count} MISALIGNED`}
        />
        <Row label="OHLC" value={`${s.invalid_ohlc_count} INVALID`} />
        <Row label="DUPLICATES" value={s.duplicate_timestamp_count} />
        <Row
          label="GAPS"
          value={`${s.suspicious_gap_count} SUSPICIOUS / ${s.weekend_closure_gap_count} CLOSURE`}
        />
        <Row label="SCHEMA" value={`${s.schema_error_count} ERROR(S)`} />
        <Row label="CERTIFICATION" value={result.certification} />
        <Row label="RESEARCH ELIGIBILITY" value={eligibility} />
        <p
          className={
            result.certification === "CERTIFIED"
              ? ""
              : result.certification === "DUPLICATE"
                ? ""
                : "failure"
          }
        >
          {result.message}
        </p>
      </Panel>
      <Panel title="<<< Findings >>>">
        {findings.length === 0 ? (
          <Empty>NO VALIDATION FINDINGS</Empty>
        ) : (
          findings.map((f, i) => (
            <div className="finding" key={`${f.error_code}-${i}`}>
              <Row label="SEVERITY" value={f.severity} />
              <Row label="CHECK" value={f.check_name} />
              <Row label="COUNT" value={f.count} />
              <Row label="DESCRIPTION" value={f.description} />
              <Row
                label="FIRST OCCURRENCE"
                value={f.first_affected_timestamp || "UNAVAILABLE"}
              />
              <Row
                label="LAST OCCURRENCE"
                value={f.last_affected_timestamp || "UNAVAILABLE"}
              />
            </div>
          ))
        )}
      </Panel>
      <Panel title="<<< Representative Examples >>>">
        {examples.length === 0 ? (
          <Empty>NO BOUNDED EXAMPLES AVAILABLE</Empty>
        ) : (
          examples.map(({ finding, example }, i) => (
            <div className="example" key={`${finding.error_code}-example-${i}`}>
              <Row
                label="CHECK / TIMESTAMP"
                value={`${finding.check_name} / ${example.timestamp || "UNAVAILABLE"}`}
              />
              <Row
                label="VALUES"
                value={
                  Object.entries(example.values)
                    .map(([k, v]) => `${k}=${v}`)
                    .join(" | ") || "NONE"
                }
              />
            </div>
          ))
        )}
      </Panel>
    </>
  );
};
const Data = () => {
  const [data, setData] = useState<DataStatus | null>(null),
    [offline, setOffline] = useState(""),
    [file, setFile] = useState<File | null>(null),
    [provider, setProvider] = useState("DUKASCOPY"),
    [symbol, setSymbol] = useState("EURUSD"),
    [dataType, setDataType] = useState("M1"),
    [priceType, setPriceType] = useState("BID"),
    [timezone, setTimezone] = useState("UTC"),
    [purpose, setPurpose] = useState("UNASSIGNED_RESEARCH_DATA"),
    [timestampFormat, setTimestampFormat] = useState("ISO8601"),
    [interpretationReason, setInterpretationReason] = useState(""),
    [phase, setPhase] = useState("IDLE"),
    [result, setResult] = useState<ImportResult | null>(null),
    [failure, setFailure] = useState("");
  const refresh = () =>
    api
      .data()
      .then(setData)
      .catch(() => setOffline("BACKEND OFFLINE"));
  useEffect(() => {
    void refresh();
  }, []);
  const inspect = (datasetId: string) => {
    setFailure("");
    api
      .importDiagnostics(datasetId)
      .then(setResult)
      .catch(() => setFailure("DIAGNOSTICS UNAVAILABLE"));
  };
  const choose = (candidate: File | undefined) => {
    if (candidate) {
      setFile(candidate);
      setResult(null);
      setFailure("");
    }
  };
  const submit = async () => {
    if (!file) return;
    setPhase("UPLOADING / VALIDATING / CERTIFYING");
    setFailure("");
    try {
      const value = await api.importHistory(file, {
        provider,
        symbol,
        data_type: dataType,
        price_type: priceType,
        source_timezone: timezone,
        purpose,
        timestamp_format: timestampFormat,
        ...(interpretationReason.trim()
          ? { interpretation_reason: interpretationReason }
          : {}),
      });
      setResult(value);
      setPhase("COMPLETE");
      await refresh();
    } catch (e) {
      setFailure(e instanceof Error ? e.message : "Import failed");
      setPhase("FAILED");
    }
  };
  const interpretationReasonRequired = Boolean(
    result?.rejection_reasons.some(
      (x) => x.error_code === "MULTIPLE_INTERPRETATIONS_REASON_REQUIRED",
    ),
  );
  return (
    <State error={offline} data={data}>
      <div className="divider">
        ================ CONTROLLED HISTORICAL DATA IMPORT ================
      </div>
      <Panel title="<<< Import Historical Data >>>">
        <div
          className="drop"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            choose(e.dataTransfer.files[0]);
          }}
        >
          DROP HISTORICAL DATA FILE HERE
          <br />
          <label className="fileButton">
            [ SELECT FILE ]
            <input
              type="file"
              accept=".csv,.gz,.parquet"
              onChange={(e) => choose(e.target.files?.[0])}
            />
          </label>
          <br />
          <small>CSV / CSV.GZ / PARQUET</small>
        </div>
        <div className="formgrid">
          <label>
            PROVIDER
            <select
              value={provider}
              onChange={(e) => setProvider(e.target.value)}
            >
              {["DUKASCOPY", "TRUEFX", "HISTDATA", "OTHER"].map((x) => (
                <option key={x}>{x}</option>
              ))}
            </select>
          </label>
          <label>
            SYMBOL
            <input
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            />
          </label>
          <label>
            DATA TYPE
            <select
              value={dataType}
              onChange={(e) => setDataType(e.target.value)}
            >
              <option>TICK</option>
              <option>M1</option>
              <option>M15</option>
            </select>
          </label>
          <label>
            PRICE TYPE
            <select
              value={priceType}
              onChange={(e) => setPriceType(e.target.value)}
            >
              {["BID", "ASK", "MID", "BID_ASK", "UNKNOWN"].map((x) => (
                <option key={x}>{x}</option>
              ))}
            </select>
          </label>
          <label>
            TIMEZONE
            <input
              value={timezone}
              onChange={(e) => setTimezone(e.target.value)}
            />
          </label>
          <label>
            TIMESTAMP FORMAT
            <select
              value={timestampFormat}
              onChange={(e) => setTimestampFormat(e.target.value)}
            >
              <option>ISO8601</option>
              <option>UNIX_MILLISECONDS</option>
            </select>
          </label>
          <div
            className={`interpretationReasonField ${interpretationReasonRequired ? "failure" : ""}`}
          >
            <label>
              INTERPRETATION REASON{interpretationReasonRequired ? " *" : ""}
            </label>

            <textarea
              value={interpretationReason}
              maxLength={500}
              required={interpretationReasonRequired}
              onChange={(e) => setInterpretationReason(e.target.value)}
              placeholder="Enter interpretation reason..."
            />

            <small>
              Required when the same source file is intentionally reprocessed
              using different material interpretation metadata. This records an
              audited interpretation; it is not a validation bypass.
            </small>
          </div>
          <label>
            PURPOSE
            <select
              value={purpose}
              onChange={(e) => setPurpose(e.target.value)}
            >
              <option>ENGINEERING_DIAGNOSTIC</option>
              <option>UNASSIGNED_RESEARCH_DATA</option>
            </select>
          </label>
        </div>
        {file && (
          <div className="preview">
            <Row label="FILE NAME" value={file.name} />
            <Row
              label="FILE SIZE"
              value={`${file.size.toLocaleString()} bytes`}
            />
            <Row label="PROVIDER" value={provider} />
            <Row label="SYMBOL" value={symbol} />
            <Row label="DATA TYPE" value={dataType} />
            <Row label="PRICE TYPE" value={priceType} />
            <Row label="TIMEZONE" value={timezone} />
            <Row label="TIMESTAMP FORMAT" value={timestampFormat} />
            {interpretationReason.trim() && (
              <Row label="INTERPRETATION REASON" value={interpretationReason} />
            )}
            <Row label="PURPOSE" value={purpose} />
          </div>
        )}
        <button
          className="importButton"
          disabled={!file || phase.includes("ING")}
          onClick={submit}
        >
          [ IMPORT & CERTIFY ]
        </button>
        <div className="phase">STATUS: {phase}</div>
        {failure && <div className="failure">{failure}</div>}
        {result && (
          <Panel title="Import Result">
            <Row label="IMPORT STATUS" value={result.import_status} />
            <Row label="CERTIFICATION" value={result.certification} />
            <Row label="DATASET ID" value={result.dataset_id || "NONE"} />
            <Row
              label="PROVIDER / SYMBOL"
              value={`${result.provider} / ${result.symbol}`}
            />
            <Row
              label="FORMAT / TYPE / PRICE"
              value={`${result.format} / ${result.data_type} / ${result.price_type}`}
            />
            <Row
              label="TIMEZONE / PURPOSE"
              value={`${result.source_timezone} / ${result.purpose}`}
            />
            <Row label="ROWS" value={result.row_count} />
            <Row
              label="EARLIEST / LATEST"
              value={`${result.earliest_timestamp || "N/A"} — ${result.latest_timestamp || "N/A"}`}
            />
            <Row
              label="DUPLICATES / INVALID"
              value={`${result.duplicates} / ${result.invalid_rows}`}
            />
            <Row
              label="SUSPICIOUS / WEEKEND GAPS"
              value={`${result.suspicious_gaps} / ${result.weekend_gaps}`}
            />
            <Row label="CROSSED QUOTES" value={result.crossed_quotes} />
            <div className="hash">
              ORIGINAL SHA-256: {result.original_sha256}
            </div>
            <div className="hash">
              DATASET FINGERPRINT: {result.dataset_fingerprint}
            </div>
            <p
              className={
                result.certification === "QUARANTINED" ? "failure" : ""
              }
            >
              {result.message}
            </p>
            <p style={{ color: dim }}>
              DECLARED_PROVENANCE means provider identity was supplied by the
              user and is not cryptographically verified by that provider.
            </p>
          </Panel>
        )}
        {result && <ImportDiagnostics result={result} />}
      </Panel>
      <div className="divider">
        ================ DATASET CATALOG ================
      </div>
      {data && (
        <>
          <Panel title="Data Foundation">
            <Row
              label="IC MARKETS"
              value={
                data.broker.connected
                  ? "CONNECTED / READ ONLY"
                  : "OFFLINE / READ ONLY"
              }
            />
            <Row label="CATALOG DATASETS" value={data.dataset_count} />
            <Row
              label="EXTERNAL IMPORT GATEWAY"
              value={
                data.external_import_gateway_available
                  ? "AVAILABLE"
                  : "UNAVAILABLE"
              }
            />
            <Row
              label="EXTERNAL IMPORT RECORDS"
              value={data.external_production_dataset_count}
            />
            <Row
              label="RESEARCH PARTITIONS"
              value={data.research_partition_count}
            />
          </Panel>
          {data.imported_datasets.map((d) => (
            <Panel key={d.dataset_id} title={d.dataset_id}>
              <button onClick={() => inspect(d.dataset_id)}>
                [ INSPECT SAFE DIAGNOSTICS ]
              </button>
              <Row
                label="PROVIDER / SYMBOL"
                value={`${d.provider} / ${d.symbol}`}
              />
              <Row
                label="TYPE / PRICE"
                value={`${d.data_type} / ${d.price_type}`}
              />
              <Row label="DATE RANGE" value={d.date_range} />
              <Row label="ROWS" value={d.row_count} />
              <Row
                label="CERTIFICATION / PURPOSE"
                value={`${d.certification} / ${d.purpose}`}
              />
              <div className="hash">FINGERPRINT: {d.fingerprint}</div>
            </Panel>
          ))}
          {data.datasets
            .filter(
              (d) =>
                !data.imported_datasets.some(
                  (i) => i.dataset_id === d.dataset_id,
                ),
            )
            .map((d) => (
              <Panel key={d.dataset_id} title={d.dataset_id}>
                <Row label="SOURCE" value={`${d.broker} / ${d.server}`} />
                <Row label="MARKET" value={`${d.symbol} ${d.timeframe}`} />
                <Row label="ROWS" value={d.row_count} />
                <Row label="RANGE" value={`${d.start} — ${d.end}`} />
                <div className="hash">SHA-256: {d.checksum}</div>
              </Panel>
            ))}
        </>
      )}
    </State>
  );
};
const Evidence = () => {
  const x = useLoad(api.evidence);
  return (
    <State {...x}>
      {x.data && (
        <Panel title="Stage 7 Evidence">
          <Row label="REPORT COUNT" value={x.data.report_count} />
          <Empty>
            {x.data.status.replaceAll("_", " ")}
            <br />
            NO PROFITABILITY IS CALCULATED CLIENT-SIDE
          </Empty>
        </Panel>
      )}
    </State>
  );
};
const Audit = () => {
  const x = useLoad(api.audit);
  return (
    <State {...x}>
      {x.data && (
        <>
          <Panel title="Stages & Interfaces">
            {x.data.checks.map((c) => (
              <Row
                key={c.name}
                label={c.name.replaceAll("_", " ")}
                value={<Badge>{c.status}</Badge>}
              />
            ))}
          </Panel>
          <Panel title="Protection Controls">
            <Row label="RESEARCH JOURNAL" value={x.data.journal_integrity} />
            <Row label="ACCESS LEDGER" value={x.data.access_ledger_integrity} />
            <Row
              label="FINAL-TEST VAULT"
              value={x.data.final_test_vault_state}
            />
            <Row label="BROKER EXECUTION" value={x.data.broker_execution} />
            <Row label="OPTIMIZATION" value={x.data.optimization} />
            <Row label="MACHINE LEARNING" value={x.data.machine_learning} />
          </Panel>
        </>
      )}
    </State>
  );
};
const tabs = [
  "OVERVIEW",
  "STRATEGIES",
  "MARKET STATE",
  "RESEARCH",
  "DATA",
  "EVIDENCE",
  "AUDIT",
];
export default function App() {
  const [tab, setTab] = useState("OVERVIEW");
  return (
    <div className="app">
      <div className="crt" />
      <header>
        <div className="glow">RESEARCH WORKSTATION</div>
        <small>
          HTTP LOCALHOST // PYTHON AUTHORITATIVE // LIVE BACKEND STATE
        </small>
      </header>
      <nav>
        {tabs.map((x) => (
          <button
            className={tab === x ? "active" : ""}
            key={x}
            onClick={() => setTab(x)}
          >
            {tab === x ? `> ${x} <` : x}
          </button>
        ))}
      </nav>
      <main>
        {tab === "OVERVIEW" && <Overview />}
        {tab === "STRATEGIES" && <Strategies />}
        {tab === "MARKET STATE" && <MarketState />}
        {tab === "RESEARCH" && <Research />}
        {tab === "DATA" && <Data />}
        {tab === "EVIDENCE" && <Evidence />}
        {tab === "AUDIT" && <Audit />}
      </main>
      <footer>
        <span>NO EXECUTION CAPABILITY EXPOSED</span>
      </footer>
    </div>
  );
}
