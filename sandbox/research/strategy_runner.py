from __future__ import annotations

"""Minimum deterministic runner for JSON strategies over causal Discovery rows."""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from sandbox.config import Settings
from sandbox.execution.engine import BacktestEngine, RunResult, canonical_market_data
from sandbox.execution.metrics import summarize, yearly_breakdown
from sandbox.execution.models import Direction, EntryType, ExecutionConfig, PositionStatus, TradeIntent
from sandbox.market_state.discovery_dataset import DiscoveryDataset
from sandbox.market_state.discovery_projection import values_by_id
from sandbox.market_data.sanitation import frame_hash
from sandbox.partition.models import AccessContext, AccessOperation, PartitionManifest, PartitionRole
from sandbox.partition.service import PartitionService
from sandbox.research.canonical import canonical_json, sha256_canonical
from sandbox.research.compiled_strategy_execution import _evaluate
from sandbox.research.discovery_record_builder import DiscoveryRecordConfig, build_discovery_record_population
from sandbox.research.registry import ResearchRegistry
from sandbox.research.errors import ResearchError
from sandbox.strategies.dsl.conditions import (
    AndCondition,
    ComparisonCondition,
    ComparisonOperator,
    ValueReference,
)


STRATEGY_RUNNER_VERSION = "STRATEGY_RUNNER_V1"
OVERLAP_POLICY = "INDEPENDENT_INTENTS_OVERLAP_ALLOWED"
_OPERATOR_MAP = {
    "=": ComparisonOperator.EQ,
    "==": ComparisonOperator.EQ,
    "<": ComparisonOperator.LT,
    "<=": ComparisonOperator.LTE,
    ">": ComparisonOperator.GT,
    ">=": ComparisonOperator.GTE,
}
_CANDLE_INTERVALS = {"M1": 60, "M15": 900}
_TRADE_COLUMNS = (
    "trade_id", "direction", "signal_time", "entry_time", "entry_price",
    "stop_price", "target_price", "exit_time", "exit_price", "exit_reason",
    "gross_r", "net_r",
)


class RunnerModel(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False,
        revalidate_instances="always",
    )


class StrategyCondition(RunnerModel):
    field_id: str
    operator: Literal["=", "==", "<", "<=", ">", ">="]
    value: JsonValue

    @field_validator("field_id")
    @classmethod
    def predictor_id(cls, value: str) -> str:
        if not value.startswith("x.") or len(value) <= 2:
            raise ValueError("condition field_id must be an x.* predictor")
        return value

    @model_validator(mode="after")
    def operand_matches_operator(self):
        if self.operator in {"<", "<=", ">", ">="}:
            if type(self.value) not in (int, float) or isinstance(self.value, bool):
                raise ValueError("ordered comparisons require a numeric value")
            if not math.isfinite(float(self.value)):
                raise ValueError("condition value must be finite")
        elif self.value is not None and type(self.value) not in (bool, int, float, str):
            raise ValueError("equality requires a scalar JSON value")
        return self


class NextOpenEntry(RunnerModel):
    type: Literal["NEXT_OPEN"]


class FixedPipsStopLoss(RunnerModel):
    type: Literal["FIXED_PIPS"]
    value: float = Field(gt=0)


class RiskMultipleTakeProfit(RunnerModel):
    type: Literal["R_MULTIPLE"]
    value: float = Field(gt=0)


class StrategyRunnerSpec(RunnerModel):
    strategy_id: str
    partition_id: str | None = None
    direction: Direction
    conditions: tuple[StrategyCondition, ...] = Field(min_length=1)
    entry: NextOpenEntry
    stop_loss: FixedPipsStopLoss
    take_profit: RiskMultipleTakeProfit

    @field_validator("strategy_id")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("identifier must be nonblank")
        return value

    @field_validator("partition_id")
    @classmethod
    def optional_nonblank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("partition_id must be nonblank when supplied")
        return value

    @model_validator(mode="after")
    def unique_conditions(self):
        keys = [canonical_json(condition.model_dump(mode="json")) for condition in self.conditions]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate strategy condition")
        return self

    def condition_expression(self) -> AndCondition:
        return AndCondition(children=tuple(
            ComparisonCondition(
                reference=ValueReference(namespace="feature", key=condition.field_id),
                operator=_OPERATOR_MAP[condition.operator],
                value=condition.value,
            )
            for condition in self.conditions
        ))


@dataclass(frozen=True)
class StrategyRun:
    run_id: str
    run_fingerprint: str
    result_directory: Path
    strategy: StrategyRunnerSpec
    intents: tuple[TradeIntent, ...]
    engine_result: RunResult
    metrics: dict[str, Any]


def load_strategy(path: Path, *, available_feature_ids=None) -> StrategyRunnerSpec:
    """Load strict JSON, optionally validating against a concrete causal schema."""
    strategy = StrategyRunnerSpec.model_validate_json(Path(path).read_text(encoding="utf-8"))
    if available_feature_ids is not None:
        _validate_feature_ids(strategy, tuple(available_feature_ids))
    return strategy


def _partitions_for_dataset(service: PartitionService,
                            dataset_id: str) -> tuple[PartitionManifest, ...]:
    with service.registry.catalog.connection() as connection:
        rows = connection.execute(
            "SELECT record_json FROM data_partitions WHERE source_dataset_id=? "
            "ORDER BY partition_id", (dataset_id,)).fetchall()
    return tuple(PartitionManifest.model_validate_json(row[0]) for row in rows)


def _require_discovery_partition(manifest: PartitionManifest | None) -> PartitionManifest:
    if manifest is None:
        raise ValueError("partition_id does not identify an existing partition")
    if manifest.role is PartitionRole.VALIDATION:
        raise ValueError("Validation partition rejected: Strategy Runner V1 requires Discovery data")
    if manifest.role is PartitionRole.FINAL_TEST:
        raise ValueError("Final Holdout partition rejected: Strategy Runner V1 requires Discovery data")
    if manifest.role is not PartitionRole.DISCOVERY:
        raise ValueError("Strategy Runner V1 requires a Discovery partition")
    return manifest


def resolve_strategy_partition(service: PartitionService, strategy: StrategyRunnerSpec,
                               dataset_id: str | None = None) -> StrategyRunnerSpec:
    """Resolve one Discovery partition without exposing protected partition data."""
    if not isinstance(strategy, StrategyRunnerSpec):
        raise TypeError("strategy must be StrategyRunnerSpec")
    if dataset_id is not None and (not isinstance(dataset_id, str) or not dataset_id.strip()):
        raise ValueError("dataset_id must be a nonblank string")
    if strategy.partition_id is not None:
        manifest = _require_discovery_partition(service.get_partition(strategy.partition_id))
        if dataset_id is not None and manifest.source_dataset_id != dataset_id:
            raise ValueError(
                "strategy partition_id conflicts with --dataset-id: partition belongs to "
                f"dataset {manifest.source_dataset_id}")
        return strategy
    if dataset_id is None:
        raise ValueError("strategy must provide partition_id or CLI must provide --dataset-id")
    partitions = _partitions_for_dataset(service, dataset_id)
    discovery = tuple(item for item in partitions if item.role is PartitionRole.DISCOVERY)
    if len(discovery) > 1:
        identifiers = ", ".join(item.partition_id for item in discovery)
        raise ValueError(
            f"dataset {dataset_id} has multiple Discovery partitions ({identifiers}); "
            "specify partition_id explicitly")
    if len(discovery) == 1:
        return strategy.model_copy(update={"partition_id": discovery[0].partition_id})
    roles = {item.role for item in partitions}
    if PartitionRole.VALIDATION in roles:
        raise ValueError(
            f"Validation partition rejected for dataset {dataset_id}; "
            "create a Discovery partition first")
    if PartitionRole.FINAL_TEST in roles:
        raise ValueError(
            f"Final Holdout partition rejected for dataset {dataset_id}; "
            "create a Discovery partition first")
    raise ValueError(
        f"no Discovery partition exists for dataset {dataset_id}; "
        "create a Discovery partition first")


def _validate_feature_ids(strategy: StrategyRunnerSpec, field_ids: tuple[str, ...]) -> None:
    if len(field_ids) != len(set(field_ids)):
        raise ValueError("DiscoveryDataset has duplicate predictor field IDs")
    missing = sorted({condition.field_id for condition in strategy.conditions} - set(field_ids))
    if missing:
        raise ValueError("condition feature does not exist in causal dataset: " + ", ".join(missing))


def _condition_kinds(strategy: StrategyRunnerSpec) -> dict[str, str]:
    kinds: dict[str, str] = {}
    for condition in strategy.conditions:
        kind = "NUMERIC" if condition.operator in {"<", "<=", ">", ">="} else "CATEGORICAL"
        previous = kinds.setdefault(condition.field_id, kind)
        if previous != kind:
            raise ValueError("one field cannot mix numeric and categorical conditions")
    return kinds


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("signal timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _number(value: object, name: str) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _causal_dataset_fingerprint(dataset: DiscoveryDataset) -> str:
    rows = []
    for row in dataset.rows:
        anchor = row.record.anchor
        rows.append({
            "evidence_end_utc": _utc(anchor.evidence_end_utc),
            "reference_price": _number(anchor.outcome_anchor.reference_price, "reference price"),
            "predictors": [(item.field_id, item.value) for item in row.predictors],
        })
    return sha256_canonical({
        "version": "CAUSAL_DISCOVERY_EXECUTION_DATASET_V1",
        "predictor_field_ids": dataset.predictor_field_ids,
        "rows": rows,
    })


def _next_open_lookup(candles: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    market = canonical_market_data(candles)
    timestamps = pd.DatetimeIndex(market["timestamp_utc"])
    return market, timestamps.as_unit("ns").asi8


def build_strategy_intents(
    strategy: StrategyRunnerSpec,
    dataset: DiscoveryDataset,
    candles: pd.DataFrame,
    *,
    dataset_id: str,
    symbol: str,
    pip_size: float,
) -> tuple[TradeIntent, ...]:
    """Evaluate causal predictors and emit next-open intents; targets are never read."""
    if not isinstance(strategy, StrategyRunnerSpec) or not isinstance(dataset, DiscoveryDataset):
        raise TypeError("strategy and dataset must use runner domain types")
    if not dataset_id.strip() or not symbol.strip():
        raise ValueError("dataset_id and symbol must be nonblank")
    pip_size = _number(pip_size, "pip_size")
    if pip_size <= 0:
        raise ValueError("pip_size must be positive")
    _validate_feature_ids(strategy, dataset.predictor_field_ids)
    kinds = _condition_kinds(strategy)
    expression = strategy.condition_expression()
    market, timestamp_ns = _next_open_lookup(candles)
    opens = market["open"].to_numpy(dtype=np.float64, copy=False)
    indices = {field: dataset.predictor_field_ids.index(field) for field in kinds}
    strategy_fingerprint = sha256_canonical(strategy.model_dump(mode="json"))
    intents = []
    for row_index, row in enumerate(dataset.rows):
        if tuple(value.field_id for value in row.predictors) != dataset.predictor_field_ids:
            raise ValueError("predictor schema mismatch")
        values = values_by_id(row.predictors)
        selected = {field: values[field] for field in indices}
        if not _evaluate(expression, selected, kinds):
            continue
        signal_time = _utc(row.record.anchor.evidence_end_utc)
        signal_ns = pd.Timestamp(signal_time).value
        candle_index = int(np.searchsorted(timestamp_ns, signal_ns, side="left"))
        if candle_index >= len(market):
            continue
        entry_price = _number(float(opens[candle_index]), "next open")
        risk = strategy.stop_loss.value * pip_size
        sign = 1.0 if strategy.direction is Direction.LONG else -1.0
        stop = entry_price - sign * risk
        target = entry_price + sign * risk * strategy.take_profit.value
        if min(entry_price, stop, target) <= 0:
            raise ValueError("entry/stop/target prices must be positive")
        identity = sha256_canonical({
            "version": STRATEGY_RUNNER_VERSION,
            "strategy_fingerprint": strategy_fingerprint,
            "dataset_id": dataset_id,
            "source_row_index": row_index,
            "signal_time": signal_time,
        })
        intents.append(TradeIntent(
            trade_intent_id="STRAT-" + identity[:24],
            strategy_id=strategy.strategy_id,
            experiment_id=strategy.strategy_id,
            dataset_id=dataset_id,
            symbol=symbol,
            direction=strategy.direction,
            signal_timestamp=signal_time,
            requested_entry_type=EntryType.MARKET_NEXT_OPEN,
            stop_loss=stop,
            take_profit=target,
            metadata={
                "strategy_runner": STRATEGY_RUNNER_VERSION,
                "strategy_fingerprint": strategy_fingerprint,
                "source_row_index": row_index,
                "overlap_policy": OVERLAP_POLICY,
            },
        ))
    return tuple(intents)


def _closed_year_groups(frame: pd.DataFrame):
    if frame.empty:
        return ()
    copy = frame.copy()
    copy["year"] = pd.to_datetime(copy["exit_timestamp"], utc=True).dt.year
    return tuple(copy.groupby("year", sort=True))


def _metrics(strategy: StrategyRunnerSpec, run_id: str, frame: pd.DataFrame) -> dict[str, Any]:
    summary = summarize(frame)
    closed = frame if frame.empty else frame[frame["status"] == PositionStatus.CLOSED.value].copy()
    values = pd.to_numeric(closed.get("net_r", pd.Series(dtype=float)), errors="coerce").dropna()
    wins = values[values > 0]
    losses = values[values < 0]
    yearly = yearly_breakdown(frame)
    year_groups = dict(_closed_year_groups(closed))
    return {
        "strategy_id": strategy.strategy_id,
        "run_id": run_id,
        "trade_count": int(len(values)),
        "wins": int((values > 0).sum()),
        "losses": int((values < 0).sum()),
        "win_rate": float((values > 0).mean()) if len(values) else 0.0,
        "gross_r": float(summary["gross_R"]),
        "net_r": float(summary["net_R"]),
        "expectancy_r": float(summary["average_R"]),
        "profit_factor": summary["profit_factor"],
        "max_drawdown_r": -float(summary["maximum_drawdown_R"]),
        "average_win_r": float(wins.mean()) if len(wins) else 0.0,
        "average_loss_r": float(losses.mean()) if len(losses) else 0.0,
        "long_count": int((closed.get("direction", pd.Series(dtype=str)) == Direction.LONG.value).sum()),
        "short_count": int((closed.get("direction", pd.Series(dtype=str)) == Direction.SHORT.value).sum()),
        "rejected_count": int(summary["rejected_trades"]),
        "open_count": int((frame.get("status", pd.Series(dtype=str)) == PositionStatus.OPEN.value).sum()),
        "yearly_results": [
            {
                "year": item["year"], "trades": item["trades"],
                "win_rate": item["win_rate"], "net_r": item["net_R"],
                "expectancy_r": float(year_groups[item["year"]]["net_r"].mean()),
            }
            for item in yearly
        ],
    }


def _trade_csv(frame: pd.DataFrame) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_TRADE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for record in frame.to_dict(orient="records"):
        writer.writerow({
            "trade_id": record["trade_id"], "direction": record["direction"],
            "signal_time": record["signal_timestamp"], "entry_time": record["entry_timestamp"],
            "entry_price": record["entry_price"], "stop_price": record["stop_loss"],
            "target_price": record["take_profit"], "exit_time": record["exit_timestamp"],
            "exit_price": record["exit_price"], "exit_reason": record["exit_reason"],
            "gross_r": record["gross_r"], "net_r": record["net_r"],
        })
    return stream.getvalue()


def _report(metrics: dict[str, Any]) -> str:
    profit_factor = "undefined" if metrics["profit_factor"] is None else f'{metrics["profit_factor"]:.4f}'
    return (
        f'# Strategy Run {metrics["run_id"]}\n\n'
        f'- Strategy: {metrics["strategy_id"]}\n- Trades: {metrics["trade_count"]}\n'
        f'- Wins: {metrics["wins"]}\n- Losses: {metrics["losses"]}\n'
        f'- Win rate: {metrics["win_rate"]:.2%}\n- Net R: {metrics["net_r"]:.4f}\n'
        f'- Expectancy: {metrics["expectancy_r"]:.4f} R/trade\n'
        f'- Profit factor: {profit_factor}\n'
        f'- Max drawdown: {metrics["max_drawdown_r"]:.4f} R\n'
    )


def _persist(root: Path, run_id: str, run_fingerprint: str, strategy: StrategyRunnerSpec,
             engine_result: RunResult, metrics: dict[str, Any], dataset_fingerprint: str) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / run_id
    files = {
        "strategy.json": canonical_json(strategy.model_dump(mode="json")) + "\n",
        "trades.csv": _trade_csv(engine_result.frame()),
        "metrics.json": canonical_json(metrics) + "\n",
        "report.md": _report(metrics),
    }
    checksums = {name: hashlib.sha256(content.encode("utf-8")).hexdigest()
                 for name, content in files.items()}
    manifest = {
        "version": STRATEGY_RUNNER_VERSION, "run_id": run_id,
        "run_fingerprint": run_fingerprint,
        "engine_run_fingerprint": engine_result.run_fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "overlap_policy": OVERLAP_POLICY, "files": checksums,
    }
    files["manifest.json"] = canonical_json(manifest) + "\n"
    if destination.exists():
        if {path.name for path in destination.iterdir()} != set(files):
            raise ValueError("existing immutable strategy run has unexpected files")
        for name, content in files.items():
            path = destination / name
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                raise ValueError("existing immutable strategy run differs")
        return destination
    stage = root / (".pending-" + uuid4().hex)
    stage.mkdir()
    try:
        for name, content in files.items():
            (stage / name).write_text(content, encoding="utf-8", newline="")
        stage.rename(destination)
    finally:
        if stage.exists():
            if stage.resolve().parent != root.resolve() or stage.is_symlink():
                raise ValueError("unsafe strategy-run staging path")
            shutil.rmtree(stage)
    return destination


def run_strategy(strategy: StrategyRunnerSpec, dataset: DiscoveryDataset, candles: pd.DataFrame,
                 *, dataset_id: str, symbol: str, pip_size: float,
                 execution_config: ExecutionConfig,
                 root: Path = Path("results/strategy_runs")) -> StrategyRun:
    execution_config = ExecutionConfig.model_validate(execution_config.model_dump(mode="python"))
    dataset_fingerprint = _causal_dataset_fingerprint(dataset)
    intents = build_strategy_intents(
        strategy, dataset, candles, dataset_id=dataset_id, symbol=symbol, pip_size=pip_size)
    engine_result = BacktestEngine(execution_config).run(intents, candles)
    run_fingerprint = sha256_canonical({
        "version": STRATEGY_RUNNER_VERSION, "strategy": strategy,
        "dataset_fingerprint": dataset_fingerprint,
        "market_fingerprint": frame_hash(canonical_market_data(candles)),
        "engine_run_fingerprint": engine_result.run_fingerprint,
        "ledger": [record.model_dump(mode="json") for record in engine_result.ledger],
    })
    run_id = "strategy-run-" + run_fingerprint[:20]
    metrics = _metrics(strategy, run_id, engine_result.frame())
    directory = _persist(root, run_id, run_fingerprint, strategy, engine_result, metrics,
                         dataset_fingerprint)
    return StrategyRun(run_id, run_fingerprint, directory, strategy, intents, engine_result, metrics)


def run_partition_strategy(service: PartitionService, strategy: StrategyRunnerSpec, *,
                           root: Path = Path("results/strategy_runs"), pip_size: float = 0.0001,
                           execution_config: ExecutionConfig | None = None,
                           record_config: DiscoveryRecordConfig = DiscoveryRecordConfig(),
                           dataset_id: str | None = None) -> StrategyRun:
    strategy = resolve_strategy_partition(service, strategy, dataset_id)
    manifest = _require_discovery_partition(service.get_partition(strategy.partition_id))
    interval = _CANDLE_INTERVALS.get(manifest.timeframe)
    if interval is None:
        raise ValueError("unsupported Discovery partition timeframe")
    population = build_discovery_record_population(service, strategy.partition_id, record_config)
    candles = service.access(strategy.partition_id, AccessContext.DISCOVERY_CONTEXT,
                             AccessOperation.BACKTEST_EXECUTION, actor="strategy-runner-v1")
    if population.partition_fingerprint != manifest.partition_fingerprint:
        raise ValueError("partition changed while preparing strategy run")
    verification = service.verify_partition(strategy.partition_id)
    if not verification["checksum_valid"] or not verification["row_count_valid"]:
        raise ValueError("partition integrity failed before strategy execution")
    config = execution_config or ExecutionConfig(candle_interval_seconds=interval)
    if config.candle_interval_seconds != interval:
        raise ValueError("execution interval must match partition timeframe")
    return run_strategy(strategy, population.dataset, candles, dataset_id=strategy.partition_id,
                        symbol=manifest.symbol, pip_size=pip_size, execution_config=config, root=root)


def _print_summary(result: StrategyRun) -> None:
    metrics = result.metrics
    profit_factor = "undefined" if metrics["profit_factor"] is None else f'{metrics["profit_factor"]:.2f}'
    print(f'Strategy: {metrics["strategy_id"]}\nTrades: {metrics["trade_count"]}')
    print(f'Wins: {metrics["wins"]}\nLosses: {metrics["losses"]}')
    print(f'Win rate: {metrics["win_rate"]:.2%}\nNet R: {metrics["net_r"]:.2f}')
    print(f'Expectancy: {metrics["expectancy_r"]:.2f} R/trade')
    print(f'Profit factor: {profit_factor}\nMax drawdown: {metrics["max_drawdown_r"]:.2f} R')
    print(f'\nResults:\n{result.result_directory}/')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one JSON strategy on its Discovery partition")
    parser.add_argument("strategy", type=Path)
    parser.add_argument("--results-dir", type=Path, default=Path("results/strategy_runs"))
    parser.add_argument("--pip-size", type=float, default=0.0001)
    parser.add_argument("--dataset-id")
    parser.add_argument("--execution-config", type=Path)
    args = parser.parse_args(argv)
    try:
        settings = Settings.load()
        registry = ResearchRegistry(settings.catalog_path, project_root=Path.cwd())
        service = PartitionService(registry)
        strategy = load_strategy(args.strategy)
        config = (ExecutionConfig.model_validate_json(
            args.execution_config.read_text(encoding="utf-8"))
            if args.execution_config else None)
        result = run_partition_strategy(service, strategy, root=args.results_dir,
                                        pip_size=args.pip_size, execution_config=config,
                                        dataset_id=args.dataset_id)
        _print_summary(result)
        return 0
    except (OSError, ValueError, ResearchError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "STRATEGY_RUNNER_VERSION", "OVERLAP_POLICY", "StrategyCondition",
    "NextOpenEntry", "FixedPipsStopLoss", "RiskMultipleTakeProfit",
    "StrategyRunnerSpec", "StrategyRun", "load_strategy", "build_strategy_intents",
    "resolve_strategy_partition", "run_strategy", "run_partition_strategy", "main",
]
