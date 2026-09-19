"""Frozen pre-validation rules only: no data loading, fitting, or evaluation.

H3 has no user-specified materiality margin. Its effect-size ratios are strictly
 descriptive; a binary materiality verdict or a post-result cutoff is forbidden.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType

DISCOVERY_ARTIFACT_SHA256 = "52f5bf2c3d14ad7feee7bf1fa79c9968c974705fcf8bc7c11a3576448e536b98"
FEATURE_ARTIFACT_SHA256 = "4354250c1f98190fe30924d6133dea652c24abe1dcad4a9e671a1fe79be2422b"
DEFAULT_OUTPUT = Path("results/market_state/future_behavior_validation_contract_v1.json")


def _definition():
    return {
        "experiment": "MARKET_STATE_FUTURE_BEHAVIOUR_VALIDATION",
        "contract_version": "MARKET_STATE_FUTURE_BEHAVIOUR_VALIDATION_CONTRACT_V1",
        "status": "FROZEN_BEFORE_VALIDATION",
        "discovery_artifact": {
            "experiment": "FUTURE_MARKET_BEHAVIOUR_V1",
            "path": "results/market_state/future_behavior_v1.json",
            "sha256": DISCOVERY_ARTIFACT_SHA256,
        },
        "feature_artifact": {
            "path": "data/derived/features/EURUSD/M15/adac311828c7e0bcf58158269069689cf7178d3720805629eed481525eacd19d/features.parquet",
            "sha256": FEATURE_ARTIFACT_SHA256,
            "metadata_sha256": "532efc118bf13fabc01d37fd5912d757cfbaf1f2ead217c5c67de990538687d2",
        },
        "feature_schema": {
            "version": "MARKET_STATE_FEATURES_V1", "feature_count": 14,
            "ordered_names": ["er_4", "er_16", "rv_16", "atr_change_1", "atr_change_8",
                              "slope_atr_4", "slope_atr_16", "range_pos_16", "h1_er_8",
                              "h1_atr_change_4", "h1_slope_atr_8", "h3_er_8", "h3_atr_change_4", "h3_slope_atr_8"],
            "definition_authority": "sandbox.market_state.universal.market_state_v1.MARKET_STATE_FEATURES_V1",
            "changes_allowed": False,
        },
        "symbol": "EURUSD", "decision_timeframe": "M15",
        "model_configuration": {
            "class": "GaussianMixtureStateModel", "n_components": 3, "random_state": 42,
            "covariance_type": "full", "n_init": 5, "reg_covar": 1e-6, "max_iter": 300,
        },
        "scaler_configuration": {"class": "DiscoveryStandardScaler", "with_mean": True, "with_std": True},
        "discovery_fit_period": {
            "start_inclusive": "2016-01-01T00:00:00+00:00", "end_exclusive": "2023-01-01T00:00:00+00:00",
            "first_fitted_observation": "2016-01-05T09:00:00+00:00", "last_fitted_observation": "2022-12-30T22:00:00+00:00",
        },
        "validation_period": {"start_inclusive": "2023-01-01T00:00:00+00:00", "end_exclusive": "2025-01-01T00:00:00+00:00"},
        "final_holdout": {"start_inclusive": "2025-01-01T00:00:00+00:00", "access_allowed": False},
        "fitted_model_reuse": {
            "allowed_scaler_operation": "transform", "allowed_gmm_operations": ["predict", "predict_proba"],
            "fit_allowed": False, "partial_fit_allowed": False,
            "component_order": ["S0", "S1", "S2"], "relabeling_allowed": False,
            "validation_prerequisites": [
                "Verify discovery and feature artifact SHA256 before validation access.",
                "Verify immutable discovery-fitted scaler and GMM parameter artifacts and their checksums against discovery provenance.",
                "Preserve scaler mean_/scale_ and GMM weights_/means_/covariances_/precisions_cholesky_ without modification.",
                "Preserve original component indices; no occupancy or outcome-based relabeling.",
                "Record model serialization and library versions in a separate immutable execution manifest before validation access.",
                "If discovery-fitted parameters or their binding are unavailable, stop; do not fit a replacement model in validation.",
            ],
            "parameter_artifact_checksums_in_discovery_summary": None,
            "seed_alone_is_sufficient_identity": False,
        },
        "horizons": [{"bars": h, "minutes": h*15} for h in (1, 2, 4, 8, 12)],
        "state_ids": ["S0", "S1", "S2"],
        "price_reference": "close of the bar opening at T-15 minutes; feature timestamp T is its decision close",
        "formulas": {
            "signed_return": "C(T+H)/C(T)-1", "absolute_return": "abs(signed_return)",
            "mfe_up": "max(future highs)-C(T), not clamped", "mae_down": "C(T)-min(future lows), not clamped",
            "realized_volatility": "sqrt(sum(log(C_i/C_(i-1))**2)), i=1..H; unannualized",
            "flat": "C(T+H) == C(T) exactly", "standard_deviation": "sample, ddof=1",
            "cohens_d": "(mean_A-mean_B)/pooled sample SD; null if either n<2 or SD=0",
        },
        "pip_normalization": "Only explicit reliable symbol pip metadata; otherwise retain price/return units.",
        "continuity": {
            "bar_timestamp_convention": "UTC bar opens; UTC feature decision closes",
            "spacing_minutes": 15, "unchanged_continuity_id_required": True,
            "future_bars": "bar opens T through T+(H-1)*15 minutes",
            "reference_and_future_bars_within_validation": True,
            "all_outcome_close_timestamps_before": "2025-01-01T00:00:00+00:00",
            "bridge_weekends_or_missing_bars": False, "fabricate_or_pad_prices": False,
            "exclude_incomplete_horizons_independently": True,
            "exclusion_counts": ["missing_reference", "gap_or_segment_boundary", "end_of_validation_data"],
            "missing_features": "Exclude incomplete frozen vectors from assignments; report count; do not impute.",
            "yearly_occupancy_basis": "UTC state-assignment year; independent denominators for 2023 and 2024",
        },
        "hypotheses": {
            "H1_MAGNITUDE": {
                "expectation": "S0 has greater future absolute return than both S1 and S2.",
                "metric": "absolute_return", "comparisons": [["S0", "S1"], ["S0", "S2"]],
                "per_horizon_rule": {"statistic": "mean", "operator": ">", "combine": "AND"},
                "required_effect_sizes": "Cohen's d for S0-S1 and S0-S2 at every horizon",
            },
            "H2_VOLATILITY": {
                "expectation": "S0 has greater future realized volatility than both S1 and S2.",
                "metric": "realized_volatility", "comparisons": [["S0", "S1"], ["S0", "S2"]],
                "per_horizon_rule": {"statistic": "mean", "operator": ">", "combine": "AND"},
                "required_effect_sizes": "Cohen's d for S0-S1 and S0-S2 at every horizon",
            },
            "H3_DIRECTION": {
                "expectation": "Signed-return separation is much smaller than magnitude/volatility separation; no substantial unconditional directional separation expected.",
                "metric": "signed_return", "comparisons": [["S0", "S1"], ["S0", "S2"], ["S1", "S2"]],
                "required_effect_sizes": "Signed-return, absolute-return and realized-volatility Cohen's d for every pair and horizon",
                "required_ratios": ["abs(d_signed_return)/abs(d_absolute_return)", "abs(d_signed_return)/abs(d_realized_volatility)"],
                "ratio_zero_denominator_or_undefined_d": "null with reason; never coerce to zero or infinity",
                "materiality_margin": None,
                "assessment": "DESCRIPTIVE_ONLY_NO_BINARY_MATERIALITY_VERDICT",
                "report": "Report ratios and whether abs(d_signed) is smaller than each comparator; do not infer materiality or equivalence from a missing prespecified margin.",
                "post_validation_materiality_cutoff_allowed": False,
            },
        },
        "replication_criteria": {
            "applies_to": ["H1_MAGNITUDE", "H2_VOLATILITY"],
            "labels": ["replicated direction", "failed direction"],
            "replicated_direction": "Both strict mean inequalities hold at that horizon.",
            "failed_direction": "Either strict inequality does not hold, including a tie or unavailable mean.",
            "missing_comparison": "Count as failed direction; separately flag insufficient evidence, not evidence of reversal.",
            "overall_report": "For each hypothesis: X / 5 horizons replicated; always list all five horizons.",
            "denominator": 5, "exact_discovery_effect_size_required": False,
            "minimum_effect_size_threshold": None, "profitability_criterion": None,
            "hypothesis_tests": "None prescribed; no significance-only success criterion.",
        },
        "robustness_diagnostics": {
            "use": "Report only; never optimization or model-selection inputs.",
            "occupancy": "State counts/fractions over all eligible validation assignments, before future-window exclusions.",
            "posterior_confidence": "Frozen-model maximum posterior: mean, median, p10, p90 overall and per assigned state; no confidence filtering.",
            "continuity_exclusions": "Reason counts for every state/horizon; include incomplete-feature exclusions separately.",
            "observations": "Observation count for every state/horizon including zero counts.",
            "yearly_occupancy": {"years": [2023, 2024], "report": "counts and fractions per state independently; empty denominator gives null"},
            "state_transitions": "Counts and row-normalized probabilities in S0/S1/S2 order; consecutive exact 15-minute observations with unchanged continuity ID only; empty rows are zero with outgoing count zero.",
        },
        "prohibited_actions": [
            "refit_scaler", "refit_gmm", "partial_fit_or_parameter_updates", "change_n_components", "change_seed",
            "change_features_or_order", "change_horizons", "change_formulas", "select_favorable_horizons",
            "use_version_0", "use_tp_sl_pnl_or_profitability", "optimize_thresholds", "access_2025_plus_data",
            "modify_criteria_after_validation_results", "relabel_components", "derive_trading_rules",
            "feed_outcomes_into_features_scaler_model_or_assignment", "filter_states_or_rows_by_posterior_confidence",
        ],
        "execution_gate": "Contract creation/testing authorizes no validation run or data access; separate authorization and fitted-parameter verification required.",
    }


def _canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


_PAYLOAD = _definition()
CONTRACT_SHA256 = hashlib.sha256(_canonical(_PAYLOAD).encode("utf-8")).hexdigest()
_PAYLOAD["contract_sha256"] = CONTRACT_SHA256
_CONTRACT_JSON = json.dumps(_PAYLOAD, sort_keys=True, indent=2, allow_nan=False) + "\n"
VALIDATION_CONTRACT_V1 = _freeze(_PAYLOAD)
del _PAYLOAD


def contract_json() -> str:
    return _CONTRACT_JSON


def validate_contract(payload: dict) -> None:
    """Reject any changed rule/binding, even if a caller recomputes its digest."""
    if _canonical(payload) != _canonical(json.loads(_CONTRACT_JSON)):
        raise ValueError("contract differs from frozen Market State validation V1")


def verify_discovery_artifact(path: Path) -> None:
    """Read only the already-completed Discovery report, never feature/price rows."""
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != DISCOVERY_ARTIFACT_SHA256:
        raise ValueError("discovery artifact checksum differs from frozen contract")


def write_contract(output: Path = DEFAULT_OUTPUT) -> Path:
    """Atomic, create-only publication; identical retries are safe, changes refused."""
    output = Path(output)
    encoded = _CONTRACT_JSON.encode("utf-8")
    if output.exists():
        if output.read_bytes() != encoded:
            raise FileExistsError("refusing to overwrite a different frozen contract")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
        try:
            os.link(temporary, output)  # atomic create-if-absent, never overwrite
        except FileExistsError:
            if output.read_bytes() != encoded:
                raise FileExistsError("refusing to overwrite a different frozen contract")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output
