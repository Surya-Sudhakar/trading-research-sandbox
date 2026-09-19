import hashlib
import json

import pytest

from sandbox.market_state.discovery import validation_contract as contract
from sandbox.market_state.universal.market_state_v1 import MARKET_STATE_FEATURE_NAMES_V1, MARKET_STATE_FEATURE_SCHEMA_VERSION


def payload():
    return json.loads(contract.contract_json())


def test_frozen_schema_model_periods_and_horizons():
    value = payload()
    assert value["feature_schema"]["version"] == MARKET_STATE_FEATURE_SCHEMA_VERSION
    assert tuple(value["feature_schema"]["ordered_names"]) == MARKET_STATE_FEATURE_NAMES_V1
    assert value["feature_schema"]["feature_count"] == 14
    assert value["model_configuration"] == {
        "class":"GaussianMixtureStateModel", "n_components":3, "random_state":42,
        "covariance_type":"full", "n_init":5, "reg_covar":1e-6, "max_iter":300,
    }
    assert value["discovery_fit_period"]["start_inclusive"] == "2016-01-01T00:00:00+00:00"
    assert value["discovery_fit_period"]["end_exclusive"] == "2023-01-01T00:00:00+00:00"
    assert value["validation_period"] == {"start_inclusive":"2023-01-01T00:00:00+00:00", "end_exclusive":"2025-01-01T00:00:00+00:00"}
    assert value["final_holdout"] == {"start_inclusive":"2025-01-01T00:00:00+00:00", "access_allowed":False}
    assert value["state_ids"] == ["S0","S1","S2"]
    assert value["horizons"] == [{"bars":h,"minutes":h*15} for h in (1,2,4,8,12)]


def test_exact_discovery_bindings_and_digest():
    value = payload()
    assert value["discovery_artifact"]["sha256"] == "52f5bf2c3d14ad7feee7bf1fa79c9968c974705fcf8bc7c11a3576448e536b98"
    assert value["feature_artifact"]["sha256"] == "4354250c1f98190fe30924d6133dea652c24abe1dcad4a9e671a1fe79be2422b"
    digest = value.pop("contract_sha256")
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert digest == hashlib.sha256(encoded).hexdigest() == contract.CONTRACT_SHA256


def test_h1_h2_all_horizons_strict_direction_no_effect_threshold():
    value = payload()
    for name, metric in (("H1_MAGNITUDE","absolute_return"), ("H2_VOLATILITY","realized_volatility")):
        h = value["hypotheses"][name]
        assert h["metric"] == metric
        assert h["comparisons"] == [["S0","S1"],["S0","S2"]]
        assert h["per_horizon_rule"] == {"statistic":"mean","operator":">","combine":"AND"}
        assert "Cohen's d" in h["required_effect_sizes"]
    criteria = value["replication_criteria"]
    assert criteria["labels"] == ["replicated direction","failed direction"]
    assert criteria["denominator"] == 5
    assert criteria["minimum_effect_size_threshold"] is None
    assert criteria["exact_discovery_effect_size_required"] is False
    assert criteria["profitability_criterion"] is None
    assert "tie" in criteria["failed_direction"]
    assert "insufficient evidence" in criteria["missing_comparison"]


def test_h3_is_descriptive_with_all_pairs_and_no_cutoff():
    h3 = payload()["hypotheses"]["H3_DIRECTION"]
    assert h3["comparisons"] == [["S0","S1"],["S0","S2"],["S1","S2"]]
    assert h3["materiality_margin"] is None
    assert h3["assessment"] == "DESCRIPTIVE_ONLY_NO_BINARY_MATERIALITY_VERDICT"
    assert h3["post_validation_materiality_cutoff_allowed"] is False
    assert len(h3["required_ratios"]) == 2
    assert "null" in h3["ratio_zero_denominator_or_undefined_d"]


def test_transform_predict_only_and_no_component_relabeling():
    reuse = payload()["fitted_model_reuse"]
    assert reuse["allowed_scaler_operation"] == "transform"
    assert reuse["allowed_gmm_operations"] == ["predict","predict_proba"]
    assert reuse["fit_allowed"] is False
    assert reuse["partial_fit_allowed"] is False
    assert reuse["relabeling_allowed"] is False
    assert reuse["seed_alone_is_sufficient_identity"] is False
    assert reuse["parameter_artifact_checksums_in_discovery_summary"] is None
    assert any("stop" in item for item in reuse["validation_prerequisites"])


def test_frozen_formulas_and_gap_rules():
    value = payload()
    assert value["formulas"] == {
        "signed_return":"C(T+H)/C(T)-1", "absolute_return":"abs(signed_return)",
        "mfe_up":"max(future highs)-C(T), not clamped", "mae_down":"C(T)-min(future lows), not clamped",
        "realized_volatility":"sqrt(sum(log(C_i/C_(i-1))**2)), i=1..H; unannualized",
        "flat":"C(T+H) == C(T) exactly", "standard_deviation":"sample, ddof=1",
        "cohens_d":"(mean_A-mean_B)/pooled sample SD; null if either n<2 or SD=0",
    }
    rule = value["continuity"]
    assert rule["spacing_minutes"] == 15
    assert rule["unchanged_continuity_id_required"] is True
    assert rule["bridge_weekends_or_missing_bars"] is False
    assert rule["fabricate_or_pad_prices"] is False
    assert rule["all_outcome_close_timestamps_before"] == "2025-01-01T00:00:00+00:00"
    assert rule["exclude_incomplete_horizons_independently"] is True


def test_diagnostics_are_reporting_only():
    diagnostics = payload()["robustness_diagnostics"]
    assert set(diagnostics) == {"use","occupancy","posterior_confidence","continuity_exclusions","observations","yearly_occupancy","state_transitions"}
    assert diagnostics["yearly_occupancy"]["years"] == [2023,2024]
    assert "never optimization" in diagnostics["use"]
    assert "no confidence filtering" in diagnostics["posterior_confidence"]


@pytest.mark.parametrize("action", [
    "refit_scaler", "refit_gmm", "change_n_components", "change_seed", "change_features_or_order",
    "change_horizons", "change_formulas", "select_favorable_horizons", "use_version_0",
    "use_tp_sl_pnl_or_profitability", "optimize_thresholds", "access_2025_plus_data",
    "modify_criteria_after_validation_results", "relabel_components", "derive_trading_rules",
])
def test_prohibited_actions_frozen(action):
    assert action in payload()["prohibited_actions"]


@pytest.mark.parametrize("path,replacement", [
    (("model_configuration","n_components"),4),
    (("model_configuration","random_state"),7),
    (("feature_schema","ordered_names"),["er_4"]),
    (("horizons",),[{"bars":1,"minutes":15}]),
    (("formulas","signed_return"),"different"),
    (("validation_period","end_exclusive"),"2026-01-01T00:00:00+00:00"),
    (("final_holdout","access_allowed"),True),
    (("fitted_model_reuse","fit_allowed"),True),
    (("hypotheses","H1_MAGNITUDE","per_horizon_rule","operator"),">="),
    (("hypotheses","H3_DIRECTION","materiality_margin"),.5),
    (("discovery_artifact","sha256"),"0"*64),
    (("feature_artifact","sha256"),"0"*64),
    (("prohibited_actions",),[]),
])
def test_changes_rejected_even_after_digest_recomputed(path, replacement):
    value = payload()
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    value.pop("contract_sha256")
    value["contract_sha256"] = hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    with pytest.raises(ValueError, match="frozen"):
        contract.validate_contract(value)


def test_deep_immutability_and_export_isolation():
    with pytest.raises(TypeError):
        contract.VALIDATION_CONTRACT_V1["model_configuration"]["n_components"] = 4
    with pytest.raises(TypeError):
        contract.VALIDATION_CONTRACT_V1["horizons"][0]["bars"] = 2
    with pytest.raises(TypeError):
        contract.VALIDATION_CONTRACT_V1["state_ids"][0] = "OTHER"
    detached = payload()
    detached["hypotheses"].clear()
    assert len(payload()["hypotheses"]) == 3
    contract.validate_contract(payload())


def test_publication_is_deterministic_and_refuses_overwrite(tmp_path):
    target = tmp_path/"contract.json"
    contract.write_contract(target)
    original = target.read_bytes()
    contract.write_contract(target)
    assert target.read_bytes() == original == contract.contract_json().encode()
    contract.validate_contract(json.loads(original))
    target.write_text("other frozen contract")
    with pytest.raises(FileExistsError, match="refusing"):
        contract.write_contract(target)
    assert target.read_text() == "other frozen contract"


def test_failed_publication_does_not_leave_completed_artifact(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("injected publish failure")
    monkeypatch.setattr(contract.os,"link",fail)
    with pytest.raises(OSError, match="publish failure"):
        contract.write_contract(tmp_path/"contract.json")
    assert list(tmp_path.iterdir()) == []


def test_changed_discovery_binding_rejected(tmp_path):
    source = tmp_path/"discovery.json"
    source.write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        contract.verify_discovery_artifact(source)


def test_contract_creation_has_no_data_or_training_dependencies():
    import ast
    from pathlib import Path
    tree = ast.parse(Path(contract.__file__).read_text())
    imports = {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)}
    imports |= {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names}
    assert imports <= {"__future__","hashlib","json","os","pathlib","tempfile","types"}
    assert "authorizes no validation run or data access" in payload()["execution_gate"]
