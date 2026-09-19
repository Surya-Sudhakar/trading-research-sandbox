"""Frozen scaler/GMM bundle and an explicitly invoked Discovery-only creation job."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sandbox.market_state.universal.market_state_v1 import MARKET_STATE_FEATURE_NAMES_V1, MARKET_STATE_FEATURE_SCHEMA_VERSION
from sandbox.provenance import sha256_file
from .gmm import GaussianMixtureStateModel
from .scaler import DiscoveryStandardScaler, DISCOVERY_START_UTC, DISCOVERY_END_UTC
from .parameter_artifact import canonical, identity, check_identity, save_payload, load_payload, scaler_state, gmm_state, restore_scaler, restore_gmm

MODEL_ARTIFACT_VERSION = "FROZEN_MARKET_STATE_MODEL_V1"
MODEL_CONFIGURATION = {"n_components":3, "random_state":42, "covariance_type":"full", "n_init":5, "reg_covar":1e-6, "max_iter":300}
EXPECTED_COUNTS = (4755, 44972, 43935)
DEFAULT_OUTPUT = Path("results/market_state/frozen_market_state_model_v1")


def discovery_input_checksum(values, timestamps):
    """SHA256 of schema header, UTC ns int64 timestamps, and C-order float64 values (little endian)."""
    from .model import validate_state_matrix
    values = validate_state_matrix(values)
    stamps = pd.DatetimeIndex(timestamps)
    if (len(stamps) != len(values) or stamps.tz is None or stamps.hasnans or stamps.has_duplicates
            or not stamps.is_monotonic_increasing or (stamps < DISCOVERY_START_UTC).any()
            or (stamps >= DISCOVERY_END_UTC).any()):
        raise ValueError("ordered unique Discovery timestamps required")
    header = canonical({"schema":MARKET_STATE_FEATURE_SCHEMA_VERSION, "ordered_features":list(MARKET_STATE_FEATURE_NAMES_V1), "rows":len(values)})
    checksum = hashlib.sha256(header.encode("utf-8") + b"\n")
    checksum.update(stamps.as_unit("ns").asi8.astype("<i8").tobytes())
    checksum.update(np.asarray(values,dtype="<f8",order="C").tobytes())
    return checksum.hexdigest()


def _sha(value):
    if not isinstance(value,str) or len(value)!=64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("valid SHA256 binding required")
    return value


class FrozenMarketStateModel:
    """Prediction-only public API. All inputs use the declared frozen feature order."""

    def __init__(self, scaler, model, provenance):
        state = {**identity(), "model_artifact_version":MODEL_ARTIFACT_VERSION,
                 "scaler":scaler_state(scaler), "gmm":gmm_state(model), "provenance":provenance,
                 "state_ids":["S0","S1","S2"]}
        self._initialize(state)

    def _initialize(self, state, **expected):
        try:
            check_identity(state, **expected)
            if state["model_artifact_version"] != MODEL_ARTIFACT_VERSION or state["state_ids"] != ["S0","S1","S2"]:
                raise ValueError("frozen model version/state order mismatch")
            for key, value in MODEL_CONFIGURATION.items():
                if state["gmm"]["configuration"][key] != value:
                    raise ValueError("frozen GMM specification mismatch")
            metadata = state["provenance"]
            for key in ("feature_artifact_sha256","discovery_input_sha256","discovery_artifact_sha256"):
                _sha(metadata[key])
            if (metadata["discovery_start"] != DISCOVERY_START_UTC.isoformat()
                    or metadata["discovery_end_exclusive"] != DISCOVERY_END_UTC.isoformat()
                    or metadata["discovery_observation_count"] != state["scaler"]["n_samples_seen"]):
                raise ValueError("discovery period/count mismatch")
            first = datetime.fromisoformat(metadata["first_observation"])
            last = datetime.fromisoformat(metadata["last_observation"])
            if not DISCOVERY_START_UTC <= first <= last < DISCOVERY_END_UTC:
                raise ValueError("fitted observations outside Discovery")
            if metadata["first_observation"] != state["scaler"]["metadata"]["fit_start_utc"]:
                raise ValueError("scaler/provenance time mismatch")
            self._scaler = restore_scaler(state["scaler"], **expected)
            self._model = restore_gmm(state["gmm"], **expected)
            self._state_json = canonical(state)
        except (KeyError, TypeError) as exc:
            raise ValueError("incomplete frozen model artifact") from exc

    @property
    def metadata(self):
        return json.loads(self._state_json)

    def transform(self, values, *, feature_names=MARKET_STATE_FEATURE_NAMES_V1):
        if tuple(feature_names) != MARKET_STATE_FEATURE_NAMES_V1:
            raise ValueError("feature order mismatch")
        return self._scaler.transform(values)

    def predict(self, values, *, feature_names=MARKET_STATE_FEATURE_NAMES_V1):
        return self._model.predict(self.transform(values, feature_names=feature_names))

    def predict_proba(self, values, *, feature_names=MARKET_STATE_FEATURE_NAMES_V1):
        return self._model.predict_proba(self.transform(values, feature_names=feature_names))

    def save(self, path):
        return save_payload(path, MODEL_ARTIFACT_VERSION, self.metadata)

    @classmethod
    def load(cls, path, *, artifact_sha256, feature_artifact_sha256, discovery_input_sha256,
             feature_names=MARKET_STATE_FEATURE_NAMES_V1, schema_version=MARKET_STATE_FEATURE_SCHEMA_VERSION):
        state = load_payload(path, MODEL_ARTIFACT_VERSION, expected_sha256=_sha(artifact_sha256))
        try:
            if (state["provenance"]["feature_artifact_sha256"] != _sha(feature_artifact_sha256)
                    or state["provenance"]["discovery_input_sha256"] != _sha(discovery_input_sha256)):
                raise ValueError("feature artifact/discovery input checksum mismatch")
        except (KeyError, TypeError) as exc:
            raise ValueError("incomplete frozen model provenance") from exc
        result = cls.__new__(cls)
        result._initialize(state, feature_names=feature_names, schema_version=schema_version)
        return result


def read_discovery_features(path):
    from .future_behavior import _guard
    from .interpretation import _prepare_frame
    schema = pq.read_schema(path)
    _guard(schema.names)  # inspect schema, not future observations
    columns = ["timestamp", *MARKET_STATE_FEATURE_NAMES_V1]
    columns += [c for c in ("symbol","decision_timeframe","continuity_segment_id") if c in schema.names]
    frame = pd.read_parquet(path, columns=columns, filters=[
        ("timestamp", ">=", DISCOVERY_START_UTC), ("timestamp", "<", DISCOVERY_END_UTC)])
    prepared = _prepare_frame(frame)
    if not prepared.timestamp.eq(prepared.timestamp.dt.floor("15min")).all():
        raise ValueError("M15 aligned Discovery features required")
    return prepared


def create_discovery_artifact(feature_path, discovery_report, output=DEFAULT_OUTPUT):
    """One authorized reconstruction fit; abort on counts/spec/checksum mismatch, never retune."""
    from .validation_contract import verify_discovery_artifact, FEATURE_ARTIFACT_SHA256, DISCOVERY_ARTIFACT_SHA256
    feature_path, discovery_report, output = Path(feature_path), Path(discovery_report), Path(output)
    if output.exists():
        raise FileExistsError("frozen artifact directory exists; load it instead of fitting again")
    verify_discovery_artifact(discovery_report)
    report = json.loads(discovery_report.read_text(encoding="utf-8"))
    if sha256_file(feature_path) != FEATURE_ARTIFACT_SHA256:
        raise ValueError("frozen feature artifact SHA256 mismatch")
    if (report["feature_artifact_sha256"] != FEATURE_ARTIFACT_SHA256
            or tuple(report["feature_names"]) != MARKET_STATE_FEATURE_NAMES_V1
            or tuple(report["assigned_state_counts"][s] for s in ("S0","S1","S2")) != EXPECTED_COUNTS
            or any(report["state_model"][k] != v for k,v in MODEL_CONFIGURATION.items())):
        raise ValueError("discovery specification/count mismatch")
    frame = read_discovery_features(feature_path)
    values = frame.loc[:,MARKET_STATE_FEATURE_NAMES_V1].to_numpy(dtype=float)
    timestamps = tuple(frame.timestamp)
    input_sha = discovery_input_checksum(values, timestamps)
    if len(values) != sum(EXPECTED_COUNTS):
        raise ValueError("unexpected discovery observation count")
    print(f"Discovery rows loaded: {len(values)}; fitting scaler and GMM once", flush=True)
    scaler = DiscoveryStandardScaler().fit(values, timestamps)
    transformed = scaler.transform(values)
    model = GaussianMixtureStateModel(3, random_state=42).fit(transformed)
    labels = model.predict(transformed)
    probabilities = model.predict_proba(transformed)
    counts = tuple(int(x) for x in np.bincount(labels, minlength=3))
    if counts != EXPECTED_COUNTS or not model._model.converged_:
        raise ValueError(f"discovery reconstruction mismatch: counts={counts}; no artifact published")
    provenance = {
        "feature_artifact_sha256":FEATURE_ARTIFACT_SHA256, "discovery_input_sha256":input_sha,
        "discovery_artifact_sha256":DISCOVERY_ARTIFACT_SHA256,
        "discovery_start":DISCOVERY_START_UTC.isoformat(), "discovery_end_exclusive":DISCOVERY_END_UTC.isoformat(),
        "discovery_observation_count":len(values), "first_observation":timestamps[0].isoformat(), "last_observation":timestamps[-1].isoformat(),
        "state_counts":dict(zip(("S0","S1","S2"),counts)),
        "input_checksum_encoding":"canonical schema/order/row-count header + LF; UTC ns int64 LE timestamps; C-order float64 LE features",
        "reconstruction":"single fixed-specification Discovery fit; prior report has counts but no per-row assignments/parameter digest",
    }
    frozen = FrozenMarketStateModel(scaler, model, provenance)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".frozen-model-", dir=output.parent) as temporary:
        stage = Path(temporary)
        model_path = stage/"model.json"
        artifact_sha = frozen.save(model_path)
        loaded = FrozenMarketStateModel.load(model_path, artifact_sha256=artifact_sha,
                    feature_artifact_sha256=FEATURE_ARTIFACT_SHA256, discovery_input_sha256=input_sha)
        restored_transform = loaded.transform(values)
        restored_labels = loaded.predict(values)
        restored_proba = loaded.predict_proba(values)
        np.testing.assert_allclose(restored_transform, transformed, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(restored_labels, labels)
        np.testing.assert_allclose(restored_proba, probabilities, rtol=1e-12, atol=1e-12)
        if sha256_file(feature_path) != FEATURE_ARTIFACT_SHA256:
            raise ValueError("feature artifact changed during creation")
        verify_discovery_artifact(discovery_report)
        verification = {
            "status":"VERIFIED_DISCOVERY_ONLY", "model_artifact_sha256":artifact_sha,
            "discovery_input_sha256":input_sha, "observations":len(values), "state_counts":provenance["state_counts"],
            "transform_exact":bool(np.array_equal(restored_transform,transformed)),
            "transform_max_abs_error":float(np.max(np.abs(restored_transform-transformed))),
            "predictions_exact":bool(np.array_equal(restored_labels,labels)),
            "probabilities_exact":bool(np.array_equal(restored_proba,probabilities)),
            "probabilities_max_abs_error":float(np.max(np.abs(restored_proba-probabilities))),
            "tolerance":{"transform_atol":1e-12,"proba_atol":1e-12,"proba_rtol":1e-12},
            "labels_sha256":hashlib.sha256(np.asarray(labels,dtype="<i8").tobytes()).hexdigest(),
            "historical_assignment_equivalence":"original per-row assignment digest unavailable; counts match original report",
            "configuration":MODEL_CONFIGURATION,
        }
        (stage/"verification.json").write_text(json.dumps(verification,sort_keys=True,indent=2)+"\n",encoding="utf-8")
        # Publish both files only after all checks pass; no existing directory replaced.
        stage.rename(output)
    print(json.dumps(verification,sort_keys=True),flush=True)
    return verification


def main(argv=None):
    from .interpretation import DEFAULT_FEATURE_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features",type=Path,default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--discovery-report",type=Path,default=Path("results/market_state/future_behavior_v1.json"))
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    create_discovery_artifact(args.features,args.discovery_report,args.output)


if __name__ == "__main__":
    main()
