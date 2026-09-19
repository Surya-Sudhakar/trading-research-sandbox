"""Explicit, non-executable JSON serialization of fitted discovery parameters."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np
import scipy
import sklearn

from sandbox.market_state.universal.market_state_v1 import MARKET_STATE_FEATURE_NAMES_V1, MARKET_STATE_FEATURE_SCHEMA_VERSION

FORMAT_VERSION = "MARKET_STATE_PARAMETERS_V1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def identity():
    return {"feature_schema_version": MARKET_STATE_FEATURE_SCHEMA_VERSION,
            "feature_names": list(MARKET_STATE_FEATURE_NAMES_V1),
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}}


def check_identity(state, *, feature_names=MARKET_STATE_FEATURE_NAMES_V1,
                   schema_version=MARKET_STATE_FEATURE_SCHEMA_VERSION):
    if schema_version != MARKET_STATE_FEATURE_SCHEMA_VERSION or state["feature_schema_version"] != schema_version:
        raise ValueError("feature schema mismatch")
    if tuple(feature_names) != MARKET_STATE_FEATURE_NAMES_V1 or tuple(state["feature_names"]) != tuple(feature_names):
        raise ValueError("feature order mismatch")
    if state["versions"] != identity()["versions"]:
        raise ValueError("library/runtime version mismatch; cross-version loading is not verified")


def save_payload(path, kind, state):
    envelope = {"format_version": FORMAT_VERSION, "kind": kind, "state": state, "payload_sha256": digest(state)}
    encoded = (canonical(envelope) + "\n").encode("utf-8")
    path = Path(path)
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError("refusing to overwrite different model artifact")
        return hashlib.sha256(encoded).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return hashlib.sha256(encoded).hexdigest()


def load_payload(path, kind, *, expected_sha256=None):
    raw = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("artifact SHA256 mismatch")
    try:
        envelope = json.loads(raw)
        if (set(envelope) != {"format_version", "kind", "state", "payload_sha256"}
                or envelope["format_version"] != FORMAT_VERSION or envelope["kind"] != kind
                or digest(envelope["state"]) != envelope["payload_sha256"]):
            raise ValueError("invalid or corrupted parameter artifact")
        return envelope["state"]
    except (KeyError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("invalid or incomplete parameter artifact") from exc


def array(value, shape):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError("invalid parameter array shape or nonfinite value")
    result.setflags(write=False)
    return result


def scaler_state(scaler):
    metadata = scaler.metadata
    fitted = scaler._scaler
    return {**identity(), "configuration": fitted.get_params(),
            "metadata": {**asdict(metadata), "feature_names": list(metadata.feature_names),
                         "fit_start_utc": metadata.fit_start_utc.isoformat(),
                         "fit_end_exclusive_utc": metadata.fit_end_exclusive_utc.isoformat()},
            "mean": fitted.mean_.tolist(), "scale": fitted.scale_.tolist(), "variance": fitted.var_.tolist(),
            "n_samples_seen": int(fitted.n_samples_seen_), "n_features_in": int(fitted.n_features_in_)}


def restore_scaler(state, **expected):
    from .scaler import DiscoveryStandardScaler, DiscoveryScalerMetadata, DISCOVERY_START_UTC, DISCOVERY_END_UTC
    try:
        check_identity(state, **expected)
        result = DiscoveryStandardScaler()
        if state["configuration"] != result._scaler.get_params() or state["n_features_in"] != 14:
            raise ValueError("scaler configuration mismatch")
        count = state["n_samples_seen"]
        if type(count) is not int or count < 2:
            raise ValueError("invalid scaler observation count")
        metadata = state["metadata"]
        first, end = datetime.fromisoformat(metadata["fit_start_utc"]), datetime.fromisoformat(metadata["fit_end_exclusive_utc"])
        if (first.tzinfo is None or not DISCOVERY_START_UTC <= first < DISCOVERY_END_UTC or end != DISCOVERY_END_UTC
                or metadata["schema_version"] != MARKET_STATE_FEATURE_SCHEMA_VERSION
                or tuple(metadata["feature_names"]) != MARKET_STATE_FEATURE_NAMES_V1 or metadata["observation_count"] != count):
            raise ValueError("invalid scaler discovery metadata")
        fitted = result._scaler
        fitted.mean_ = array(state["mean"], (14,))
        fitted.scale_ = array(state["scale"], (14,))
        fitted.var_ = array(state["variance"], (14,))
        if (fitted.scale_ <= 0).any() or (fitted.var_ < 0).any():
            raise ValueError("invalid scaler scale/variance")
        fitted.n_samples_seen_ = np.int64(count)
        fitted.n_features_in_ = 14
        result._metadata = DiscoveryScalerMetadata(MARKET_STATE_FEATURE_SCHEMA_VERSION, MARKET_STATE_FEATURE_NAMES_V1, first, end, count)
        return result
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("incomplete or invalid scaler artifact") from exc


def gmm_state(model):
    model._require_fit()
    fitted = model._model
    arrays = {name: getattr(fitted, name + "_").tolist() for name in
              ("weights", "means", "covariances", "precisions", "precisions_cholesky")}
    return {**identity(), "configuration": fitted.get_params(), "arrays": arrays,
            "convergence": {"converged": bool(fitted.converged_), "n_iter": int(fitted.n_iter_),
                            "lower_bound": float(fitted.lower_bound_),
                            "lower_bounds": list(getattr(fitted, "lower_bounds_", []))},
            "n_features_in": int(fitted.n_features_in_)}


def restore_gmm(state, **expected):
    from .gmm import GaussianMixtureStateModel
    try:
        check_identity(state, **expected)
        config = state["configuration"]
        result = GaussianMixtureStateModel(config["n_components"], random_state=config["random_state"],
                  n_init=config["n_init"], reg_covar=config["reg_covar"], max_iter=config["max_iter"])
        if config != result._model.get_params() or state["n_features_in"] != 14:
            raise ValueError("GMM configuration mismatch")
        count = result.n_components
        fitted = result._model
        for name, shape in (("weights",(count,)), ("means",(count,14)), ("covariances",(count,14,14)),
                            ("precisions",(count,14,14)), ("precisions_cholesky",(count,14,14))):
            setattr(fitted, name + "_", array(state["arrays"][name], shape))
        if (fitted.weights_ <= 0).any() or not np.isclose(fitted.weights_.sum(), 1, rtol=1e-12, atol=1e-12):
            raise ValueError("invalid GMM weights")
        for cov, precision, chol in zip(fitted.covariances_, fitted.precisions_, fitted.precisions_cholesky_):
            if (not np.allclose(cov, cov.T, rtol=1e-12, atol=1e-12)
                    or not np.allclose(chol, np.triu(chol), rtol=0, atol=1e-14)
                    or (np.diag(chol) <= 0).any()
                    or not np.allclose(chol @ chol.T, precision, rtol=1e-10, atol=1e-10)
                    or not np.allclose(cov @ precision, np.eye(14), rtol=1e-7, atol=1e-7)):
                raise ValueError("inconsistent GMM covariance/precision parameters")
            np.linalg.cholesky(cov)
        convergence = state["convergence"]
        if (type(convergence["converged"]) is not bool or type(convergence["n_iter"]) is not int
                or not 1 <= convergence["n_iter"] <= config["max_iter"]
                or not np.isfinite(convergence["lower_bound"])):
            raise ValueError("invalid GMM convergence metadata")
        fitted.converged_ = convergence["converged"]
        fitted.n_iter_ = convergence["n_iter"]
        fitted.lower_bound_ = convergence["lower_bound"]
        fitted.lower_bounds_ = array(convergence["lower_bounds"], (len(convergence["lower_bounds"]),)).tolist()
        fitted.n_features_in_ = 14
        result._fitted = True
        return result
    except (KeyError, TypeError, OverflowError, np.linalg.LinAlgError) as exc:
        raise ValueError("incomplete or invalid GMM artifact") from exc
