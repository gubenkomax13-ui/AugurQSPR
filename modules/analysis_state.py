# -*- coding: utf-8 -*-
"""Central analysis configuration, bundle, and dependency reset helpers."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Mapping


ANALYSIS_BUNDLE_VERSION = "analysis_bundle_1"

ALGORITHM_VERSIONS = {
    "chemical_space": "chemical_space_1.3",
    "saod": "saod_3.2",
    "descriptor_matrix": "descriptor_matrix_1.1",
    "validation_quality": "validation_quality_1.1",
    "xtb": "xtb_1.1",
}


@dataclass
class AnalysisConfig:
    random_seed: int = 42
    split_method: str = "random"
    test_size: float = 0.20
    descriptor_mode: str = "mordred_extended"
    correlation_threshold: float = 0.95
    scaling: str = "standard"
    validation_method: str = "holdout"
    fingerprint_radius: int = 2
    fingerprint_bits: int = 2048
    tanimoto_threshold: float = 0.70
    saod_min_points: int = 3
    bootstrap_rmse_p95_ratio_threshold: float = 2.0
    y_randomization_q2_gap_threshold: float = 0.10


RESULT_DEPENDENCY_GRAPH = {
    "dataset": ["standardization", "chemical_space", "saod", "descriptors", "reports"],
    "standardization": ["chemical_space", "saod", "descriptors", "reports"],
    "chemical_space": ["reports"],
    "saod": ["reports"],
    "descriptors": ["feature_selection", "models", "validation", "applicability_domain", "reports"],
    "feature_selection": ["models", "validation", "applicability_domain", "reports"],
    "models": ["validation", "applicability_domain", "reports"],
    "validation": ["reports"],
    "applicability_domain": ["reports"],
    "reports": [],
}


STATE_KEYS_BY_NODE = {
    "dataset": [
        "data_source_note",
        "target_col",
        "descriptor_source_mode",
        "descriptor_source_mode_radio_v2",
        "custom_descriptor_cols_multiselect",
        "uploaded_descriptor_structure_mismatch",
        "uploaded_descriptor_structure_mismatch_message",
        "allow_mismatched_uploaded_descriptors",
        "saod2_cleaning_summary",
        "saod2_show_cleaning_status",
    ],
    "standardization": [
        "standardization_result",
        "standardization_report",
    ],
    "chemical_space": [
        "chemical_diversity_result",
        "chemical_diversity_signature",
        "chemical_space_result",
        "csa_result",
    ],
    "saod": [
        "saod2_result",
        "saod2_review_df",
        "saod2_cleaned_df",
        "saod2_cleaning_applied",
        "saod2_original_before_cleaning",
        "saod2_summary",
        "saod2_tables",
        "saod3_result",
        "saod_result",
    ],
    "descriptors": [
        "desc_calculated",
        "X_all",
        "y_all",
        "valid_indices",
        "row_positions",
        "source_indices",
        "record_ids_current",
        "desc_names",
        "df_desc",
        "descriptor_bundle",
        "descriptor_quality_df",
        "molecular_df_desc",
        "molecular_desc_names",
        "molecular_X_all",
        "molecular_y_all",
        "molecular_valid_indices",
        "spectral_descriptors_df",
        "spectral_descriptors_report",
        "spectral_descriptors_saved_path",
        "spectral_qspr_match_info",
        "xtb_descriptors_df",
        "xtb_descriptors_report",
        "xtb_descriptor_bundle",
        "xtb_descriptor_bank_report",
        "cached_xtb_df",
        "morfeus_descriptors_df",
        "dscribe_descriptors_df",
        "custom_descriptors_used",
        "custom_descriptor_source",
        "custom_descriptor_cols",
    ],
    "feature_selection": [
        "auto_tuning_result",
        "model_used_descriptor_names",
        "model_used_descriptor_model_name",
    ],
    "models": [
        "trained_models",
        "model_comparison_df",
        "model_comparison_errors_df",
        "auto_model_comparison_df",
        "auto_model_comparison_table",
        "true_model_comparison_df",
        "best_model_from_comparison",
        "pending_selected_model",
        "prog_model",
        "prog_scaler",
        "prog_desc_names",
    ],
    "validation": [
        "validation_done",
        "holdout_results_dict",
        "kfold_results_dict",
        "loo_results_dict",
        "repeated_holdout_results_dict",
        "montecarlo_results_dict",
        "bootstrap_results_dict",
        "y_randomization_results_dict",
        "yrandom_best_model_result",
        "ext_validation_result",
        "ext_validation_results_dict",
        "ad_info",
        "model_ad_info",
        "applicability_domain_result",
        "prediction_uncertainty_result",
        "standalone_prediction_uncertainty_result",
        "prediction_uncertainty_results",
        "prediction_uncertainty_table",
        "consensus_result",
        "consensus_df",
        "consensus_prediction_table",
        "consensus_models",
        "consensus_weights",
    ],
    "applicability_domain": [
        "ad_info",
    ],
    "reports": [
        "methodology_history",
        "methodology_current_index",
        "report_full_history",
        "report_full_current_index",
        "report_history",
        "generated_report",
        "last_report_payload",
        "incremental_result",
        "incremental_cols",
    ],
}


def _state_get(state: Mapping[str, Any], key: str, default: Any = None) -> Any:
    try:
        return state.get(key, default)
    except AttributeError:
        return default


def analysis_config_from_session(state: Mapping[str, Any]) -> AnalysisConfig:
    descriptor_mode = (
        _state_get(state, "descriptor_mode_radio_v2")
        or _state_get(state, "molecular_descriptor_calculation_mode")
        or _state_get(state, "descriptor_calculation_mode")
        or "mordred_extended"
    )
    split_method = "random" if _state_get(state, "holdout_random", True) else "manual"
    return AnalysisConfig(
        random_seed=int(_state_get(state, "holdout_rs", _state_get(state, "random_state", 42)) or 42),
        split_method=str(split_method),
        test_size=float(_state_get(state, "holdout_test_size", 20) or 20) / 100.0,
        descriptor_mode=str(descriptor_mode),
        correlation_threshold=float(_state_get(state, "auto_corr_threshold", 0.95) or 0.95),
        scaling=str(_state_get(state, "scaling", "standard") or "standard"),
        validation_method=str(_state_get(state, "validation_method", "holdout") or "holdout"),
        fingerprint_radius=int(
            _state_get(state, "chemical_diversity_morgan_radius", _state_get(state, "chemical_space_fp_radius", 2)) or 2
        ),
        fingerprint_bits=int(
            _state_get(state, "chemical_diversity_morgan_n_bits", _state_get(state, "chemical_space_fp_bits", 2048))
            or 2048
        ),
        tanimoto_threshold=float(
            _state_get(
                state,
                "chemical_diversity_analogue_threshold",
                _state_get(state, "chemical_space_tanimoto_threshold", 0.70),
            )
            or 0.70
        ),
        saod_min_points=int(_state_get(state, "saod2_min_rule_points", 3) or 3),
        bootstrap_rmse_p95_ratio_threshold=float(
            _state_get(state, "bootstrap_rmse_p95_ratio_threshold", 2.0) or 2.0
        ),
        y_randomization_q2_gap_threshold=float(
            _state_get(state, "y_randomization_q2_gap_threshold", 0.10) or 0.10
        ),
    )


def analysis_config_to_dict(config: AnalysisConfig) -> dict[str, Any]:
    return asdict(config)


def current_analysis_parameters_table(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return user-visible parameters that should travel with reports."""
    config = analysis_config_from_session(state)
    rows = [
        {"parameter": "Random seed", "value": config.random_seed, "module": "Validation"},
        {"parameter": "Split method", "value": config.split_method, "module": "Validation"},
        {"parameter": "Test size", "value": config.test_size, "module": "Validation"},
        {"parameter": "Descriptor mode", "value": config.descriptor_mode, "module": "Descriptor selection"},
        {"parameter": "Correlation cutoff", "value": config.correlation_threshold, "module": "Descriptor selection"},
        {"parameter": "Scaling", "value": config.scaling, "module": "QSPR"},
        {"parameter": "Validation method", "value": config.validation_method, "module": "Validation"},
        {"parameter": "Morgan radius", "value": config.fingerprint_radius, "module": "Chemical space"},
        {"parameter": "nBits", "value": config.fingerprint_bits, "module": "Chemical space"},
        {"parameter": "Tanimoto analogue", "value": config.tanimoto_threshold, "module": "Chemical space"},
        {
            "parameter": "Tanimoto near duplicate",
            "value": _state_get(state, "chemical_diversity_duplicate_threshold", 0.95),
            "module": "Chemical space",
        },
        {
            "parameter": "Tanimoto cluster",
            "value": _state_get(state, "chemical_diversity_cluster_threshold", 0.60),
            "module": "Chemical space",
        },
        {
            "parameter": "Projection method",
            "value": _state_get(state, "chemical_diversity_projection_method", "auto"),
            "module": "Chemical space",
        },
        {
            "parameter": "Projection random seed",
            "value": _state_get(state, "chemical_diversity_projection_seed", config.random_seed),
            "module": "Chemical space",
        },
        {"parameter": "SAOD min points", "value": config.saod_min_points, "module": "SAOD"},
        {
            "parameter": "Bootstrap RMSE P95 / median threshold",
            "value": config.bootstrap_rmse_p95_ratio_threshold,
            "module": "Validation quality",
        },
        {
            "parameter": "Y-randomization Q2 gap threshold",
            "value": config.y_randomization_q2_gap_threshold,
            "module": "Validation quality",
        },
    ]
    if _state_get(state, "chemical_space_exact_pattern_small_threshold") is not None:
        rows.extend(
            [
                {"parameter": "Exact pattern singleton threshold", "value": 1, "module": "Chemical space"},
                {
                    "parameter": "Exact pattern small group threshold",
                    "value": _state_get(state, "chemical_space_exact_pattern_small_threshold"),
                    "module": "Chemical space",
                },
                {
                    "parameter": "Exact pattern rare group threshold",
                    "value": _state_get(state, "chemical_space_exact_pattern_rare_threshold"),
                    "module": "Chemical space",
                },
            ]
        )
    return rows


def dataset_signature(df, file_name: str = "", file_size: int | None = None) -> str:
    if df is None:
        payload = {"file_name": file_name, "file_size": file_size, "empty": True}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    payload = {
        "file_name": str(file_name or ""),
        "file_size": int(file_size or 0),
        "shape": [int(getattr(df, "shape", (0, 0))[0]), int(getattr(df, "shape", (0, 0))[1])],
        "columns": [str(col) for col in getattr(df, "columns", [])],
        "dtypes": [str(dtype) for dtype in getattr(df, "dtypes", [])],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8"))
    try:
        import pandas as pd

        content_hash = pd.util.hash_pandas_object(df.reset_index(drop=True), index=True)
        digest.update(content_hash.values.tobytes())
    except Exception:
        digest.update(str(df.head(1000).to_dict()).encode("utf-8", errors="replace"))
    return digest.hexdigest()


def descriptor_matrix_signature(
    X: Any = None,
    y: Any = None,
    desc_names: Any = None,
    valid_indices: Any = None,
) -> str:
    digest = hashlib.sha256()
    payload = {
        "desc_names": [str(name) for name in (desc_names or [])],
        "valid_indices": [int(i) for i in (valid_indices or [])],
    }
    digest.update(json.dumps(payload, sort_keys=True).encode("utf-8", errors="replace"))
    for value in [X, y]:
        if value is None:
            digest.update(b":none")
            continue
        try:
            import numpy as np

            arr = np.asarray(value)
            digest.update(str(arr.shape).encode("utf-8"))
            digest.update(str(arr.dtype).encode("utf-8"))
            digest.update(np.ascontiguousarray(arr).tobytes())
        except Exception:
            digest.update(str(value).encode("utf-8", errors="replace"))
    return digest.hexdigest()


def analysis_result_hash(
    state: Mapping[str, Any],
    model_name: str,
    params: Mapping[str, Any] | None = None,
    validation_settings: Mapping[str, Any] | None = None,
    X: Any = None,
    y: Any = None,
    desc_names: Any = None,
    valid_indices: Any = None,
) -> str:
    payload = {
        "dataset_signature": _state_get(
            state,
            "dataset_signature",
            _state_get(state, "current_dataset_signature", ""),
        ),
        "descriptor_signature": descriptor_matrix_signature(X, y, desc_names, valid_indices),
        "model_name": str(model_name),
        "model_params": params or {},
        "validation_settings": validation_settings or {},
        "analysis_config": analysis_config_to_dict(analysis_config_from_session(state)),
        "algorithm_versions": dict(ALGORITHM_VERSIONS),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8", errors="replace")
    ).hexdigest()


def attach_result_cache_metadata(result: Any, config_hash: str, data_hash: str | None = None) -> Any:
    if isinstance(result, dict):
        result = dict(result)
        result["config_hash"] = config_hash
        result["data_hash"] = data_hash or config_hash
        result["cache_key_version"] = "analysis_result_hash_1"
    return result


def cached_result_is_current(result: Any, config_hash: str) -> bool:
    return isinstance(result, dict) and result.get("config_hash") == config_hash


def downstream_nodes(changed_node: str) -> list[str]:
    seen = set()
    ordered: list[str] = []

    def visit(node: str) -> None:
        for child in RESULT_DEPENDENCY_GRAPH.get(node, []):
            if child in seen:
                continue
            seen.add(child)
            ordered.append(child)
            visit(child)

    visit(str(changed_node))
    return ordered


def reset_analysis_nodes(state, changed_node: str, defaults: Mapping[str, Any] | None = None) -> list[str]:
    nodes = [str(changed_node)] + downstream_nodes(str(changed_node))
    reset_keys: list[str] = []
    for node in nodes:
        for key in STATE_KEYS_BY_NODE.get(node, []):
            if key in reset_keys:
                continue
            reset_keys.append(key)
            if defaults is not None and key in defaults:
                state[key] = deepcopy(defaults[key])
            elif key in state:
                del state[key]

    bundle = ensure_analysis_bundle(state)
    for node in nodes:
        if node in bundle:
            bundle[node] = {} if node != "config" else bundle.get("config", {})
    bundle["invalidated_by"] = str(changed_node)
    bundle["reset_keys"] = reset_keys
    return reset_keys


def ensure_analysis_bundle(state) -> dict[str, Any]:
    bundle = _state_get(state, "analysis_bundle")
    if not isinstance(bundle, dict) or bundle.get("bundle_version") != ANALYSIS_BUNDLE_VERSION:
        bundle = {
            "bundle_version": ANALYSIS_BUNDLE_VERSION,
            "algorithm_versions": dict(ALGORITHM_VERSIONS),
            "dataset": {},
            "descriptors": {},
            "models": {},
            "validation": {},
            "saod": {},
            "chemical_space": {},
            "config": {},
        }
        state["analysis_bundle"] = bundle
    else:
        bundle.setdefault("algorithm_versions", dict(ALGORITHM_VERSIONS))
        for key in ["dataset", "descriptors", "models", "validation", "saod", "chemical_space", "config"]:
            bundle.setdefault(key, {})
    return bundle


def update_analysis_bundle(state, section: str, payload: Mapping[str, Any] | None) -> dict[str, Any]:
    bundle = ensure_analysis_bundle(state)
    bundle[str(section)] = dict(payload or {})
    bundle["algorithm_versions"] = dict(ALGORITHM_VERSIONS)
    state["analysis_bundle"] = bundle
    return bundle


def sync_analysis_bundle_from_session(state) -> dict[str, Any]:
    bundle = ensure_analysis_bundle(state)
    bundle["config"] = analysis_config_to_dict(analysis_config_from_session(state))
    bundle["algorithm_versions"] = dict(ALGORITHM_VERSIONS)
    bundle["descriptors"] = {
        "ready": bool(_state_get(state, "desc_calculated", False)),
        "n_rows": _shape_value(_state_get(state, "X_all"), 0),
        "n_descriptors": len(_state_get(state, "desc_names", []) or []),
        "descriptor_source": _state_get(state, "custom_descriptor_source", ""),
    }
    bundle["models"] = {
        "trained_model_names": sorted(list((_state_get(state, "trained_models", {}) or {}).keys())),
        "best_model_from_comparison": _state_get(state, "best_model_from_comparison"),
    }
    bundle["validation"] = {
        "holdout_models": sorted(list((_state_get(state, "holdout_results_dict", {}) or {}).keys())),
        "bootstrap_models": sorted(list((_state_get(state, "bootstrap_results_dict", {}) or {}).keys())),
        "y_randomization_models": sorted(list((_state_get(state, "y_randomization_results_dict", {}) or {}).keys())),
    }
    if _state_get(state, "saod2_result") is not None:
        bundle["saod"] = {"ready": True, "algorithm_version": ALGORITHM_VERSIONS["saod"]}
    state["analysis_bundle"] = bundle
    return bundle


def _shape_value(value: Any, axis: int) -> int:
    shape = getattr(value, "shape", None)
    if not shape or len(shape) <= axis:
        return 0
    try:
        return int(shape[axis])
    except Exception:
        return 0
