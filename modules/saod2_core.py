# -*- coding: utf-8 -*-

"""
saod2_core.py

Ядро Universal SAOD v3 для Augur QSPR:
Structure-Aware Outlier Detection / Consistency Rule Discovery.

Модуль делает:
- стандартизацию SMILES;
- автоматическое выделение химических рядов;
- поиск общих каркасов и одноточечных R-групп;
- проверку закономерности свойства внутри каждого ряда;
- иерархическое построение сети референтных рядов;
- проверку повторяемых R-групповых трансформаций;
- классификацию ациклических алканов для обратной совместимости;
- оценку проверяемости веществ;
- построение сырых попарных сравнений внутри формул;
- агрегацию сравнений;
- автоматический поиск правил между паттернами;
- поиск поломок правил;
- оценку подозрительности веществ;
- подготовку таблиц "кухни проверки".
"""

import json
import re
import itertools
import hashlib
from collections import defaultdict, deque
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.isotonic import IsotonicRegression


SAOD_VERSION = "3.2"


def _saod_stable_record_id(row, row_position, smiles_col, property_col):
    digest = hashlib.sha1()
    for value in (
        row.get("source_row", row.name),
        row.get(smiles_col, ""),
        row.get(property_col, ""),
    ):
        digest.update(str(value).encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return f"record_{int(row_position) + 1:06d}_{digest.hexdigest()[:12]}"


def _saod_clean_identifier(value, fallback):
    try:
        if pd.isna(value):
            return str(fallback)
    except Exception:
        pass
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null", "<na>"}:
        return str(fallback)
    return text


def _saod_experimental_context_id(row):
    fields = [
        "assay_id",
        "endpoint_type",
        "activity_unit",
        "temperature",
        "phase",
        "laboratory",
        "source",
        "batch",
    ]
    assay = _saod_clean_identifier(row.get("assay_id", ""), "")
    if assay:
        return assay
    digest = hashlib.sha1()
    any_value = False
    for field in fields[1:]:
        value = row.get(field, "")
        text = "" if pd.isna(value) else str(value).strip()
        if text:
            any_value = True
        digest.update(field.encode("utf-8"))
        digest.update(b"=")
        digest.update(text.encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return f"CTX_{digest.hexdigest()[:12]}" if any_value else "CTX_DEFAULT"


def _saod_conflicting_old_columns(df):
    conflicts = []
    for old_col in [col for col in df.columns if str(col).endswith("_old")]:
        new_col = str(old_col)[:-4]
        if new_col not in df.columns:
            continue
        left = df[new_col]
        right = df[old_col]
        comparable = left.notna() & right.notna()
        if not comparable.any():
            continue
        mismatch = left[comparable].astype(str) != right[comparable].astype(str)
        if bool(mismatch.any()):
            conflicts.append({
                "column": new_col,
                "old_column": old_col,
                "n_conflicts": int(mismatch.sum()),
            })
    return conflicts


def _saod_coerce_boolean_series(values, default: bool = False) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    true_values = {True, 1, "1", "true", "True", "TRUE", "yes", "Yes", "YES", "y", "Y"}
    false_values = {False, 0, "0", "false", "False", "FALSE", "no", "No", "NO", "n", "N", ""}

    def coerce_one(value):
        try:
            if pd.isna(value):
                return bool(default)
        except (TypeError, ValueError):
            pass
        if value in true_values:
            return True
        if value in false_values:
            return False
        return bool(default)

    return series.map(coerce_one).astype(bool)


def _saod_coerce_boolean_value(value, default: bool = False) -> bool:
    return bool(_saod_coerce_boolean_series([value], default=default).iloc[0])


@dataclass
class SAODConfig:
    min_series_points: int = 3
    min_own_series_points: int = 3
    min_reference_points: int = 3
    min_transformation_pairs: int = 3
    min_independent_contexts: int = 2
    min_legacy_alkane_rule_points: int = 3

SAOD_ANALYSIS_STATUS_ANALYZED = "analyzed"
SAOD_ANALYSIS_STATUS_EXCLUDED = "excluded"
SAOD_ANALYSIS_STATUS_UNSUPPORTED = "unsupported"
SAOD_ANALYSIS_STATUS_INSUFFICIENT_SUPPORT = "insufficient_support"
SAOD_ANALYSIS_STATUS_FAILED = "failed"


SAOD_STATUS_AGREED = "согласовано"
SAOD_STATUS_UNCHECKABLE = "почти непроверяемое"
SAOD_STATUS_NEEDS_CHECK = "требует проверки"
SAOD_STATUS_STRUCTURAL_MISMATCH = "сильное структурное несоответствие"
SAOD_STATUS_PRIORITY_REVIEW = "приоритетная ручная проверка"

SAOD_STATUS_UNKNOWN_PRIORITY = 1
SAOD_STATUS_PRIORITY = {
    "": -1,
    SAOD_STATUS_AGREED: 0,
    SAOD_STATUS_UNCHECKABLE: 0,
    SAOD_STATUS_NEEDS_CHECK: 1,
    SAOD_STATUS_STRUCTURAL_MISMATCH: 2,
    SAOD_STATUS_PRIORITY_REVIEW: 3,
}
SAOD_STATUS_SOURCE_PRIORITY = {
    "hierarchical_status": 20,
    "final_status": 10,
}

SAOD_CHECKABILITY_GOOD = "GOOD"
SAOD_CHECKABILITY_MODERATE = "MODERATE"
SAOD_CHECKABILITY_LOW = "LOW"
SAOD_CHECKABILITY_UNCHECKABLE = "UNCHECKABLE"


def saod_checkability_code(value=None, score=None):
    text = str(value or "").strip().lower()
    text_map = {
        "хорошо проверяемое": SAOD_CHECKABILITY_GOOD,
        "умеренно проверяемое": SAOD_CHECKABILITY_MODERATE,
        "слабо проверяемое": SAOD_CHECKABILITY_LOW,
        "почти непроверяемое": SAOD_CHECKABILITY_UNCHECKABLE,
        "good": SAOD_CHECKABILITY_GOOD,
        "moderate": SAOD_CHECKABILITY_MODERATE,
        "low": SAOD_CHECKABILITY_LOW,
        "uncheckable": SAOD_CHECKABILITY_UNCHECKABLE,
    }
    if text.upper() in {
        SAOD_CHECKABILITY_GOOD,
        SAOD_CHECKABILITY_MODERATE,
        SAOD_CHECKABILITY_LOW,
        SAOD_CHECKABILITY_UNCHECKABLE,
    }:
        return text.upper()
    if text in text_map:
        return text_map[text]
    try:
        score_value = float(score)
    except Exception:
        return ""
    if score_value >= 7:
        return SAOD_CHECKABILITY_GOOD
    if score_value >= 5:
        return SAOD_CHECKABILITY_MODERATE
    if score_value >= 3:
        return SAOD_CHECKABILITY_LOW
    return SAOD_CHECKABILITY_UNCHECKABLE


def saod_status_code(value):
    text = str(value or "").strip()
    if text == SAOD_STATUS_PRIORITY_REVIEW:
        return "PRIORITY_REVIEW"
    if text == SAOD_STATUS_STRUCTURAL_MISMATCH:
        return "STRUCTURAL_MISMATCH"
    if text == SAOD_STATUS_NEEDS_CHECK:
        return "REVIEW_REQUIRED"
    if text == SAOD_STATUS_UNCHECKABLE:
        return "UNCHECKABLE"
    if text == SAOD_STATUS_AGREED:
        return "CONSISTENT"
    return ""


SAOD_THRESHOLD_PRESETS = {
    "strict": {
        "label": "строгий",
        "edge_strong_broken": 3,
        "edge_strong_score": 5.0,
        "edge_critical_broken": 4,
        "edge_critical_score": 8.0,
        "hierarchical_strong_independent_types": 2,
        "hierarchical_strong_total_breaks": 4,
        "hierarchical_critical_independent_types": 3,
        "hierarchical_critical_reference_breaks": 3,
    },
    "standard": {
        "label": "стандартный",
        "edge_strong_broken": 2,
        "edge_strong_score": 4.0,
        "edge_critical_broken": 3,
        "edge_critical_score": 6.0,
        "hierarchical_strong_independent_types": 2,
        "hierarchical_strong_total_breaks": 3,
        "hierarchical_critical_independent_types": 3,
        "hierarchical_critical_reference_breaks": 2,
    },
    "sensitive": {
        "label": "чувствительный",
        "edge_strong_broken": 1,
        "edge_strong_score": 3.0,
        "edge_critical_broken": 2,
        "edge_critical_score": 5.0,
        "hierarchical_strong_independent_types": 1,
        "hierarchical_strong_total_breaks": 2,
        "hierarchical_critical_independent_types": 2,
        "hierarchical_critical_reference_breaks": 1,
    },
}


def saod2_get_threshold_config(mode="standard", overrides=None):
    mode = str(mode or "standard").strip().lower()
    if mode not in SAOD_THRESHOLD_PRESETS:
        mode = "standard"
    config = dict(SAOD_THRESHOLD_PRESETS[mode])
    config["mode"] = mode
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            if key in config and key not in {"label", "mode"}:
                config[key] = value
    return config


def saod2_threshold_config_table(config):
    config = saod2_get_threshold_config(config.get("mode", "standard"), config)
    rows = [
        ("Режим SAOD", config["label"]),
        ("Статус: сильное структурное несоответствие, минимум сломанных edge-правил", config["edge_strong_broken"]),
        ("Статус: сильное структурное несоответствие, score edge-правила", config["edge_strong_score"]),
        ("Статус: приоритетная ручная проверка, минимум сломанных edge-правил", config["edge_critical_broken"]),
        ("Статус: приоритетная ручная проверка, score edge-правила", config["edge_critical_score"]),
        ("Иерархия: сильное несоответствие, независимых типов доказательств", config["hierarchical_strong_independent_types"]),
        ("Иерархия: сильное несоответствие, всего несогласованностей", config["hierarchical_strong_total_breaks"]),
        ("Иерархия: приоритетная проверка, независимых типов доказательств", config["hierarchical_critical_independent_types"]),
        ("Иерархия: приоритетная проверка, референтных несогласованностей", config["hierarchical_critical_reference_breaks"]),
    ]
    return pd.DataFrame(rows, columns=["Порог", "Значение"])


# ------------------------------------------------------------------
# Numeric / robust utilities

def saod2_to_numeric(series):
    """
    Приводит колонку свойства к числовому виду.
    Поддерживает десятичную запятую.
    """
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )


def saod2_robust_sigma(values, min_sigma=None, measurement_uncertainty=None, range_fraction=0.01):
    """
    Робастная оценка sigma через MAD с fallback.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        base_floor = 1e-6
        if measurement_uncertainty is not None and np.isfinite(float(measurement_uncertainty)):
            base_floor = max(base_floor, float(measurement_uncertainty))
        if min_sigma is not None and np.isfinite(float(min_sigma)):
            base_floor = max(base_floor, float(min_sigma))
        return base_floor

    value_range = float(np.nanmax(values) - np.nanmin(values)) if len(values) else 0.0
    med = np.nanmedian(values)
    value_scale = max(abs(float(med)), 1.0)
    sigma_floor = max(1e-6, float(range_fraction) * 0.1 * value_scale)
    if np.isfinite(value_range) and value_range > 0:
        sigma_floor = max(sigma_floor, float(range_fraction) * value_range)
    if measurement_uncertainty is not None and np.isfinite(float(measurement_uncertainty)):
        sigma_floor = max(sigma_floor, float(measurement_uncertainty))
    if min_sigma is not None and np.isfinite(float(min_sigma)):
        sigma_floor = max(sigma_floor, float(min_sigma))

    mad = np.nanmedian(np.abs(values - med))
    sigma = 1.4826 * mad

    if not np.isfinite(sigma) or sigma <= 1e-12:
        sigma = np.nanstd(values)

    if not np.isfinite(sigma) or sigma <= 1e-12:
        q75, q25 = np.nanpercentile(values, [75, 25])
        iqr = q75 - q25
        sigma = iqr / 1.349 if iqr > 1e-12 else sigma_floor

    return max(float(sigma), float(sigma_floor))


def saod2_level(score):
    """
    Текстовая интерпретация score.
    """
    if pd.isna(score):
        return "не оценено"

    if score < 2:
        return "норма"

    if score < 3:
        return "слабое предупреждение"

    if score < 4:
        return "возможная поломка"

    if score < 6:
        return "сильная поломка"

    return "критическая поломка"


# ------------------------------------------------------------------
# Structure standardization

def saod2_standardize_smiles(smiles):
    """
    Стандартизация структуры:
    - SMILES -> Mol;
    - основной фрагмент;
    - cleanup;
    - uncharge;
    - canonical SMILES;
    - InChIKey.
    """
    status = {
        "parse": "not_run",
        "largest_fragment": "not_run",
        "cleanup": "not_run",
        "uncharging": "not_run",
        "inchikey": "not_run",
        "overall": "failed",
        "code": "",
        "details": {},
    }

    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())

        if mol is None:
            status.update(parse="failed", overall="failed", code="INVALID_SMILES")
            return None, "", "", saod2_encode_standardization_status(status)

        status["parse"] = "ok"

        try:
            if len(Chem.GetMolFrags(mol)) > 1:
                status.update(overall="failed", code="MIXTURE")
                return None, "", "", saod2_encode_standardization_status(status)
        except Exception as exc:
            status["details"]["fragment_check"] = type(exc).__name__

        try:
            chooser = rdMolStandardize.LargestFragmentChooser()
            mol = chooser.choose(mol)
            status["largest_fragment"] = "ok"
        except Exception as exc:
            status["largest_fragment"] = "failed"
            status["details"]["largest_fragment"] = type(exc).__name__

        try:
            mol = rdMolStandardize.Cleanup(mol)
            status["cleanup"] = "ok"
        except Exception as exc:
            status["cleanup"] = "failed"
            status["details"]["cleanup"] = type(exc).__name__

        try:
            uncharger = rdMolStandardize.Uncharger()
            mol = uncharger.uncharge(mol)
            status["uncharging"] = "ok"
        except Exception as exc:
            status["uncharging"] = "failed"
            status["details"]["uncharging"] = type(exc).__name__

        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

        try:
            inchikey = Chem.MolToInchiKey(mol)
            status["inchikey"] = "ok"
        except Exception as exc:
            inchikey = ""
            status["inchikey"] = "failed"
            status["details"]["inchikey"] = type(exc).__name__

        failed_steps = [
            step for step in ("largest_fragment", "cleanup", "uncharging", "inchikey")
            if status.get(step) == "failed"
        ]
        if failed_steps:
            status.update(overall="partial_success", code="STANDARDIZATION_PARTIAL")
        else:
            status.update(overall="ok", code="OK")

        return mol, canonical_smiles, inchikey, saod2_encode_standardization_status(status)

    except Exception as exc:
        status.update(
            parse="failed" if status.get("parse") == "not_run" else status.get("parse"),
            overall="failed",
            code="STANDARDIZATION_FAILED",
        )
        status["details"]["exception_type"] = type(exc).__name__
        return None, "", "", saod2_encode_standardization_status(status)


def saod2_encode_standardization_status(status):
    return json.dumps(status, ensure_ascii=False, sort_keys=True)


def saod2_decode_standardization_status(status):
    if isinstance(status, dict):
        return status
    text = str(status or "")
    if text.startswith("{"):
        try:
            decoded = json.loads(text)
            if isinstance(decoded, dict):
                return decoded
        except Exception:
            pass
    legacy_code = {
        "ok": "OK",
        "invalid_smiles": "INVALID_SMILES",
        "mixture": "MIXTURE",
    }.get(text, text)
    if text.startswith("standardization_error"):
        legacy_code = "STANDARDIZATION_FAILED"
    overall = "ok" if legacy_code == "OK" else "failed"
    return {"overall": overall, "code": legacy_code}


def saod2_structure_status_code(status):
    return str(saod2_decode_standardization_status(status).get("code", ""))


def saod2_structure_status_overall(status):
    return str(saod2_decode_standardization_status(status).get("overall", ""))


def saod2_is_hydrocarbon(mol):
    """
    Проверяет, состоит ли молекула только из C и H.
    """
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num == 1:
            continue
        if atomic_num != 6:
            return False
    return True


def saod2_is_acyclic(mol):
    """
    Проверяет отсутствие циклов.
    """
    return mol.GetRingInfo().NumRings() == 0


def saod2_is_saturated(mol):
    """
    Проверяет, что все связи одинарные.
    """
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            return False

    return True


# ------------------------------------------------------------------
# Alkane graph logic

def saod2_carbon_graph(mol):
    """
    Строит граф C-C связей.
    """
    carbon_ids = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 6
    ]

    graph = {idx: [] for idx in carbon_ids}

    for bond in mol.GetBonds():
        a = bond.GetBeginAtom()
        b = bond.GetEndAtom()

        if a.GetAtomicNum() == 6 and b.GetAtomicNum() == 6:
            graph[a.GetIdx()].append(b.GetIdx())
            graph[b.GetIdx()].append(a.GetIdx())

    return graph


def saod2_longest_path(graph):
    """
    Находит самый длинный путь в дереве C-C.
    Для ациклических алканов C-C граф является деревом.
    """
    if not graph:
        return []

    def bfs(start):
        visited = {start}
        parent = {start: None}
        distance = {start: 0}
        queue = deque([start])
        best_node = start

        while queue:
            node = queue.popleft()

            if distance[node] > distance[best_node]:
                best_node = node

            for nb in graph.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    parent[nb] = node
                    distance[nb] = distance[node] + 1
                    queue.append(nb)

        best_path = []
        current = best_node
        while current is not None:
            best_path.append(current)
            current = parent[current]
        best_path.reverse()
        return best_node, best_path

    start = next(iter(graph))
    node_a, _ = bfs(start)
    node_b, path = bfs(node_a)

    return path


def saod2_collect_branch(graph, start, parent, main_chain_set):
    """
    Собирает атомы заместителя, отходящего от главной цепи.
    """
    nodes = []
    stack = [(start, parent)]

    while stack:
        node, prev = stack.pop()

        if node in main_chain_set:
            continue

        nodes.append(node)

        for nb in graph.get(node, []):
            if nb != prev and nb not in main_chain_set:
                stack.append((nb, node))

    return nodes


def saod2_get_substituents(mol, main_chain):
    """
    Возвращает список заместителей:
    [{'position': 2, 'type': 'methyl', 'size': 1}, ...]
    """
    graph = saod2_carbon_graph(mol)
    main_chain_set = set(main_chain)
    chain_len = len(main_chain)

    substituents = []

    for pos_idx, chain_atom in enumerate(main_chain, start=1):
        for nb in graph.get(chain_atom, []):
            if nb not in main_chain_set:
                branch_nodes = saod2_collect_branch(
                    graph=graph,
                    start=nb,
                    parent=chain_atom,
                    main_chain_set=main_chain_set
                )

                size = len(branch_nodes)

                if size == 1:
                    sub_type = "methyl"
                elif size == 2:
                    sub_type = "ethyl"
                elif size == 3:
                    sub_type = "propyl"
                else:
                    sub_type = f"C{size}_alkyl"

                try:
                    rgroup_smiles = Chem.MolFragmentToSmiles(
                        mol,
                        atomsToUse=sorted(set(branch_nodes + [chain_atom])),
                        rootedAtAtom=int(chain_atom),
                        canonical=True,
                        isomericSmiles=True,
                    )
                except Exception:
                    rgroup_smiles = ""

                substituents.append({
                    "raw_position": int(pos_idx),
                    "type": sub_type,
                    "size": size,
                    "rgroup_smiles": rgroup_smiles,
                    "branch_atom_ids": ";".join(str(x) for x in sorted(branch_nodes)),
                })

    if substituents:
        forward = sorted(
            (int(item["raw_position"]), str(item["type"]), int(item["size"]))
            for item in substituents
        )
        reverse = sorted(
            (int(chain_len + 1 - item["raw_position"]), str(item["type"]), int(item["size"]))
            for item in substituents
        )
        use_reverse = reverse < forward
        for item in substituents:
            raw_position = int(item["raw_position"])
            item["position"] = (
                int(chain_len + 1 - raw_position)
                if use_reverse else raw_position
            )
            item["locant_orientation"] = "reverse" if use_reverse else "forward"
    else:
        use_reverse = False

    substituents = sorted(
        substituents,
        key=lambda x: (x["position"], x["type"], x["size"], x.get("rgroup_smiles", ""))
    )

    return substituents


def saod2_all_longest_paths(graph):
    if not graph:
        return []
    leaves = [node for node, neighbors in graph.items() if len(neighbors) <= 1]
    if len(leaves) <= 1:
        return [[next(iter(graph))]]

    def path_between(start, end):
        stack = [(start, [start])]
        seen = set()
        while stack:
            node, path = stack.pop()
            if node == end:
                return path
            seen.add(node)
            for nb in sorted(graph.get(node, []), reverse=True):
                if nb not in seen and nb not in path:
                    stack.append((nb, path + [nb]))
        return []

    paths = []
    max_len = 0
    for i, start in enumerate(sorted(leaves)):
        for end in sorted(leaves)[i + 1:]:
            path = path_between(start, end)
            if not path:
                continue
            path_len = len(path)
            if path_len > max_len:
                max_len = path_len
                paths = [path]
            elif path_len == max_len:
                paths.append(path)
    normalized = []
    seen = set()
    for path in paths:
        forward = tuple(path)
        backward = tuple(reversed(path))
        key = min(forward, backward)
        if key not in seen:
            seen.add(key)
            normalized.append(list(key))
    return normalized


def saod2_select_main_chain(mol, graph):
    candidates = saod2_all_longest_paths(graph)
    if not candidates:
        return saod2_longest_path(graph), {
            "main_chain_ambiguous": False,
            "main_chain_candidate_count": 0,
            "main_chain_selection_rule": "fallback_double_bfs",
        }

    scored = []
    for chain in candidates:
        substituents = saod2_get_substituents(mol, chain) if mol is not None else []
        locants = tuple(int(item.get("position", 0)) for item in substituents)
        types = tuple(str(item.get("type", "")) for item in substituents)
        chain_key = tuple(int(atom_id) for atom_id in chain)
        score = (
            -len(substituents),
            locants,
            types,
            chain_key,
        )
        scored.append((score, chain, substituents))

    scored.sort(key=lambda item: item[0])
    selected = list(scored[0][1])
    selected_substituents = scored[0][2]
    equivalent_best = [
        item for item in scored
        if item[0][:3] == scored[0][0][:3]
    ]
    return selected, {
        "main_chain_ambiguous": len(candidates) > 1,
        "main_chain_candidate_count": int(len(candidates)),
        "main_chain_equivalent_best_count": int(len(equivalent_best)),
        "main_chain_selection_rule": (
            "longest_path_then_max_substituents_then_lowest_locants_then_substituent_type_then_atom_id"
        ),
        "selected_main_chain_atoms": ";".join(str(x) for x in selected),
        "selected_main_chain_substituent_count": int(len(selected_substituents)),
    }


def saod2_exact_pattern_from_substituents(substituents, is_n_alkane=False):
    """
    Формирует точный паттерн замещения.

    Примеры:
    n-alkane
    2-methyl
    3-methyl
    2-methyl; 3-methyl
    """
    if is_n_alkane:
        return "n-alkane"

    if not substituents:
        return ""

    parts = [
        f"{s['position']}-{s['type']}"
        for s in substituents
    ]

    return "; ".join(parts)


def saod2_classify_alkane(mol):
    """
    Классифицирует ациклический насыщенный углеводород.

    Главное поле:
    exact_pattern.
    """
    result = {
        "is_hydrocarbon": False,
        "is_acyclic": False,
        "is_saturated": False,
        "is_acyclic_alkane": False,
        "alkane_class": "",
        "exact_pattern": "",
        "longest_carbon_chain": np.nan,
        "branch_count": 0,
        "branching_index": np.nan,
        "substituent_summary": "",
        "main_chain_ambiguous": False,
        "main_chain_candidate_count": 0,
        "main_chain_equivalent_best_count": 0,
        "main_chain_selection_rule": "",
        "selected_main_chain_atoms": "",
        "substituent_rgroup_signature": "",
        "substituent_typing_note": "",
    }

    if mol is None:
        return result

    is_hydrocarbon = saod2_is_hydrocarbon(mol)
    is_acyclic = saod2_is_acyclic(mol)
    is_saturated = saod2_is_saturated(mol)

    result["is_hydrocarbon"] = is_hydrocarbon
    result["is_acyclic"] = is_acyclic
    result["is_saturated"] = is_saturated

    if not (is_hydrocarbon and is_acyclic and is_saturated):
        return result

    carbon_count = sum(
        1 for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 6
    )

    graph = saod2_carbon_graph(mol)
    main_chain, chain_selection = saod2_select_main_chain(mol, graph)
    main_chain_len = len(main_chain)

    substituents = saod2_get_substituents(mol, main_chain)
    branch_count = len(substituents)

    if carbon_count > 0:
        branching_index = (carbon_count - main_chain_len) / carbon_count
    else:
        branching_index = np.nan

    is_n_alkane = branch_count == 0

    exact_pattern = saod2_exact_pattern_from_substituents(
        substituents,
        is_n_alkane=is_n_alkane
    )
    rgroup_signature = "; ".join(
        f"{s.get('position')}-{s.get('rgroup_smiles', '')}"
        for s in substituents
        if s.get("rgroup_smiles", "")
    )

    methyl_count = sum(1 for s in substituents if s["type"] == "methyl")
    ethyl_count = sum(1 for s in substituents if s["type"] == "ethyl")

    if is_n_alkane:
        alkane_class = "n-alkane"
    elif methyl_count == 1 and branch_count == 1:
        pos = substituents[0]["position"]
        alkane_class = f"{pos}-methylalkane"
    elif methyl_count == 2 and branch_count == 2:
        alkane_class = "dimethylalkane"
    elif methyl_count == 3 and branch_count == 3:
        alkane_class = "trimethylalkane"
    elif ethyl_count >= 1:
        alkane_class = "ethyl-substituted alkane"
    else:
        alkane_class = "branched alkane"

    result.update({
        "is_acyclic_alkane": True,
        "alkane_class": alkane_class,
        "exact_pattern": exact_pattern,
        "longest_carbon_chain": main_chain_len,
        "branch_count": branch_count,
        "branching_index": branching_index,
        "substituent_summary": exact_pattern if not is_n_alkane else "",
        "substituent_rgroup_signature": rgroup_signature,
        "substituent_typing_note": (
            "Legacy exact_pattern uses branch size labels; "
            "substituent_rgroup_signature stores attachment-aware canonical fragment SMILES."
            if substituents else ""
        ),
        **chain_selection,
    })

    return result


def saod2_check_exact_pattern_smiles_invariance(smiles, n_random=20, random_seed=42):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {
            "input_smiles": str(smiles),
            "status": "invalid_smiles",
            "invariant": False,
            "patterns": [],
            "unique_patterns": [],
        }
    rng = np.random.default_rng(int(random_seed))
    smiles_variants = {Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)}
    for _ in range(int(max(1, n_random))):
        smiles_variants.add(
            Chem.MolToSmiles(
                mol,
                canonical=False,
                doRandom=True,
                isomericSmiles=True,
            )
        )
    rows = []
    for variant in sorted(smiles_variants):
        variant_mol = Chem.MolFromSmiles(variant)
        classification = saod2_classify_alkane(variant_mol)
        rows.append({
            "smiles": variant,
            "exact_pattern": classification.get("exact_pattern", ""),
            "substituent_rgroup_signature": classification.get("substituent_rgroup_signature", ""),
            "selected_main_chain_atoms": classification.get("selected_main_chain_atoms", ""),
            "main_chain_ambiguous": classification.get("main_chain_ambiguous", False),
        })
    unique_patterns = sorted({str(row["exact_pattern"]) for row in rows})
    return {
        "input_smiles": str(smiles),
        "status": "ok",
        "invariant": len(unique_patterns) == 1,
        "patterns": rows,
        "unique_patterns": unique_patterns,
    }


# ------------------------------------------------------------------
# General chemical-series passport

def saod2_normalized_locant(position, chain_length):
    if chain_length <= 0:
        return int(position)
    return int(min(position, chain_length + 1 - position))


def saod2_bond_locant(bond, chain, chain_length):
    index = {atom_id: i + 1 for i, atom_id in enumerate(chain)}
    a = bond.GetBeginAtomIdx()
    b = bond.GetEndAtomIdx()
    if a not in index or b not in index:
        return None
    locant = min(index[a], index[b])
    return min(locant, chain_length - locant)


def saod2_carbon_locant(atom_id, chain):
    try:
        position = chain.index(atom_id) + 1
    except ValueError:
        return None
    return saod2_normalized_locant(position, len(chain))


def saod2_branch_pattern(mol, main_chain):
    substituents = saod2_get_substituents(mol, main_chain)
    if not substituents:
        return "unbranched", 0
    parts = [f"{item['position']}-{item['type']}" for item in substituents]
    return ";".join(parts), len(substituents)


def saod2_functional_group_profile(mol):
    atoms = list(mol.GetAtoms())
    halogens = [
        atom.GetIdx()
        for atom in atoms
        if atom.GetAtomicNum() in [9, 17, 35, 53]
    ]
    carbon_double = [
        bond
        for bond in mol.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetBeginAtom().GetAtomicNum() == 6
        and bond.GetEndAtom().GetAtomicNum() == 6
    ]
    profile = {
        "carboxyl": list(mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[OX2H1]"))),
        "aldehyde": list(mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3H1](=O)[#6]"))),
        "ketone": list(mol.GetSubstructMatches(Chem.MolFromSmarts("[#6][CX3](=O)[#6]"))),
        "alcohol": list(mol.GetSubstructMatches(Chem.MolFromSmarts("[#6;!$(C=O)][OX2H1]"))),
        "amine": list(mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H0,H1,H2;!$(N-C=O)]"))),
        "amide": list(mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3][CX3](=O)"))),
        "carbon_double": carbon_double,
        "halogen": halogens,
    }
    hits = [name for name, matches in profile.items() if len(matches) > 0]
    return profile, hits


def saod2_is_linear_side_chain(graph, side_nodes, attachment=None):
    side_set = set(side_nodes or [])
    if not side_set:
        return False
    for node in side_set:
        side_degree = sum(1 for nb in graph.get(node, []) if nb in side_set)
        if attachment is not None and node == attachment and side_degree > 1:
            return False
        if side_degree > 2:
            return False
    return True


def saod2_detect_functional_class(mol):
    """
    Conservative single-class detector. Multifunctional or unsupported
    compounds are returned as unsupported instead of being forced into a row.
    """
    atoms = list(mol.GetAtoms())
    atomic_numbers = [atom.GetAtomicNum() for atom in atoms]
    ring_count = mol.GetRingInfo().NumRings()
    functional_profile, functional_hits = saod2_functional_group_profile(mol)
    carboxyl = functional_profile["carboxyl"]
    aldehyde = functional_profile["aldehyde"]
    ketone = functional_profile["ketone"]
    alcohol = functional_profile["alcohol"]
    amine = functional_profile["amine"]
    carbon_double = functional_profile["carbon_double"]
    halogens = functional_profile["halogen"]

    aromatic_rings = Chem.GetSymmSSSR(mol)
    benzene_rings = []
    for ring in aromatic_rings:
        ring_ids = list(ring)
        if len(ring_ids) == 6 and all(
            mol.GetAtomWithIdx(idx).GetIsAromatic()
            and mol.GetAtomWithIdx(idx).GetAtomicNum() == 6
            for idx in ring_ids
        ):
            benzene_rings.append(ring_ids)

    features = [name for name in functional_hits if name != "amide"]
    base_details = {
        "functional_group_profile": {
            name: len(matches)
            for name, matches in functional_profile.items()
        },
        "functional_group_hits": list(functional_hits),
        "primary_scaffold_class": "acyclic" if ring_count == 0 else "ring_system",
        "r_group_profile": "",
    }

    if len(carboxyl) == 1 and len(features) == 1 and ring_count == 0:
        return "carboxylic_acid", {**base_details, "match": carboxyl[0]}
    if len(aldehyde) == 1 and len(features) == 1 and ring_count == 0:
        return "aldehyde", {**base_details, "match": aldehyde[0]}
    if len(ketone) == 1 and len(features) == 1 and ring_count == 0:
        return "ketone", {**base_details, "match": ketone[0]}
    if len(alcohol) == 1 and len(features) == 1 and ring_count == 0:
        return "alcohol", {**base_details, "match": alcohol[0]}
    if len(amine) == 1 and len(features) == 1 and ring_count == 0:
        return "amine", {**base_details, "match": amine[0]}
    if len(carbon_double) == 1 and len(features) == 1 and ring_count == 0:
        return "alkene", {**base_details, "bond": carbon_double[0]}
    if len(halogens) == 1 and len(features) == 1 and ring_count == 0:
        return "haloalkane", {**base_details, "atom": halogens[0]}

    if (
        len(benzene_rings) == 1
        and all(number in [1, 6] for number in atomic_numbers)
    ):
        ring_set = set(benzene_rings[0])
        attachments = []
        for ring_atom in ring_set:
            for neighbor in mol.GetAtomWithIdx(ring_atom).GetNeighbors():
                if neighbor.GetIdx() not in ring_set:
                    attachments.append(neighbor.GetIdx())
        if len(set(attachments)) == 1:
            return "alkylbenzene", {
                **base_details,
                "ring": benzene_rings[0],
                "attachment": attachments[0],
                "primary_scaffold_class": "benzene",
                "r_group_profile": "monoalkyl",
            }

    if (
        all(number in [1, 6] for number in atomic_numbers)
        and ring_count == 0
        and not carbon_double
        and all(bond.GetBondType() == Chem.BondType.SINGLE for bond in mol.GetBonds())
    ):
        return "alkane", {**base_details, "primary_scaffold_class": "acyclic_alkane"}

    reason = "unsupported structure"
    if len(functional_hits) > 1:
        reason = "multifunctional structure; route to scaffold/SAR layer"
    elif ring_count > 0:
        reason = "ring system outside simple supported classes"
    return "unsupported", {
        **base_details,
        "series_reason": reason,
    }


def saod2_general_series_passport(mol):
    result = {
        "chemical_class": "unsupported",
        "primary_scaffold_class": "",
        "functional_group_profile": "",
        "r_group_profile": "",
        "series_pattern": "",
        "series_family": "",
        "series_coordinate": np.nan,
        "series_complexity": np.nan,
        "series_supported": False,
        "series_reason": "",
    }
    if mol is None:
        result["series_reason"] = "invalid structure"
        return result

    chemical_class, details = saod2_detect_functional_class(mol)
    result["chemical_class"] = chemical_class
    result["primary_scaffold_class"] = str(details.get("primary_scaffold_class", ""))
    result["functional_group_profile"] = ";".join(
        f"{key}:{value}"
        for key, value in dict(details.get("functional_group_profile", {})).items()
        if int(value or 0) > 0
    )
    result["r_group_profile"] = str(details.get("r_group_profile", ""))

    carbon_count = sum(
        atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()
    )
    result["series_coordinate"] = carbon_count

    if chemical_class == "unsupported":
        result["series_reason"] = (
            details.get("series_reason")
            or "unsupported, cyclic, or multifunctional structure"
        )
        return result

    if chemical_class == "alkylbenzene":
        ring_set = set(details["ring"])
        graph = saod2_carbon_graph(mol)
        attachment = details["attachment"]
        side_nodes = saod2_collect_branch(
            graph, attachment, next(iter(ring_set)), ring_set
        )
        if not side_nodes:
            result["series_reason"] = "empty alkyl substituent"
            return result
        if not saod2_is_linear_side_chain(graph, side_nodes, attachment=attachment):
            result["series_reason"] = "branched alkylbenzene side chain; not mono-n-alkyl"
            result["r_group_profile"] = "branched_alkyl"
            return result
        result.update({
            "series_pattern": "alkylbenzene|mono-n-alkyl",
            "series_family": "alkylbenzene",
            "series_complexity": 0,
            "series_supported": True,
            "series_reason": "monosubstituted alkylbenzene",
            "r_group_profile": f"linear_alkyl_C{len(side_nodes)}",
        })
        return result

    graph = saod2_carbon_graph(mol)
    main_chain, chain_selection = saod2_select_main_chain(mol, graph)
    chain_length = len(main_chain)
    branch_pattern, branch_count = saod2_branch_pattern(mol, main_chain)

    if chemical_class == "alkane":
        alkane = saod2_classify_alkane(mol)
        pattern = alkane.get("exact_pattern", "")
        result.update({
            "series_pattern": f"alkane|{pattern}",
            "series_family": "alkane",
            "series_complexity": int(alkane.get("branch_count", 0)),
            "series_supported": bool(pattern),
            "series_reason": "acyclic alkane pattern",
            "main_chain_ambiguous": alkane.get("main_chain_ambiguous", False),
            "main_chain_candidate_count": alkane.get("main_chain_candidate_count", 0),
            "main_chain_selection_rule": alkane.get("main_chain_selection_rule", ""),
        })
        return result

    functional_label = ""
    functional_locant = 0

    if chemical_class == "alkene":
        locant = saod2_bond_locant(details["bond"], main_chain, chain_length)
        if locant is None:
            result["series_reason"] = "double bond outside selected main chain"
            return result
        functional_locant = locant
        functional_label = f"ene-{locant}"

    elif chemical_class in ["alcohol", "amine", "haloalkane"]:
        match = details.get("match")
        if chemical_class == "haloalkane":
            hetero = details["atom"]
            carbon_neighbors = [
                atom for atom in hetero.GetNeighbors()
                if atom.GetAtomicNum() == 6
            ]
            if len(carbon_neighbors) != 1:
                result["series_reason"] = "ambiguous haloalkane attachment"
                return result
            carbon_id = carbon_neighbors[0].GetIdx()
            symbol = hetero.GetSymbol()
            functional_label = f"{symbol}-{{locant}}"
        else:
            carbon_id = int(match[0])
            functional_label = (
                "OH-{locant}" if chemical_class == "alcohol"
                else "amine-{locant}"
            )

        locant = saod2_carbon_locant(carbon_id, main_chain)
        if locant is None:
            result["series_reason"] = "functional carbon outside main chain"
            return result
        functional_locant = locant
        functional_label = functional_label.format(locant=locant)

        if chemical_class == "amine":
            nitrogen = mol.GetAtomWithIdx(int(match[0]))
            carbon_degree = sum(
                neighbor.GetAtomicNum() == 6
                for neighbor in nitrogen.GetNeighbors()
            )
            functional_label += f"|degree-{carbon_degree}"

    elif chemical_class in ["aldehyde", "ketone", "carboxylic_acid"]:
        carbonyl_id = int(details["match"][0])
        locant = saod2_carbon_locant(carbonyl_id, main_chain)
        if locant is None:
            result["series_reason"] = "carbonyl carbon outside main chain"
            return result
        functional_locant = locant
        functional_label = f"carbonyl-{locant}"

    pattern = (
        f"{chemical_class}|{functional_label}|branches:{branch_pattern}"
    )
    result.update({
        "series_pattern": pattern,
        "series_family": chemical_class,
        "series_complexity": int(branch_count + functional_locant),
        "series_supported": True,
        "series_reason": "supported homologous-series profile",
        "main_chain_ambiguous": chain_selection.get("main_chain_ambiguous", False),
        "main_chain_candidate_count": chain_selection.get("main_chain_candidate_count", 0),
        "main_chain_selection_rule": chain_selection.get("main_chain_selection_rule", ""),
    })
    return result


# ------------------------------------------------------------------
# Prepare structures

def saod2_prepare_structures(
    input_df,
    smiles_col,
    property_col,
    duplicate_key_policy="canonical_smiles",
    duplicate_abs_tolerance=1e-6,
    duplicate_rel_tolerance=1e-6,
    duplicate_measurement_uncertainty=None,
):
    """
    Главная подготовка структур для SAOD v2.
    """
    df = input_df.copy()
    smiles_col = str(smiles_col).strip()
    property_col = str(property_col).strip()
    df.columns = pd.Index([str(col).strip() for col in df.columns])
    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicate_columns:
        duplicates_text = ", ".join(sorted(set(map(str, duplicate_columns))))
        raise ValueError(
            "Duplicate column names after whitespace normalization: "
            f"{duplicates_text}"
        )

    if smiles_col not in df.columns:
        raise ValueError(f"Не найдена колонка SMILES: {smiles_col}")

    if property_col not in df.columns:
        raise ValueError(f"Не найдена колонка свойства: {property_col}")

    if "source_row" not in df.columns:
        df["source_row"] = df.index

    if "record_id" not in df.columns:
        df["record_id"] = [
            _saod_stable_record_id(row, i, smiles_col, property_col)
            for i, (_, row) in enumerate(df.iterrows())
        ]
    else:
        df["record_id"] = [
            _saod_clean_identifier(
                row.get("record_id", ""),
                _saod_stable_record_id(row, i, smiles_col, property_col),
            )
            for i, (_, row) in enumerate(df.iterrows())
        ]
        duplicated_record_ids = df["record_id"].astype(str).duplicated(keep=False)
        if duplicated_record_ids.any():
            df.loc[duplicated_record_ids, "record_id"] = [
                _saod_stable_record_id(row, i, smiles_col, property_col)
                for i, (_, row) in enumerate(
                    df.loc[duplicated_record_ids].iterrows()
                )
            ]

    if "compound_id" not in df.columns:
        df["compound_id"] = df["record_id"].astype(str)
    else:
        df["compound_id"] = [
            _saod_clean_identifier(value, fallback)
            for value, fallback in zip(df["compound_id"], df["record_id"])
        ]

    if "measurement_id" not in df.columns:
        df["measurement_id"] = df["record_id"].astype(str)

    if "assay_id" not in df.columns:
        df["assay_id"] = ""
    df["experimental_context_id"] = [
        _saod_experimental_context_id(row)
        for _, row in df.iterrows()
    ]

    if "name" not in df.columns:
        df["name"] = ""

    df["property_value"] = saod2_to_numeric(df[property_col])

    rows = []

    for idx, row in df.iterrows():
        smiles = row.get(smiles_col, "")

        mol, canonical_smiles, inchikey, structure_status = saod2_standardize_smiles(
            smiles
        )
        structure_status_code = saod2_structure_status_code(structure_status)
        structure_standardization_overall = saod2_structure_status_overall(structure_status)

        base = {
            "row_index": idx,
            "record_id": row.get("record_id", ""),
            "compound_id": row.get("compound_id", ""),
            "measurement_id": row.get("measurement_id", ""),
            "assay_id": row.get("assay_id", ""),
            "experimental_context_id": row.get("experimental_context_id", "CTX_DEFAULT"),
            "source_row": row.get("source_row", idx),
            "input_smiles": smiles,
            "canonical_smiles": canonical_smiles,
            "inchikey": inchikey,
            "structure_status": structure_status,
            "structure_status_code": structure_status_code,
            "structure_standardization_overall": structure_standardization_overall,
            "valid_structure": mol is not None
        }

        if mol is None:
            base.update({
                "molecular_formula": "",
                "molecular_weight": np.nan,
                "atom_count": np.nan,
                "carbon_count": np.nan,
                "ring_count": np.nan
            })

            base.update(saod2_classify_alkane(None))
            base.update(saod2_general_series_passport(None))
            rows.append(base)
            continue

        try:
            formula = rdMolDescriptors.CalcMolFormula(mol)
        except Exception:
            formula = ""

        try:
            mw = Descriptors.MolWt(mol)
        except Exception:
            mw = np.nan

        carbon_count = sum(
            1 for atom in mol.GetAtoms()
            if atom.GetAtomicNum() == 6
        )

        base.update({
            "molecular_formula": formula,
            "molecular_weight": mw,
            "atom_count": mol.GetNumAtoms(),
            "carbon_count": carbon_count,
            "ring_count": mol.GetRingInfo().NumRings()
        })

        base.update(saod2_classify_alkane(mol))
        base.update(saod2_general_series_passport(mol))
        rows.append(base)

    desc = pd.DataFrame(rows)

    out = pd.concat(
        [
            df.reset_index(drop=True),
            desc.drop(
                columns=[
                    "row_index",
                    "record_id",
                    "compound_id",
                    "measurement_id",
                    "assay_id",
                    "experimental_context_id",
                    "source_row",
                ],
                errors="ignore",
            )
        ],
        axis=1
    )

    out["duplicate_structure"] = False
    out["duplicate_conflict"] = False
    out["duplicate_key_policy"] = duplicate_key_policy
    out["duplicate_key"] = ""
    out["duplicate_value_range"] = np.nan
    out["duplicate_tolerance_used"] = np.nan

    key_column = "canonical_smiles"
    policy = str(duplicate_key_policy or "canonical_smiles").strip().lower()
    if policy in {"inchikey", "stereo_inchikey"} and "inchikey" in out.columns:
        key_column = "inchikey"
    elif policy in {"connectivity_inchikey", "stereo_independent_inchikey"} and "inchikey" in out.columns:
        out["_duplicate_connectivity_key"] = out["inchikey"].astype(str).str.split("-").str[0]
        key_column = "_duplicate_connectivity_key"

    valid = out[key_column].astype(str).str.len() > 0

    if valid.any():
        counts = (
            out.loc[valid]
            .groupby(key_column, dropna=False)[key_column]
            .transform("count")
        )

        out.loc[valid, "duplicate_structure"] = counts > 1
        out.loc[valid, "duplicate_key"] = out.loc[valid, key_column].astype(str)

        for _, group in out.loc[valid].groupby(key_column, dropna=False):
            vals = group["property_value"].dropna().astype(float).values

            if len(vals) > 1:
                value_range = float(np.nanmax(vals) - np.nanmin(vals))
                value_scale = max(abs(float(np.nanmedian(vals))), 1.0)
                tolerance = max(
                    float(duplicate_abs_tolerance or 0.0),
                    float(duplicate_rel_tolerance or 0.0) * value_scale,
                )
                if (
                    duplicate_measurement_uncertainty is not None
                    and np.isfinite(float(duplicate_measurement_uncertainty))
                ):
                    tolerance = max(tolerance, float(duplicate_measurement_uncertainty))
                out.loc[group.index, "duplicate_value_range"] = value_range
                out.loc[group.index, "duplicate_tolerance_used"] = tolerance
                if value_range > tolerance:
                    out.loc[group.index, "duplicate_conflict"] = True

    out = out.drop(columns=["_duplicate_connectivity_key"], errors="ignore")

    return out


def saod2_measurement_uncertainty_summary(
    processed,
    uncertainty_col=None,
    typical_uncertainty=None,
):
    values = []
    if (
        uncertainty_col
        and isinstance(processed, pd.DataFrame)
        and uncertainty_col in processed.columns
    ):
        values.extend(
            pd.to_numeric(processed[uncertainty_col], errors="coerce")
            .dropna()
            .astype(float)
            .tolist()
        )
    if typical_uncertainty is not None:
        try:
            value = float(typical_uncertainty)
            if np.isfinite(value) and value >= 0:
                values.append(value)
        except Exception:
            pass
    values = [float(v) for v in values if np.isfinite(float(v)) and float(v) >= 0]
    if not values:
        return None
    return float(np.nanmedian(values))


def saod2_replicate_measurement_report(
    processed,
    structure_col="inchikey",
    property_value_col="property_value",
    uncertainty_col=None,
):
    if not isinstance(processed, pd.DataFrame) or processed.empty:
        return pd.DataFrame()
    if structure_col not in processed.columns:
        structure_col = "canonical_smiles"
    if structure_col not in processed.columns or property_value_col not in processed.columns:
        return pd.DataFrame()

    work = processed.copy()
    work["_replicate_key"] = work[structure_col].astype(str)
    work = work[work["_replicate_key"].str.len() > 0].copy()
    work[property_value_col] = pd.to_numeric(
        work[property_value_col],
        errors="coerce",
    )
    work = work[work[property_value_col].notna()].copy()
    if work.empty:
        return pd.DataFrame()

    rows = []
    for key, group in work.groupby("_replicate_key", dropna=False):
        values = group[property_value_col].astype(float)
        if len(values) < 2:
            continue
        uncertainty = np.nan
        if uncertainty_col and uncertainty_col in group.columns:
            u_values = pd.to_numeric(group[uncertainty_col], errors="coerce")
            if u_values.notna().any():
                uncertainty = float(np.nanmedian(u_values))
        rows.append({
            "structure_key": key,
            "structure_key_type": structure_col,
            "n_measurements": int(len(values)),
            "mean_property_value": float(np.nanmean(values)),
            "median_property_value": float(np.nanmedian(values)),
            "std_property_value": (
                float(np.nanstd(values, ddof=1)) if len(values) > 1 else np.nan
            ),
            "range_property_value": float(np.nanmax(values) - np.nanmin(values)),
            "median_measurement_uncertainty": uncertainty,
            "compound_ids": "; ".join(sorted(set(group["compound_id"].astype(str)))),
            "record_ids": "; ".join(sorted(set(group["record_id"].astype(str)))),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Pattern utilities

def saod2_pattern_family(pattern):
    """
    Семейство паттерна без позиций.

    2-methyl -> methyl
    3-methyl -> methyl
    2-methyl; 3-methyl -> methyl+methyl
    """
    pattern = str(pattern).strip()

    if pattern == "n-alkane":
        return "n-alkane"

    if pattern == "":
        return ""

    parts = [
        p.strip()
        for p in pattern.split(";")
        if p.strip()
    ]

    types = []

    for part in parts:
        m = re.match(r"^(\d+)\-(.+)$", part)

        if m:
            types.append(m.group(2).strip())
        else:
            types.append(part)

    return "+".join(sorted(types))


def saod2_pattern_direction_rank(pattern):
    pattern = str(pattern or "").strip()
    if pattern == "n-alkane":
        return (0, 0, pattern)
    parts = [p.strip() for p in pattern.split(";") if p.strip()]
    substituent_count = len(parts)
    family = saod2_pattern_family(pattern)
    locants = []
    for part in parts:
        match = re.match(r"^(\d+)\-", part)
        if match:
            locants.append(int(match.group(1)))
    first_locant = min(locants) if locants else 99
    return (1, substituent_count, first_locant, family, pattern)


def saod2_order_patterns_for_direction(record_a, record_b):
    pattern_a = str(record_a.get("exact_pattern", ""))
    pattern_b = str(record_b.get("exact_pattern", ""))
    canonical_pair_id = " || ".join(sorted([pattern_a, pattern_b]))
    rank_a = saod2_pattern_direction_rank(pattern_a)
    rank_b = saod2_pattern_direction_rank(pattern_b)
    if rank_a <= rank_b:
        reference, derived = record_a, record_b
    else:
        reference, derived = record_b, record_a
    return reference, derived, canonical_pair_id


# ------------------------------------------------------------------
# Checkability

def saod2_checkability_table(df):
    """
    Оценивает проверяемость каждого вещества в SAOD v2.

    Проверяемость разделена на уровни:
    - собственный ряд exact_pattern;
    - изомеры той же molecular_formula;
    - сеть правил между паттернами;
    - итоговая оценка.
    """
    work = df.copy()

    alk = work[
        (work["valid_structure"] == True) &
        (work["is_acyclic_alkane"] == True) &
        (work["property_value"].notna()) &
        (work["exact_pattern"].astype(str).str.len() > 0)
    ].copy()

    if alk.empty:
        return pd.DataFrame()

    alk["series_size"] = (
        alk
        .groupby("exact_pattern", dropna=False)["exact_pattern"]
        .transform("count")
    )

    alk["formula_group_size"] = (
        alk
        .groupby("molecular_formula", dropna=False)["molecular_formula"]
        .transform("count")
    )

    has_same_pattern_neighbors = []

    for _, row in alk.iterrows():
        pattern = row["exact_pattern"]
        c = row["carbon_count"]

        neighbors = alk[
            (alk["exact_pattern"] == pattern) &
            (alk["carbon_count"].isin([c - 1, c + 1]))
        ]

        has_same_pattern_neighbors.append(len(neighbors) > 0)

    alk["has_same_pattern_neighbors"] = has_same_pattern_neighbors
    alk["has_formula_isomers"] = alk["formula_group_size"] >= 2

    has_positional_analogs = []

    for _, row in alk.iterrows():
        formula = row["molecular_formula"]
        pattern = row["exact_pattern"]
        family = saod2_pattern_family(pattern)

        analogs = alk[
            (alk["molecular_formula"] == formula) &
            (alk["exact_pattern"] != pattern) &
            (alk["exact_pattern"].apply(saod2_pattern_family) == family)
        ]

        has_positional_analogs.append(len(analogs) > 0)

    alk["has_positional_analogs"] = has_positional_analogs

    alk["raw_edges_total"] = 0
    alk["trusted_edges_total"] = 0
    alk["broken_trusted_edges"] = 0

    def get_series_checkability(series_size, has_neighbors):
        if series_size >= 4:
            return "есть хороший собственный ряд"

        if series_size >= 2 and has_neighbors:
            return "есть короткий собственный ряд с соседями"

        if series_size >= 2:
            return "есть короткий собственный ряд"

        return "уникальное по собственному паттерну"

    alk["series_checkability"] = alk.apply(
        lambda row: get_series_checkability(
            row["series_size"],
            row["has_same_pattern_neighbors"]
        ),
        axis=1
    )

    def get_formula_checkability(formula_group_size):
        if formula_group_size >= 8:
            return "хорошо проверяемое по изомерам той же формулы"

        if formula_group_size >= 4:
            return "проверяемое по изомерам той же формулы"

        if formula_group_size >= 2:
            return "слабо проверяемое по формуле"

        return "нет изомеров той же формулы"

    alk["formula_checkability"] = alk["formula_group_size"].apply(
        get_formula_checkability
    )

    def get_network_checkability(trusted_edges_total):
        if trusted_edges_total >= 4:
            return "хорошо проверяемое через сеть паттернов"

        if trusted_edges_total >= 2:
            return "проверяемое через сеть паттернов"

        if trusted_edges_total >= 1:
            return "слабо проверяемое через сеть паттернов"

        return "пока нет доверительных правил вокруг вещества"

    alk["network_checkability"] = alk["trusted_edges_total"].apply(
        get_network_checkability
    )

    def get_overall_checkability(row):
        series_size = row["series_size"]
        formula_group_size = row["formula_group_size"]
        trusted_edges_total = row["trusted_edges_total"]

        if series_size >= 4 and formula_group_size >= 4:
            return "хорошо проверяемое"

        if series_size >= 4:
            return "хорошо проверяемое по собственному ряду"

        if series_size == 1 and formula_group_size >= 4:
            return "уникальное по паттерну, но проверяемое по формуле"

        if series_size == 1 and formula_group_size >= 2:
            return "уникальное по паттерну, частично проверяемое по формуле"

        if series_size >= 2 and formula_group_size >= 2:
            return "умеренно проверяемое"

        if trusted_edges_total >= 2:
            return "проверяемое через сеть паттернов"

        if formula_group_size == 1 and series_size == 1:
            return "почти непроверяемое"

        return "слабо проверяемое"

    alk["overall_checkability"] = alk.apply(
        get_overall_checkability,
        axis=1
    )

    def get_dataset_noise_risk(overall_checkability):
        if overall_checkability in [
            "хорошо проверяемое",
            "хорошо проверяемое по собственному ряду"
        ]:
            return "низкий"

        if overall_checkability in [
            "умеренно проверяемое",
            "уникальное по паттерну, но проверяемое по формуле"
        ]:
            return "умеренный"

        if overall_checkability in [
            "уникальное по паттерну, частично проверяемое по формуле",
            "слабо проверяемое",
            "проверяемое через сеть паттернов"
        ]:
            return "повышенный"

        return "высокий"

    alk["dataset_noise_risk"] = alk["overall_checkability"].apply(
        get_dataset_noise_risk
    )

    def make_checkability_comment(row):
        comments = []

        comments.append(
            f"Размер собственного ряда exact_pattern = {int(row['series_size'])}."
        )

        comments.append(
            f"Размер группы изомеров с той же формулой = {int(row['formula_group_size'])}."
        )

        comments.append(
            f"Проверяемость по ряду: {row['series_checkability']}."
        )

        comments.append(
            f"Проверяемость по формуле: {row['formula_checkability']}."
        )

        comments.append(
            f"Проверяемость через сеть: {row['network_checkability']}."
        )

        if row["series_size"] == 1 and row["formula_group_size"] >= 4:
            comments.append(
                "Вещество уникально по собственному структурному паттерну, "
                "но его можно проверять внутри группы изомеров той же молекулярной формулы."
            )

        if row["series_size"] == 1 and row["formula_group_size"] == 1:
            comments.append(
                "Вещество практически не имеет внутренних структурных аналогов в датасете."
            )

        if _saod_coerce_boolean_value(row.get("duplicate_conflict", False)):
            comments.append(
                "Также обнаружен конфликт дубликатов: одна структура имеет разные значения свойства."
            )

        return " ".join(comments)

    alk["checkability_comment"] = alk.apply(
        make_checkability_comment,
        axis=1
    )

    scores = []

    for _, row in alk.iterrows():
        score = 0

        if row["series_size"] >= 4:
            score += 2
        elif row["series_size"] >= 2:
            score += 1

        if row["formula_group_size"] >= 8:
            score += 3
        elif row["formula_group_size"] >= 4:
            score += 2
        elif row["formula_group_size"] >= 2:
            score += 1

        if row["has_same_pattern_neighbors"]:
            score += 1

        if row["has_positional_analogs"]:
            score += 2

        if row["trusted_edges_total"] >= 2:
            score += 2
        elif row["trusted_edges_total"] >= 1:
            score += 1

        scores.append(score)

    alk["checkability_score"] = scores

    def old_style_level(score):
        if score >= 7:
            return "хорошо проверяемое"

        if score >= 5:
            return "умеренно проверяемое"

        if score >= 3:
            return "слабо проверяемое"

        return "почти непроверяемое"

    alk["checkability_level"] = alk["checkability_score"].apply(
        old_style_level
    )
    alk["checkability_code"] = alk.apply(
        lambda row: saod_checkability_code(
            row.get("checkability_level", ""),
            row.get("checkability_score", np.nan),
        ),
        axis=1,
    )

    alk["checkability_recommendation"] = alk["overall_checkability"].map({
        "хорошо проверяемое": "Вещество хорошо поддержано и собственным рядом, и изомерами той же формулы.",
        "хорошо проверяемое по собственному ряду": "Вещество хорошо поддержано собственным структурным рядом.",
        "умеренно проверяемое": "Вещество имеет несколько внутренних проверок.",
        "уникальное по паттерну, но проверяемое по формуле": "Не считать непроверяемым: собственного ряда нет, но есть изомерная группа той же формулы.",
        "уникальное по паттерну, частично проверяемое по формуле": "Собственного ряда нет, но есть ограниченная проверка внутри той же формулы.",
        "проверяемое через сеть паттернов": "Вещество поддержано найденными связями между структурными паттернами.",
        "слабо проверяемое": "Вещество проверяется ограниченно; использовать осторожно.",
        "почти непроверяемое": "В датасете почти нет структурных аналогов для внутренней проверки."
    }).fillna("Проверяемость ограничена; требуется ручная интерпретация.")

    cols = [
        "compound_id",
        "name",
        "canonical_smiles",
        "molecular_formula",
        "carbon_count",
        "exact_pattern",
        "property_value",

        "series_size",
        "formula_group_size",
        "raw_edges_total",
        "trusted_edges_total",
        "broken_trusted_edges",

        "has_same_pattern_neighbors",
        "has_formula_isomers",
        "has_positional_analogs",

        "series_checkability",
        "formula_checkability",
        "network_checkability",
        "overall_checkability",

        "checkability_score",
        "checkability_level",
        "checkability_code",
        "dataset_noise_risk",

        "checkability_comment",
        "checkability_recommendation"
    ]

    cols = [c for c in cols if c in alk.columns]

    return alk[cols].reset_index(drop=True)
    
# ------------------------------------------------------------------
# Edge table: сырые сравнения внутри одинаковой формулы

def saod2_build_edge_table(df):
    """
    Строит все попарные связи между паттернами внутри одинаковой формулы.

    Для каждой формулы CnH2n+2:
        pattern A vs pattern B
        delta = property(A) - property(B)

    Это сырая таблица: если внутри одной формулы есть несколько веществ
    одного паттерна, будут все пары.
    """
    alk = df[
        (df["valid_structure"] == True) &
        (df["is_acyclic_alkane"] == True) &
        (df["property_value"].notna()) &
        (df["exact_pattern"].astype(str).str.len() > 0) &
        (df["molecular_formula"].astype(str).str.len() > 0)
    ].copy()

    if alk.empty:
        return pd.DataFrame()

    rows = []

    for formula, g in alk.groupby("molecular_formula", dropna=False):
        if len(g) < 2:
            continue

        records = g.to_dict("records")

        for a, b in itertools.combinations(records, 2):
            pattern_a = a["exact_pattern"]
            pattern_b = b["exact_pattern"]

            if pattern_a == pattern_b:
                continue

            reference, derived, canonical_pair_id = saod2_order_patterns_for_direction(a, b)
            pattern_a = reference["exact_pattern"]
            pattern_b = derived["exact_pattern"]

            # Фиксируем порядок паттернов, чтобы edge_label был одинаковым
            # по всем формулам.
            if False and pattern_a > pattern_b:
                a, b = b, a
                pattern_a = a["exact_pattern"]
                pattern_b = b["exact_pattern"]

            delta = derived["property_value"] - reference["property_value"]

            try:
                c_count = int(reference["carbon_count"])
            except Exception:
                c_count = np.nan

            edge_label = f"{pattern_a} -> {pattern_b}"

            rows.append({
                "formula": formula,
                "carbon_count": c_count,
                "edge_label": edge_label,
                "canonical_pair_id": canonical_pair_id,
                "chemical_direction": edge_label,
                "reference_pattern": pattern_a,
                "derived_pattern": pattern_b,

                "pattern_a": pattern_a,
                "pattern_b": pattern_b,

                "family_a": saod2_pattern_family(pattern_a),
                "family_b": saod2_pattern_family(pattern_b),

                "compound_a_id": reference.get("compound_id", ""),
                "compound_b_id": derived.get("compound_id", ""),

                "name_a": reference.get("name", ""),
                "name_b": derived.get("name", ""),

                "smiles_a": reference.get("canonical_smiles", ""),
                "smiles_b": derived.get("canonical_smiles", ""),

                "value_a": reference.get("property_value", np.nan),
                "value_b": derived.get("property_value", np.nan),

                "delta_a_minus_b": delta,
                "delta_derived_minus_reference": delta,

                "is_n_alkane_edge": (
                    pattern_a == "n-alkane" or pattern_b == "n-alkane"
                ),
                "same_substituent_family": (
                    saod2_pattern_family(pattern_a) ==
                    saod2_pattern_family(pattern_b)
                )
            })

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out = out.sort_values(
        ["edge_label", "carbon_count", "formula"]
    ).reset_index(drop=True)

    return out


def saod2_aggregate_edge_table_by_formula(raw_edge_table):
    """
    Делает одну строку на:
        edge_label + molecular_formula

    Это нужно, чтобы график правила строился корректно:
    одна формула = одна точка.

    Если внутри одной формулы есть несколько сравнений A-B,
    берём медианную Δ, а разброс сохраняем в отдельных колонках.
    """
    if raw_edge_table.empty:
        return pd.DataFrame()

    rows = []

    group_cols = [
        "edge_label",
        "formula",
        "carbon_count",
        "canonical_pair_id",
        "chemical_direction",
        "reference_pattern",
        "derived_pattern",
        "pattern_a",
        "pattern_b",
        "family_a",
        "family_b",
        "is_n_alkane_edge",
        "same_substituent_family"
    ]

    for keys, group in raw_edge_table.groupby(group_cols, dropna=False):
        key_dict = dict(zip(group_cols, keys))

        deltas = pd.to_numeric(
            group["delta_a_minus_b"],
            errors="coerce"
        ).dropna()

        if deltas.empty:
            continue

        n_raw = len(deltas)
        median_delta = float(np.nanmedian(deltas))
        mean_delta = float(np.nanmean(deltas))
        min_delta = float(np.nanmin(deltas))
        max_delta = float(np.nanmax(deltas))
        delta_spread = max_delta - min_delta
        mad_delta = float(np.nanmedian(np.abs(deltas - median_delta)))
        positive_fraction = float(np.mean(deltas > 0))
        negative_fraction = float(np.mean(deltas < 0))
        direction_conflict = bool(positive_fraction >= 0.25 and negative_fraction >= 0.25)

        names_a = sorted(set(group["name_a"].astype(str).tolist()))
        names_b = sorted(set(group["name_b"].astype(str).tolist()))

        ids_a = sorted(set(group["compound_a_id"].astype(str).tolist()))
        ids_b = sorted(set(group["compound_b_id"].astype(str).tolist()))

        values_a = sorted(set(group["value_a"].astype(str).tolist()))
        values_b = sorted(set(group["value_b"].astype(str).tolist()))

        ambiguity_note = ""

        if n_raw > 1:
            ambiguity_note = (
                f"Внутри формулы найдено {n_raw} сырых сравнений для этой пары паттернов. "
                f"Для тренда использована медианная Δ. "
                f"Разброс Δ внутри формулы: {delta_spread:.3f}."
            )

        row = {}
        row.update(key_dict)

        row.update({
            "compound_a_id": "; ".join(ids_a),
            "compound_b_id": "; ".join(ids_b),

            "name_a": "; ".join(names_a),
            "name_b": "; ".join(names_b),

            "smiles_a": "; ".join(sorted(set(group["smiles_a"].astype(str).tolist()))),
            "smiles_b": "; ".join(sorted(set(group["smiles_b"].astype(str).tolist()))),

            "value_a": "; ".join(values_a),
            "value_b": "; ".join(values_b),

            "delta_a_minus_b": median_delta,
            "delta_derived_minus_reference": median_delta,

            "mean_delta_within_formula": mean_delta,
            "min_delta_within_formula": min_delta,
            "max_delta_within_formula": max_delta,
            "delta_spread_within_formula": delta_spread,
            "delta_mad_within_formula": mad_delta,
            "positive_delta_fraction": positive_fraction,
            "negative_delta_fraction": negative_fraction,
            "direction_conflict": direction_conflict,
            "n_raw_comparisons": n_raw,
            "ambiguity_note": ambiguity_note
        })

        rows.append(row)

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out = out.sort_values(
        ["edge_label", "carbon_count", "formula"]
    ).reset_index(drop=True)

    return out


# ------------------------------------------------------------------
# Rule discovery

def saod2_leave_one_out_trend_predictions(x, y, model_kind="linear"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pred = np.repeat(np.nanmedian(y), len(y))
    if len(y) < 3:
        return pred

    for index in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[index] = False
        x_train = x[mask]
        y_train = y[mask]
        finite = np.isfinite(x_train) & np.isfinite(y_train)
        x_train = x_train[finite]
        y_train = y_train[finite]
        if len(y_train) < 2 or len(np.unique(x_train)) < 2:
            if len(y_train):
                pred[index] = float(np.nanmedian(y_train))
            continue
        try:
            if model_kind in {"robust_theil_sen", "theil_sen"} and len(y_train) >= 3:
                model = TheilSenRegressor(random_state=0)
            else:
                model = LinearRegression()
            model.fit(x_train.reshape(-1, 1), y_train)
            pred[index] = float(model.predict(np.asarray([[x[index]]], dtype=float))[0])
        except Exception:
            pred[index] = float(np.nanmedian(y_train))
    return pred


def saod2_fit_edge_trend(x, y, method="linear"):
    """
    Тренд для ряда разностей.

    Если точек мало — медиана.
    Если точек 5+ — линейная модель.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    method = (method or "linear").strip().lower()
    allowed = {
        "linear",
        "monotonic",
        "robust_theil_sen",
        "theil_sen",
        "lowess",
        "local_differences",
    }
    if method not in allowed:
        method = "linear"

    if len(y) < 3:
        pred = np.repeat(np.nanmedian(y), len(y))
        method = "median"

    elif method == "local_differences":
        order = np.argsort(x)
        pred = np.repeat(np.nanmedian(y), len(y))
        sorted_y = y[order]
        for rank, original_index in enumerate(order):
            lo = max(0, rank - 1)
            hi = min(len(sorted_y), rank + 2)
            neighbors = [
                sorted_y[i]
                for i in range(lo, hi)
                if i != rank and np.isfinite(sorted_y[i])
            ]
            if neighbors:
                pred[original_index] = float(np.nanmedian(neighbors))
        method = "local_differences"

    elif method == "monotonic":
        if len(y) < 4:
            pred = np.repeat(np.nanmedian(y), len(y))
            method = "median_small_series"
        else:
            try:
                increasing = bool(np.nanmedian(np.diff(y[np.argsort(x)])) >= 0)
                model = IsotonicRegression(
                    increasing=increasing,
                    out_of_bounds="clip",
                )
                pred = model.fit_transform(x, y)
                method = "monotonic_isotonic"
            except Exception:
                pred = np.repeat(np.nanmedian(y), len(y))
                method = "median_fallback"

    elif method in {"robust_theil_sen", "theil_sen"}:
        pred = saod2_leave_one_out_trend_predictions(
            x,
            y,
            model_kind="robust_theil_sen",
        )
        method = "robust_theil_sen_loo"

    elif method == "lowess":
        if len(y) < 6:
            pred = np.repeat(np.nanmedian(y), len(y))
            method = "median_small_series"
        else:
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess

                smooth = lowess(y, x, frac=0.75, return_sorted=False)
                pred = np.asarray(smooth, dtype=float)
                method = "lowess"
            except Exception:
                pred = np.repeat(np.nanmedian(y), len(y))
                method = "median_fallback"

    else:
        pred = saod2_leave_one_out_trend_predictions(
            x,
            y,
            model_kind="linear",
        )
        method = "linear_loo"

    return pred, method


def saod2_discover_rules(
    edge_table,
    min_points=3,
    trend_method="linear",
    measurement_uncertainty=None,
    sigma_floor=None,
):
    """
    Автоматически ищет правила между паттернами:
    - кто обычно выше;
    - устойчивость знака;
    - плавность разности;
    - поломки тренда.
    """
    if edge_table.empty:
        return pd.DataFrame(), pd.DataFrame()

    rule_rows = []
    edge_detail_rows = []

    for edge_label, g in edge_table.groupby("edge_label", dropna=False):
        g = g.sort_values("carbon_count").copy()

        if len(g) < min_points:
            continue

        x = g["carbon_count"].astype(float).values
        y = g["delta_a_minus_b"].astype(float).values

        valid = np.isfinite(x) & np.isfinite(y)

        g = g.loc[valid].copy()
        x = x[valid]
        y = y[valid]

        if len(g) < min_points:
            continue

        median_delta = np.nanmedian(y)
        mean_delta = np.nanmean(y)

        if median_delta > 0:
            main_sign = 1
            main_direction = (
                f"{g['pattern_a'].iloc[0]} обычно выше {g['pattern_b'].iloc[0]}"
            )
        elif median_delta < 0:
            main_sign = -1
            main_direction = (
                f"{g['pattern_a'].iloc[0]} обычно ниже {g['pattern_b'].iloc[0]}"
            )
        else:
            main_sign = 0
            main_direction = "разность около нуля"

        signs = np.sign(y)
        nonzero_signs = signs[signs != 0]

        if len(nonzero_signs) == 0 or main_sign == 0:
            sign_consistency = 0.0
        else:
            sign_consistency = np.mean(nonzero_signs == main_sign)

        pred, fitted_trend_method = saod2_fit_edge_trend(
            x,
            y,
            method=trend_method,
        )
        trend_comparison = {}
        for candidate_method in [
            "linear",
            "robust_theil_sen",
            "local_differences",
        ]:
            candidate_pred, candidate_label = saod2_fit_edge_trend(
                x,
                y,
                method=candidate_method,
            )
            trend_comparison[candidate_label] = np.asarray(
                candidate_pred,
                dtype=float,
            )
        comparison_matrix = np.vstack(
            list(trend_comparison.values())
        ) if trend_comparison else np.empty((0, len(y)))
        trend_model_spread = (
            np.nanmax(comparison_matrix, axis=0)
            - np.nanmin(comparison_matrix, axis=0)
            if comparison_matrix.size
            else np.repeat(np.nan, len(y))
        )
        max_trend_model_spread = (
            float(np.nanmax(trend_model_spread))
            if np.isfinite(trend_model_spread).any()
            else np.nan
        )

        residual = y - pred
        residual_sigma = saod2_robust_sigma(
            residual,
            measurement_uncertainty=measurement_uncertainty,
            min_sigma=sigma_floor,
        )
        residual_scores = np.abs(residual) / residual_sigma

        delta_change = np.full(len(y), np.nan)

        for i in range(1, len(y)):
            delta_change[i] = y[i] - y[i - 1]

        valid_changes = delta_change[np.isfinite(delta_change)]

        if len(valid_changes) >= 2:
            change_med = np.nanmedian(valid_changes)
            change_sigma = saod2_robust_sigma(
                valid_changes - change_med,
                measurement_uncertainty=measurement_uncertainty,
                min_sigma=sigma_floor,
            )
        else:
            change_med = np.nan
            change_sigma = np.nan

        change_scores = np.full(len(y), np.nan)

        for i in range(len(y)):
            if (
                np.isfinite(delta_change[i]) and
                np.isfinite(change_sigma) and
                change_sigma > 1e-12
            ):
                change_scores[i] = abs(delta_change[i] - change_med) / change_sigma
            elif np.isfinite(delta_change[i]):
                change_scores[i] = 0.0

        broken_formulas = []
        max_score = 0.0
        break_count = 0
        sign_break_count = 0

        for local_i, (_, row) in enumerate(g.iterrows()):
            residual_score = residual_scores[local_i]

            if np.isfinite(change_scores[local_i]):
                change_score = change_scores[local_i]
            else:
                change_score = 0.0

            combined_score = max(residual_score, change_score)

            if np.isfinite(combined_score):
                max_score = max(max_score, combined_score)

            current_delta = y[local_i]

            if main_sign == 0 or current_delta == 0:
                sign_break = False
            else:
                sign_break = np.sign(current_delta) != main_sign

            if sign_break:
                sign_break_count += 1

            if combined_score >= 3 or sign_break:
                break_count += 1
                broken_formulas.append(str(row["formula"]))

            if sign_break and combined_score >= 3:
                edge_status = "ломает знак и тренд"
            elif sign_break:
                edge_status = "ломает знак"
            elif combined_score >= 3:
                edge_status = "ломает плавность"
            else:
                edge_status = "согласовано"

            if pd.isna(delta_change[local_i]):
                delta_change_text = "—"
            else:
                if delta_change[local_i] > 0:
                    delta_change_text = (
                        f"разность увеличилась на {delta_change[local_i]:.3f}"
                    )
                elif delta_change[local_i] < 0:
                    delta_change_text = (
                        f"разность уменьшилась на {abs(delta_change[local_i]):.3f}"
                    )
                else:
                    delta_change_text = "разность не изменилась"

            edge_detail_rows.append({
                "edge_label": edge_label,
                "formula": row["formula"],
                "carbon_count": row["carbon_count"],

                "pattern_a": row["pattern_a"],
                "pattern_b": row["pattern_b"],

                "compound_a_id": row["compound_a_id"],
                "compound_b_id": row["compound_b_id"],

                "name_a": row["name_a"],
                "name_b": row["name_b"],

                "value_a": row["value_a"],
                "value_b": row["value_b"],

                "delta_a_minus_b": current_delta,
                "expected_delta": pred[local_i],
                "expected_delta_linear_loo": trend_comparison.get(
                    "linear_loo",
                    np.repeat(np.nan, len(y)),
                )[local_i],
                "expected_delta_theil_sen_loo": trend_comparison.get(
                    "robust_theil_sen_loo",
                    np.repeat(np.nan, len(y)),
                )[local_i],
                "expected_delta_local_differences": trend_comparison.get(
                    "local_differences",
                    np.repeat(np.nan, len(y)),
                )[local_i],
                "trend_model_spread": trend_model_spread[local_i],
                "delta_residual": residual[local_i],

                "delta_change_to_previous": delta_change[local_i],
                "delta_change_observation": delta_change_text,

                "residual_score": residual_score,
                "change_score": change_score,
                "combined_edge_score": combined_score,
                "edge_level": saod2_level(combined_score),

                "main_direction": main_direction,
                "sign_break": bool(sign_break),
                "edge_status": edge_status,

                "smiles_a": row["smiles_a"],
                "smiles_b": row["smiles_b"]
            })

        if len(y) >= 5:
            try:
                slope = LinearRegression().fit(x.reshape(-1, 1), y).coef_[0]
            except Exception:
                slope = np.nan
        else:
            slope = np.nan

        if len(y) < min_points:
            rule_status = "недостаточно данных"
            can_use = False
        elif sign_consistency < 0.7:
            rule_status = "направление неустойчиво"
            can_use = False
        elif break_count == 0 and max_score < 2:
            rule_status = "устойчивое правило"
            can_use = True
        elif break_count == 0 and max_score < 3:
            rule_status = "предварительное правило"
            can_use = True
        elif break_count > 0:
            rule_status = "правило найдено, но есть поломки"
            can_use = True
        else:
            rule_status = "слабое правило"
            can_use = False

        if can_use:
            recommendation = "Можно использовать для перекрёстной проверки значений."
        else:
            recommendation = "Использовать осторожно или не использовать как правило."

        rule_rows.append({
            "edge_label": edge_label,
            "pattern_a": g["pattern_a"].iloc[0],
            "pattern_b": g["pattern_b"].iloc[0],
            "n_points": len(y),
            "carbon_range": f"C{int(np.nanmin(x))}–C{int(np.nanmax(x))}",
            "main_direction": main_direction,
            "mean_delta": mean_delta,
            "median_delta": median_delta,
            "sign_consistency": sign_consistency,
            "trend_slope": slope,
            "trend_method": fitted_trend_method,
            "trend_models_compared": ", ".join(sorted(trend_comparison.keys())),
            "max_trend_model_spread": max_trend_model_spread,
            "measurement_uncertainty_used": measurement_uncertainty,
            "sigma_floor_used": sigma_floor,
            "max_edge_score": max_score,
            "break_count": break_count,
            "sign_break_count": sign_break_count,
            "rule_status": rule_status,
            "can_be_used_for_checking": can_use,
            "broken_formulas": "; ".join(sorted(set(broken_formulas))),
            "recommendation": recommendation
        })

    rules = pd.DataFrame(rule_rows)
    details = pd.DataFrame(edge_detail_rows)

    if not rules.empty:
        rules = rules.sort_values(
            ["can_be_used_for_checking", "rule_status", "n_points", "max_edge_score"],
            ascending=[False, True, False, False]
        ).reset_index(drop=True)

    if not details.empty:
        details = details.sort_values(
            ["edge_label", "carbon_count"]
        ).reset_index(drop=True)

    return rules, details


# ------------------------------------------------------------------
# Broken edges and suspicion

def saod2_broken_edges(edge_detail_table):
    """
    Выделяет конкретные формулы/связи, где правило ломается.
    """
    if edge_detail_table.empty:
        return pd.DataFrame()

    out = edge_detail_table[
        edge_detail_table["edge_status"].isin([
            "ломает знак и тренд",
            "ломает знак",
            "ломает плавность"
        ])
    ].copy()

    if out.empty:
        return out

    out = out.sort_values(
        "combined_edge_score",
        ascending=False,
        na_position="last"
    ).reset_index(drop=True)

    return out


def saod2_compound_suspicion(edge_table, edge_detail_table, broken_edges, threshold_config=None):
    """
    Определяет вещества, вокруг которых концентрируются поломки.
    """
    if edge_table.empty:
        return pd.DataFrame()

    threshold_config = saod2_get_threshold_config(
        threshold_config.get("mode", "standard") if isinstance(threshold_config, dict) else "standard",
        threshold_config if isinstance(threshold_config, dict) else None,
    )
    compounds = {}

    def ensure_compound(compound_id, name, pattern, formula, value):
        if compound_id not in compounds:
            compounds[compound_id] = {
                "compound_id": compound_id,
                "name": name,
                "pattern": pattern,
                "formula_examples": set(),
                "property_value_examples": [],
                "edges_total": 0,
                "edges_broken": 0,
                "max_edge_score": 0.0,
                "broken_edge_labels": set()
            }

        compounds[compound_id]["formula_examples"].add(str(formula))

        if pd.notna(value):
            compounds[compound_id]["property_value_examples"].append(value)

    for _, row in edge_table.iterrows():
        for side in ["a", "b"]:
            cid = row.get(f"compound_{side}_id", "")
            name = row.get(f"name_{side}", "")
            pattern = row.get(f"pattern_{side}", "")
            value = row.get(f"value_{side}", np.nan)

            ensure_compound(
                compound_id=cid,
                name=name,
                pattern=pattern,
                formula=row.get("formula", ""),
                value=value
            )

            compounds[cid]["edges_total"] += 1

    if not broken_edges.empty:
        for _, row in broken_edges.iterrows():
            for side in ["a", "b"]:
                cid = row.get(f"compound_{side}_id", "")
                name = row.get(f"name_{side}", "")
                pattern = row.get(f"pattern_{side}", "")
                value = row.get(f"value_{side}", np.nan)

                ensure_compound(
                    compound_id=cid,
                    name=name,
                    pattern=pattern,
                    formula=row.get("formula", ""),
                    value=value
                )

                compounds[cid]["edges_broken"] += 1

                score = row.get("combined_edge_score", np.nan)

                if pd.notna(score):
                    compounds[cid]["max_edge_score"] = max(
                        compounds[cid]["max_edge_score"],
                        score
                    )

                compounds[cid]["broken_edge_labels"].add(
                    row.get("edge_label", "")
                )

    rows = []

    for cid, info in compounds.items():
        if info["edges_total"] == 0:
            broken_fraction = np.nan
        else:
            broken_fraction = info["edges_broken"] / info["edges_total"]

        if info["edges_total"] < 2:
            status = SAOD_STATUS_UNCHECKABLE
        elif info["edges_broken"] == 0:
            status = SAOD_STATUS_AGREED
        elif (
            info["edges_broken"] >= threshold_config["edge_critical_broken"]
            or info["max_edge_score"] >= threshold_config["edge_critical_score"]
        ):
            status = SAOD_STATUS_PRIORITY_REVIEW
        elif (
            info["edges_broken"] >= threshold_config["edge_strong_broken"]
            or info["max_edge_score"] >= threshold_config["edge_strong_score"]
        ):
            status = SAOD_STATUS_STRUCTURAL_MISMATCH
        else:
            status = SAOD_STATUS_NEEDS_CHECK

        if status == SAOD_STATUS_UNCHECKABLE:
            recommendation = "Недостаточно структурных связей для проверки."
        elif status == SAOD_STATUS_AGREED:
            recommendation = "Значение согласовано с найденными правилами."
        elif status == SAOD_STATUS_PRIORITY_REVIEW:
            recommendation = (
                "Множественные несогласованности в независимых структурных проверках. "
                "Нужна приоритетная ручная проверка значения, единиц, структуры и источника данных."
            )
        else:
            recommendation = (
                "Проверить экспериментальное значение, единицы измерения, "
                "структуру, название вещества и источник данных. Это структурная эвристика, а не доказанная ошибка."
            )

        if info["property_value_examples"]:
            value_example = info["property_value_examples"][0]
        else:
            value_example = np.nan

        rows.append({
            "compound_id": cid,
            "name": info["name"],
            "pattern": info["pattern"],
            "formula_examples": "; ".join(sorted(info["formula_examples"])),
            "property_value_example": value_example,
            "edges_total": info["edges_total"],
            "edges_broken": info["edges_broken"],
            "broken_fraction": broken_fraction,
            "max_edge_score": info["max_edge_score"],
            "broken_edge_labels": "; ".join(sorted(info["broken_edge_labels"])),
            "final_status": status,
            "final_status_code": saod_status_code(status),
            "recommendation": recommendation
        })

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out = out.sort_values(
        ["final_status", "edges_broken", "max_edge_score"],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    return out


# ------------------------------------------------------------------
# Kitchen tables and plots

def saod2_make_edge_kitchen_table(edge_details, edge_label):
    """
    Делает подробную таблицу проверки одного правила / одной пары паттернов.
    """
    if edge_details.empty:
        return pd.DataFrame()

    group = edge_details[
        edge_details["edge_label"] == edge_label
    ].copy()

    if group.empty:
        return pd.DataFrame()

    group = group.sort_values("carbon_count").reset_index(drop=True)

    group["previous_formula"] = group["formula"].shift(1)
    group["previous_delta"] = group["delta_a_minus_b"].shift(1)

    group["delta_delta_to_previous_recalculated"] = (
        group["delta_a_minus_b"] - group["previous_delta"]
    )

    observations = []

    for _, row in group.iterrows():
        delta = row.get("delta_a_minus_b", np.nan)
        prev_delta = row.get("previous_delta", np.nan)
        delta_delta = row.get("delta_delta_to_previous_recalculated", np.nan)

        pattern_a = row.get("pattern_a", "")
        pattern_b = row.get("pattern_b", "")

        if pd.isna(delta):
            obs = "Разность не рассчитана."
        else:
            if delta > 0:
                obs = f"{pattern_a} выше {pattern_b} на {abs(delta):.3f}"
            elif delta < 0:
                obs = f"{pattern_b} выше {pattern_a} на {abs(delta):.3f}"
            else:
                obs = f"{pattern_a} и {pattern_b} имеют одинаковое значение"

        if pd.notna(prev_delta) and pd.notna(delta_delta):
            if delta_delta > 0:
                obs += (
                    f"; по сравнению с предыдущей формулой "
                    f"разность увеличилась на {delta_delta:.3f}"
                )
            elif delta_delta < 0:
                obs += (
                    f"; по сравнению с предыдущей формулой "
                    f"разность уменьшилась на {abs(delta_delta):.3f}"
                )
            else:
                obs += "; по сравнению с предыдущей формулой разность не изменилась"
        else:
            obs += "; первая точка ряда, ΔΔ не рассчитывается"

        observations.append(obs)

    group["human_observation"] = observations

    show_cols = [
        "formula",
        "carbon_count",

        "pattern_a",
        "compound_a_id",
        "name_a",
        "value_a",

        "pattern_b",
        "compound_b_id",
        "name_b",
        "value_b",

        "delta_a_minus_b",
        "previous_formula",
        "previous_delta",
        "delta_delta_to_previous_recalculated",

        "expected_delta",
        "delta_residual",
        "residual_score",
        "change_score",
        "combined_edge_score",

        "main_direction",
        "sign_break",
        "edge_status",
        "edge_level",
        "human_observation"
    ]

    show_cols = [c for c in show_cols if c in group.columns]

    return group[show_cols]


def saod2_edge_kitchen_explanation(edge_details, edge_label):
    """
    Текстовое объяснение кухни проверки выбранной пары паттернов.
    """
    if edge_details.empty:
        return "Нет данных для объяснения."

    group = edge_details[
        edge_details["edge_label"] == edge_label
    ].copy()

    if group.empty:
        return "Для выбранной пары паттернов нет данных."

    group = group.sort_values("carbon_count")

    pattern_a = group["pattern_a"].iloc[0]
    pattern_b = group["pattern_b"].iloc[0]

    deltas = group["delta_a_minus_b"].astype(float).values
    carbons = group["carbon_count"].astype(int).values

    median_delta = np.nanmedian(deltas)

    if median_delta > 0:
        direction = f"{pattern_a} обычно выше {pattern_b}"
    elif median_delta < 0:
        direction = f"{pattern_b} обычно выше {pattern_a}"
    else:
        direction = "разность между паттернами около нуля"

    delta_sequence = "; ".join([
        f"C{c}: {d:.3f}"
        for c, d in zip(carbons, deltas)
    ])

    changes = []

    for i in range(1, len(deltas)):
        changes.append(deltas[i] - deltas[i - 1])

    if changes:
        change_sequence = "; ".join([
            f"C{carbons[i - 1]}→C{carbons[i]}: {changes[i - 1]:+.3f}"
            for i in range(1, len(carbons))
        ])
    else:
        change_sequence = "Недостаточно точек для расчёта ΔΔ."

    broken = group[group["edge_status"] != "согласовано"].copy()

    if broken.empty:
        broken_text = "Явных поломок для этой пары не найдено."
    else:
        broken_formulas = ", ".join(broken["formula"].astype(str).tolist())
        broken_text = f"Поломки отмечены для формул: {broken_formulas}."

    text = f"""
**Проверяемая пара паттернов:** `{pattern_a}` и `{pattern_b}`

**Что считается:**  
Δ = значение(`{pattern_a}`) − значение(`{pattern_b}`)

**Найденное направление:**  
{direction}

**Ряд Δ по формулам:**  
{delta_sequence}

**Изменение самой Δ между соседними формулами:**  
{change_sequence}

**Итог по выбранной паре:**  
{broken_text}

Интерпретация: если Δ меняется плавно, пара паттернов согласована.  
Если Δ резко меняется или меняет знак, это не доказывает ошибку автоматически,  
но показывает место, где нужна перекрёстная проверка через другие связи.
"""

    return text


def saod2_plot_edge_delta(edge_details, edge_label):
    """
    График:
    X = carbon_count
    Y = delta = property(A) - property(B)
    """
    if edge_details.empty:
        return None

    group = edge_details[
        edge_details["edge_label"] == edge_label
    ].copy()

    if group.empty or len(group) < 2:
        return None

    import matplotlib.pyplot as plt

    group = group.sort_values("carbon_count")

    fig, ax = plt.subplots(figsize=(5.2, 3.6))

    ax.plot(
        group["carbon_count"],
        group["delta_a_minus_b"],
        marker="o",
        linewidth=2,
        label="Наблюдаемая Δ"
    )

    if "expected_delta" in group.columns and group["expected_delta"].notna().any():
        ax.plot(
            group["carbon_count"],
            group["expected_delta"],
            marker="s",
            linestyle="--",
            linewidth=1.5,
            label="Ожидаемая Δ по тренду"
        )

    for _, row in group.iterrows():
        label = str(row.get("formula", ""))

        ax.annotate(
            label,
            (row["carbon_count"], row["delta_a_minus_b"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7
        )

    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_title(f"Δ свойства: {edge_label}", fontsize=9)
    ax.set_xlabel("Число атомов углерода", fontsize=8)
    ax.set_ylabel("Δ = значение A − значение B", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, alpha=0.3)

    x_values = sorted(group["carbon_count"].dropna().astype(int).unique())
    ax.set_xticks(x_values)
    ax.set_xticklabels([str(x) for x in x_values])

    ax.legend(fontsize=7)
    fig.tight_layout()

    return fig


def saod2_plot_edge_delta_change(edge_details, edge_label):
    """
    График:
    X = переход между формулами
    Y = ΔΔ = Δ(Cn) - Δ(Cn-1)
    """
    if edge_details.empty:
        return None

    group = edge_details[
        edge_details["edge_label"] == edge_label
    ].copy()

    if group.empty or len(group) < 3:
        return None

    import matplotlib.pyplot as plt

    group = group.sort_values("carbon_count").reset_index(drop=True)

    group["previous_carbon_count"] = group["carbon_count"].shift(1)
    group["previous_delta"] = group["delta_a_minus_b"].shift(1)
    group["delta_delta"] = group["delta_a_minus_b"] - group["previous_delta"]

    plot_group = group[group["delta_delta"].notna()].copy()

    if plot_group.empty:
        return None

    x_labels = []

    for _, row in plot_group.iterrows():
        try:
            prev_c = int(row["previous_carbon_count"])
            curr_c = int(row["carbon_count"])
            x_labels.append(f"C{prev_c}→C{curr_c}")
        except Exception:
            x_labels.append(str(row.get("formula", "")))

    fig, ax = plt.subplots(figsize=(5.2, 3.6))

    ax.plot(
        x_labels,
        plot_group["delta_delta"],
        marker="o",
        linewidth=2,
        label="ΔΔ"
    )

    ax.axhline(0, linestyle="--", linewidth=1)

    for i, (_, row) in enumerate(plot_group.iterrows()):
        label = str(row.get("formula", ""))

        ax.annotate(
            label,
            (i, row["delta_delta"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7
        )

    ax.set_title(f"Изменение Δ между формулами: {edge_label}", fontsize=9)
    ax.set_xlabel("Переход между соседними формулами", fontsize=8)
    ax.set_ylabel("ΔΔ = Δ текущая − Δ предыдущая", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    return fig


# ------------------------------------------------------------------
# Dataset summary

def saod2_dataset_summary(processed, checkability, rules, broken_edges, suspicion, threshold_config=None):
    """
    Краткая сводка по SAOD v2.
    """
    rows = []

    total = len(processed)

    if "valid_structure" in processed.columns:
        valid = int(processed["valid_structure"].sum())
    else:
        valid = 0

    if "is_acyclic_alkane" in processed.columns:
        alk = int(processed["is_acyclic_alkane"].sum())
    else:
        alk = 0

    invalid = total - valid

    rows.append({"Показатель": "Всего строк", "Значение": total})
    rows.append({"Показатель": "Валидных структур", "Значение": valid})
    rows.append({"Показатель": "Некорректных структур", "Значение": invalid})
    rows.append({"Показатель": "Ациклических алканов", "Значение": alk})

    if not checkability.empty:
        if "checkability_code" in checkability.columns or "checkability_level" in checkability.columns:
            checkability_codes = (
                checkability["checkability_code"].astype(str)
                if "checkability_code" in checkability.columns
                else checkability.apply(
                    lambda row: saod_checkability_code(
                        row.get("checkability_level", ""),
                        row.get("checkability_score", np.nan),
                    ),
                    axis=1,
                )
            )
            rows.append({
                "Показатель": "Почти непроверяемых веществ",
                "Значение": int(
                    (checkability_codes == SAOD_CHECKABILITY_UNCHECKABLE).sum()
                )
            })

            rows.append({
                "Показатель": "Хорошо проверяемых веществ",
                "Значение": int(
                    (checkability_codes == SAOD_CHECKABILITY_GOOD).sum()
                )
            })

    if not rules.empty:
        rows.append({
            "Показатель": "Найдено правил",
            "Значение": len(rules)
        })

        if "can_be_used_for_checking" in rules.columns:
            rows.append({
                "Показатель": "Правил, пригодных для проверки",
                "Значение": int(rules["can_be_used_for_checking"].sum())
            })

    if not broken_edges.empty:
        rows.append({
            "Показатель": "Поломок правил",
            "Значение": len(broken_edges)
        })

    if not suspicion.empty:
        if "final_status_code" in suspicion.columns or "final_status" in suspicion.columns:
            final_status_codes = (
                suspicion["final_status_code"].astype(str)
                if "final_status_code" in suspicion.columns
                else suspicion["final_status"].map(saod_status_code)
            )
            rows.append({
                "Показатель": "Веществ с приоритетной ручной проверкой",
                "Значение": int(
                    (final_status_codes == "PRIORITY_REVIEW").sum()
                )
            })

            rows.append({
                "Показатель": "Веществ с сильным структурным несоответствием",
                "Значение": int(
                    (final_status_codes == "STRUCTURAL_MISMATCH").sum()
                )
            })

    if threshold_config:
        threshold_config = saod2_get_threshold_config(
            threshold_config.get("mode", "standard"),
            threshold_config,
        )
        rows.append({
            "Показатель": "Режим порогов SAOD",
            "Значение": threshold_config["label"],
        })
        rows.append({
            "Показатель": "Использованные пороги SAOD",
            "Значение": "; ".join(
                f"{key}={value}"
                for key, value in threshold_config.items()
                if key not in {"label", "mode"}
            ),
        })

    return pd.DataFrame(rows)


def saod2_analysis_reason_label(reason_code):
    labels = {
        "analyzed": "вошло в анализ",
        "empty_smiles": "пустой SMILES",
        "invalid_smiles": "невалидный SMILES",
        "mixture": "смесь",
        "missing_property": "нет целевого свойства",
        "standardization_failed": "не удалось стандартизовать",
        "unsupported_saod_mode": "не поддерживается данным SAOD-режимом",
        "no_structural_analogs": "нет структурных аналогов",
        "insufficient_points": "нет достаточного числа точек",
        "calculation_failed": "расчёт завершился ошибкой",
    }
    return labels.get(str(reason_code), str(reason_code))


def saod2_analysis_status_label(status_code):
    labels = {
        SAOD_ANALYSIS_STATUS_ANALYZED: "проанализировано",
        SAOD_ANALYSIS_STATUS_EXCLUDED: "исключено",
        SAOD_ANALYSIS_STATUS_UNSUPPORTED: "не поддерживается",
        SAOD_ANALYSIS_STATUS_INSUFFICIENT_SUPPORT: "нет данных для опоры",
        SAOD_ANALYSIS_STATUS_FAILED: "ошибка расчёта",
    }
    return labels.get(str(status_code), str(status_code))


def saod2_build_analysis_inclusion_tables(
    processed,
    memberships,
    checkability,
    min_rule_points=3,
    errors=None,
):
    if processed is None or processed.empty:
        coverage = pd.DataFrame([
            {"metric": "loaded", "value": 0},
            {"metric": "analyzed", "value": 0},
            {"metric": "excluded", "value": 0},
        ])
        empty = pd.DataFrame()
        return coverage, empty, empty

    support_counts = {}
    max_series_size = {}
    if memberships is not None and not memberships.empty:
        support_counts = memberships.groupby("compound_id", dropna=False)["series_id"].nunique().to_dict()
        max_series_size = memberships.groupby("compound_id", dropna=False)["series_size"].max().to_dict()

    exact_pattern_sizes = {}
    if "exact_pattern" in processed.columns:
        supported_patterns = processed[
            processed.get("exact_pattern", "").astype(str).str.len() > 0
        ].copy()
        if not supported_patterns.empty:
            exact_pattern_sizes = (
                supported_patterns.groupby("exact_pattern", dropna=False)["compound_id"]
                .nunique()
                .to_dict()
            )

    checkability_by_compound = {}
    if checkability is not None and not checkability.empty and "compound_id" in checkability.columns:
        checkability_by_compound = (
            checkability.set_index("compound_id").to_dict(orient="index")
        )

    rows = []
    calculation_failed = bool(errors)
    for _, row in processed.iterrows():
        compound_id = str(row.get("compound_id", ""))
        smiles = str(row.get("input_smiles", "") or "").strip()
        structure_status = str(row.get("structure_status", "") or "").strip()
        structure_status_code = str(
            row.get("structure_status_code", "")
            or saod2_structure_status_code(structure_status)
        )
        property_missing = pd.isna(row.get("property_value", np.nan))
        valid_structure = _saod_coerce_boolean_value(row.get("valid_structure", False))
        n_series = int(support_counts.get(compound_id, 0) or 0)
        largest_series = int(max_series_size.get(compound_id, 0) or 0)
        series_supported = _saod_coerce_boolean_value(row.get("series_supported", False))
        series_reason = str(row.get("series_reason", "") or "")
        exact_pattern = str(row.get("exact_pattern", "") or "")
        if largest_series <= 0 and exact_pattern:
            largest_series = int(exact_pattern_sizes.get(exact_pattern, 0) or 0)

        if calculation_failed:
            status_code = SAOD_ANALYSIS_STATUS_FAILED
            reason_code = "calculation_failed"
        elif not smiles:
            status_code = SAOD_ANALYSIS_STATUS_EXCLUDED
            reason_code = "empty_smiles"
        elif structure_status_code == "MIXTURE":
            status_code = SAOD_ANALYSIS_STATUS_EXCLUDED
            reason_code = "mixture"
        elif structure_status_code == "STANDARDIZATION_FAILED":
            status_code = SAOD_ANALYSIS_STATUS_FAILED
            reason_code = "standardization_failed"
        elif structure_status_code == "INVALID_SMILES" or not valid_structure:
            status_code = SAOD_ANALYSIS_STATUS_EXCLUDED
            reason_code = "invalid_smiles"
        elif property_missing:
            status_code = SAOD_ANALYSIS_STATUS_EXCLUDED
            reason_code = "missing_property"
        elif n_series > 0:
            status_code = SAOD_ANALYSIS_STATUS_ANALYZED
            reason_code = "analyzed"
        elif not series_supported and "unsupported" in series_reason.lower():
            status_code = SAOD_ANALYSIS_STATUS_UNSUPPORTED
            reason_code = "unsupported_saod_mode"
        elif largest_series > 0 and largest_series < int(min_rule_points):
            status_code = SAOD_ANALYSIS_STATUS_INSUFFICIENT_SUPPORT
            reason_code = "insufficient_points"
        else:
            status_code = SAOD_ANALYSIS_STATUS_INSUFFICIENT_SUPPORT
            reason_code = "no_structural_analogs"

        checkability_row = checkability_by_compound.get(compound_id, {})
        rows.append({
            "record_id": row.get("record_id", ""),
            "compound_id": compound_id,
            "source_row": row.get("source_row", row.get("row_index", "")),
            "name": row.get("name", ""),
            "input_smiles": row.get("input_smiles", ""),
            "canonical_smiles": row.get("canonical_smiles", ""),
            "property_value": row.get("property_value", np.nan),
            "status_code": status_code,
            "status_label": saod2_analysis_status_label(status_code),
            "reason_code": reason_code,
            "reason": saod2_analysis_reason_label(reason_code),
            "structure_status": structure_status,
            "series_supported": series_supported,
            "series_reason": series_reason,
            "support_series_count": n_series,
            "largest_supported_series_size": largest_series,
            "checkability_level": checkability_row.get("checkability_level", ""),
            "structural_support": checkability_row.get("structural_support", ""),
            "data_noise_evidence": checkability_row.get("data_noise_evidence", ""),
            "structural_uniqueness": checkability_row.get("structural_uniqueness", ""),
            "property_inconsistency": checkability_row.get("property_inconsistency", ""),
            "model_residual_outlier": checkability_row.get("model_residual_outlier", ""),
            "data_quality_issue": checkability_row.get("data_quality_issue", ""),
            "checkability_comment": checkability_row.get("checkability_comment", ""),
        })

    inclusion = pd.DataFrame(rows)
    excluded = inclusion[
        inclusion["status_code"] != SAOD_ANALYSIS_STATUS_ANALYZED
    ].copy()
    coverage = pd.DataFrame([
        {"metric": "loaded", "label": "загружено", "value": int(len(inclusion))},
        {
            "metric": "analyzed",
            "label": "проанализировано",
            "value": int((inclusion["status_code"] == SAOD_ANALYSIS_STATUS_ANALYZED).sum()),
        },
        {"metric": "excluded", "label": "исключено", "value": int(len(excluded))},
        {
            "metric": "unsupported",
            "label": "не поддерживается",
            "value": int((inclusion["status_code"] == SAOD_ANALYSIS_STATUS_UNSUPPORTED).sum()),
        },
        {
            "metric": "insufficient_support",
            "label": "нет данных для опоры",
            "value": int((inclusion["status_code"] == SAOD_ANALYSIS_STATUS_INSUFFICIENT_SUPPORT).sum()),
        },
        {
            "metric": "failed",
            "label": "ошибка расчёта",
            "value": int((inclusion["status_code"] == SAOD_ANALYSIS_STATUS_FAILED).sum()),
        },
    ])
    return coverage, inclusion, excluded.reset_index(drop=True)


# ------------------------------------------------------------------
# Universal automatic chemical-series discovery

def saod3_fragment_descriptor(fragment_smiles):
    mol = Chem.MolFromSmiles(str(fragment_smiles))
    if mol is None:
        return None

    atoms = [
        atom for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 0
    ]
    elements = sorted(
        atom.GetSymbol() for atom in atoms
        if atom.GetAtomicNum() != 6
    )
    carbon_count = sum(atom.GetAtomicNum() == 6 for atom in atoms)
    heavy_count = len(atoms)
    ring_count = mol.GetRingInfo().NumRings()
    unsaturated = any(
        bond.GetBondType() != Chem.BondType.SINGLE
        for bond in mol.GetBonds()
        if (
            bond.GetBeginAtom().GetAtomicNum() > 0
            and bond.GetEndAtom().GetAtomicNum() > 0
        )
    )
    halogens = [
        atom.GetAtomicNum()
        for atom in atoms
        if atom.GetAtomicNum() in [9, 17, 35, 53]
    ]
    try:
        canonical_fragment_smiles = Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        canonical_fragment_smiles = ""
    carbon_branch_degrees = []
    for atom in atoms:
        if atom.GetAtomicNum() != 6:
            continue
        carbon_branch_degrees.append(
            sum(
                1
                for neighbor in atom.GetNeighbors()
                if neighbor.GetAtomicNum() == 6
            )
        )
    branching_topology = (
        "branched"
        if carbon_branch_degrees and max(carbon_branch_degrees) > 2
        else "linear_or_unbranched"
    )

    if heavy_count == 1 and len(halogens) == 1:
        family = "halogen"
        coordinate = float({9: 1, 17: 2, 35: 3, 53: 4}[halogens[0]])
        coordinate_type = "halogen_periodic_order_heuristic"
        coordinate_metric_role = "ordinal_category_not_metric_distance"
    elif ring_count == 0 and not unsaturated:
        family = (
            "acyclic:"
            + ",".join(elements)
            + f":topology={branching_topology}:smiles={canonical_fragment_smiles}"
        )
        coordinate = float(carbon_count)
        coordinate_type = "carbon_count"
        coordinate_metric_role = "coarse_numeric_descriptor"
    else:
        family = f"unordered:{','.join(elements)}:rings{ring_count}"
        coordinate = np.nan
        coordinate_type = "unordered"
        coordinate_metric_role = "unordered_category"

    return {
        "variable_family": family,
        "variable_coordinate": coordinate,
        "coordinate_type": coordinate_type,
        "coordinate_metric_role": coordinate_metric_role,
        "variable_carbon_count": carbon_count,
        "variable_heavy_count": heavy_count,
        "variable_elements": ",".join(elements),
        "variable_ring_count": ring_count,
        "variable_unsaturated": unsaturated,
        "variable_canonical_smiles": canonical_fragment_smiles,
        "variable_branching_topology": branching_topology,
        "coordinate_semantics": (
            "ordinal_heuristic_not_universal_linear"
            if coordinate_type == "halogen_periodic_order_heuristic"
            else "numeric_descriptor"
        ),
    }


def _saod3_cut_quality(mol, bond, core, variable, core_heavy, variable_heavy, total_heavy):
    begin = bond.GetBeginAtom()
    end = bond.GetEndAtom()
    core_fraction = float(core_heavy) / float(total_heavy) if total_heavy else 0.0
    variable_fraction = float(variable_heavy) / float(total_heavy) if total_heavy else 0.0
    core_ring_atoms = sum(atom.IsInRing() for atom in core.GetAtoms() if atom.GetAtomicNum() > 0)
    variable_ring_atoms = sum(atom.IsInRing() for atom in variable.GetAtoms() if atom.GetAtomicNum() > 0)
    core_ring_fraction = float(core_ring_atoms) / max(float(core_heavy), 1.0)
    variable_ring_fraction = float(variable_ring_atoms) / max(float(variable_heavy), 1.0)

    def is_carbonyl_carbon(atom):
        if atom.GetAtomicNum() != 6:
            return False
        return any(
            nb.GetAtomicNum() == 8
            and mol.GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
            for nb in atom.GetNeighbors()
        )

    amide_like_cut = (
        (is_carbonyl_carbon(begin) and end.GetAtomicNum() == 7)
        or (is_carbonyl_carbon(end) and begin.GetAtomicNum() == 7)
    )
    terminal_replacement = variable_heavy <= max(6, int(round(total_heavy * 0.35)))
    keeps_ring_scaffold = core_ring_atoms >= variable_ring_atoms
    near_symmetric_cut = abs(core_heavy - variable_heavy) <= 1

    score = 0.0
    if terminal_replacement:
        score += 2.0
    if keeps_ring_scaffold:
        score += 2.0
    if core_fraction >= 0.55:
        score += 1.5
    if variable_fraction <= 0.35:
        score += 1.0
    if amide_like_cut:
        score -= 3.0
    if near_symmetric_cut:
        score -= 2.0
    if core_ring_fraction < variable_ring_fraction:
        score -= 1.5

    if amide_like_cut:
        category = "penalized_amide_or_key_bond"
    elif near_symmetric_cut:
        category = "ambiguous_symmetric_cut"
    elif terminal_replacement and keeps_ring_scaffold and core_fraction >= 0.55:
        category = "synthetically_interpretable_terminal_replacement"
    elif keeps_ring_scaffold:
        category = "secondary_scaffold_cut"
    else:
        category = "low_confidence_cut"

    return {
        "cut_quality_score": float(score),
        "cut_quality_category": category,
        "core_fraction": core_fraction,
        "variable_fraction": variable_fraction,
        "core_ring_atom_count": int(core_ring_atoms),
        "variable_ring_atom_count": int(variable_ring_atoms),
        "terminal_replacement_cut": bool(terminal_replacement),
        "keeps_ring_scaffold": bool(keeps_ring_scaffold),
        "amide_like_cut": bool(amide_like_cut),
        "near_symmetric_cut": bool(near_symmetric_cut),
    }


def saod3_single_cut_fragments(mol, min_core_heavy_atoms=3):
    """
    Generates candidate one-attachment cores without requiring a predefined
    chemical class. Ring bonds and terminal H-like cuts are excluded.
    """
    candidates = []
    total_heavy = mol.GetNumHeavyAtoms()

    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        if (
            bond.GetBeginAtom().GetAtomicNum() <= 1
            or bond.GetEndAtom().GetAtomicNum() <= 1
        ):
            continue

        try:
            fragmented = Chem.FragmentOnBonds(
                mol,
                [bond.GetIdx()],
                addDummies=True,
                dummyLabels=[(0, 0)],
            )
            fragments = Chem.GetMolFrags(
                fragmented,
                asMols=True,
                sanitizeFrags=True,
            )
        except Exception:
            continue

        if len(fragments) != 2:
            continue

        oriented = []
        for core, variable in [
            (fragments[0], fragments[1]),
            (fragments[1], fragments[0]),
        ]:
            core_heavy = sum(
                atom.GetAtomicNum() > 0 for atom in core.GetAtoms()
            )
            variable_heavy = sum(
                atom.GetAtomicNum() > 0 for atom in variable.GetAtoms()
            )
            if core_heavy < min_core_heavy_atoms:
                continue
            if core_heavy < variable_heavy:
                continue
            if core_heavy == variable_heavy:
                continue
            if total_heavy and core_heavy / total_heavy < 0.55:
                continue

            core_smiles = Chem.MolToSmiles(core, canonical=True)
            variable_smiles = Chem.MolToSmiles(variable, canonical=True)
            descriptor = saod3_fragment_descriptor(variable_smiles)
            if descriptor is None:
                continue
            quality = _saod3_cut_quality(
                mol,
                bond,
                core,
                variable,
                core_heavy,
                variable_heavy,
                total_heavy,
            )
            if quality["cut_quality_score"] < 1.0:
                continue

            oriented.append({
                "core_smiles": core_smiles,
                "variable_smiles": variable_smiles,
                "core_heavy_atoms": core_heavy,
                "variable_heavy_atoms": variable_heavy,
                "cut_bond_index": int(bond.GetIdx()),
                **quality,
                **descriptor,
            })
        if oriented:
            oriented = sorted(
                oriented,
                key=lambda item: (
                    -float(item["cut_quality_score"]),
                    -int(item["core_heavy_atoms"]),
                    str(item["core_smiles"]),
                    str(item["variable_smiles"]),
                ),
            )
            candidates.append(oriented[0])

    unique = {}
    for item in candidates:
        key = (item["core_smiles"], item["variable_smiles"])
        if key not in unique or item["cut_quality_score"] > unique[key]["cut_quality_score"]:
            unique[key] = item
    ranked = sorted(
        unique.values(),
        key=lambda item: (
            -float(item["cut_quality_score"]),
            -int(item["core_heavy_atoms"]),
            str(item["core_smiles"]),
            str(item["variable_smiles"]),
        ),
    )
    for rank, item in enumerate(ranked, start=1):
        item["scaffold_candidate_rank"] = int(rank)
        if rank == 1:
            item["scaffold_candidate_class"] = "primary_scaffold"
        elif rank <= 3 and float(item["cut_quality_score"]) >= 2.0:
            item["scaffold_candidate_class"] = "secondary_scaffold"
        else:
            item["scaffold_candidate_class"] = "redundant_or_nested_scaffold"
    return [
        item
        for item in ranked
        if item["scaffold_candidate_class"] != "redundant_or_nested_scaffold"
    ]


def saod3_discover_series_memberships(processed, min_series_points=3):
    rows = []

    # Profile-derived homologous rows preserve strong chemistry where it can
    # be inferred unambiguously, but use the same downstream series engine.
    for _, row in processed.iterrows():
        if (
            _saod_coerce_boolean_value(row.get("series_supported", False))
            and pd.notna(row.get("property_value", np.nan))
        ):
            rows.append({
                "series_source": "graph_profile",
                "series_id": (
                    f"profile::{row.get('experimental_context_id', 'CTX_DEFAULT')}"
                    f"::{row['series_pattern']}"
                ),
                "experimental_context_id": str(
                    row.get("experimental_context_id", "CTX_DEFAULT")
                ),
                "series_domain": str(row.get("series_family", "")),
                "core_smiles": "",
                "variable_smiles": "",
                "variable_family": str(row.get("series_family", "")),
                "coordinate_type": "total_carbon_count",
                "coordinate_metric_role": "coarse_numeric_descriptor",
                "series_coordinate": float(row["series_coordinate"]),
                "series_complexity": float(
                    row.get("series_complexity", 0)
                ),
                "record_id": str(row.get("record_id", "")),
                "compound_id": str(row.get("compound_id", "")),
                "assay_id": str(row.get("assay_id", "")),
                "name": str(row.get("name", "")),
                "canonical_smiles": str(
                    row.get("canonical_smiles", "")
                ),
                "molecular_formula": str(
                    row.get("molecular_formula", "")
                ),
                "property_value": float(row["property_value"]),
                "scaffold_id": "",
                "scaffold_smiles": "",
                "variable_family_id": "",
                "scaffold_confidence": "profile",
                "cut_quality": "not_applicable",
                "cut_quality_score": np.nan,
                "primary_membership": True,
                "secondary_membership": False,
                "ambiguity_score": 0.0,
                "raw_candidate_series_count": 0,
                "deduplicated_series_count": 0,
                "independent_scaffold_families": 0,
            })

    fragment_rows = []
    for _, row in processed.iterrows():
        if (
            not _saod_coerce_boolean_value(row.get("valid_structure", False))
            or pd.isna(row.get("property_value", np.nan))
        ):
            continue
        mol = Chem.MolFromSmiles(str(row.get("canonical_smiles", "")))
        if mol is None:
            continue
        for fragment in saod3_single_cut_fragments(mol):
            fragment_rows.append({
                **fragment,
                "record_id": str(row.get("record_id", "")),
                "compound_id": str(row.get("compound_id", "")),
                "assay_id": str(row.get("assay_id", "")),
                "experimental_context_id": str(
                    row.get("experimental_context_id", "CTX_DEFAULT")
                ),
                "name": str(row.get("name", "")),
                "canonical_smiles": str(
                    row.get("canonical_smiles", "")
                ),
                "molecular_formula": str(
                    row.get("molecular_formula", "")
                ),
                "property_value": float(row["property_value"]),
            })

    fragments = pd.DataFrame(fragment_rows)
    if not fragments.empty:
        raw_candidate_series_count = int(
            fragments.groupby(
                ["core_smiles", "variable_family"],
                dropna=False,
            ).ngroups
        )
        fragments = fragments.drop_duplicates(
            ["core_smiles", "variable_smiles", "record_id"]
        )

        core_sizes = (
            fragments.groupby("core_smiles", dropna=False)["record_id"]
            .nunique()
            .to_dict()
        )
        fragments = fragments[
            fragments["core_smiles"].map(core_sizes)
            >= int(min_series_points)
        ].copy()

        if not fragments.empty:
            filtered_core_sizes = (
                fragments.groupby("core_smiles", dropna=False)["record_id"]
                .nunique()
                .to_dict()
            )
            ordered_cores = sorted(
                filtered_core_sizes,
                key=lambda core: (-int(filtered_core_sizes.get(core, 0)), str(core)),
            )
            scaffold_ids = {
                core: f"SCF_{pos:06d}"
                for pos, core in enumerate(ordered_cores, start=1)
            }
            fragments["scaffold_id"] = fragments["core_smiles"].map(scaffold_ids)
            fragments["scaffold_smiles"] = fragments["core_smiles"].astype(str)
            fragments["variable_family_id"] = fragments.apply(
                lambda row: "VAR_"
                + hashlib.sha1(
                    (
                        str(row.get("scaffold_id", ""))
                        + "::"
                        + str(row.get("variable_family", ""))
                    ).encode("utf-8", errors="replace")
                ).hexdigest()[:10].upper(),
                axis=1,
            )
            scaffold_counts_by_compound = (
                fragments.groupby("record_id", dropna=False)["scaffold_id"]
                .nunique()
                .to_dict()
            )
            fragments["ambiguity_score"] = fragments["record_id"].map(
                lambda value: float(
                    max(0, int(scaffold_counts_by_compound.get(value, 0)) - 1)
                )
            )
            fragments["primary_membership"] = (
                fragments.get("scaffold_candidate_class", "")
                .astype(str)
                .eq("primary_scaffold")
            )
            fragments["secondary_membership"] = (
                fragments.get("scaffold_candidate_class", "")
                .astype(str)
                .eq("secondary_scaffold")
            )
            cut_scores = pd.to_numeric(
                fragments.get("cut_quality_score", np.nan),
                errors="coerce",
            )
            fragments["scaffold_confidence"] = np.select(
                [
                    fragments["primary_membership"] & cut_scores.ge(5.0),
                    fragments["primary_membership"] | cut_scores.ge(3.0),
                ],
                ["high", "moderate"],
                default="low",
            )
            fragments["cut_quality"] = (
                fragments.get("cut_quality_category", "unknown")
                .fillna("unknown")
                .astype(str)
            )
            deduplicated_series_count = int(
                fragments.groupby(
                    ["scaffold_id", "variable_family_id"],
                    dropna=False,
                ).ngroups
            )
            independent_scaffold_families = int(
                fragments["scaffold_id"].nunique(dropna=True)
            )

        # A core can yield several coherent variable families. Each becomes a
        # series; unordered families remain useful for transformation checks.
        for _, item in fragments.iterrows():
            context_id = str(item.get("experimental_context_id", "CTX_DEFAULT"))
            series_id = (
                f"{context_id}::{item['scaffold_id']}::{item['variable_family_id']}"
            )
            rows.append({
                "series_source": "automatic_scaffold",
                "series_id": series_id,
                "experimental_context_id": item.get(
                    "experimental_context_id",
                    "CTX_DEFAULT",
                ),
                "series_domain": (
                    f"{item['scaffold_id']}::"
                    f"{item['coordinate_type']}"
                ),
                "scaffold_id": item["scaffold_id"],
                "scaffold_smiles": item["scaffold_smiles"],
                "core_smiles": item["core_smiles"],
                "variable_smiles": item["variable_smiles"],
                "variable_family": item["variable_family"],
                "variable_family_id": item["variable_family_id"],
                "coordinate_type": item["coordinate_type"],
                "coordinate_metric_role": item.get(
                    "coordinate_metric_role",
                    "unknown",
                ),
                "series_coordinate": item["variable_coordinate"],
                "series_complexity": float(item["core_heavy_atoms"]),
                "scaffold_confidence": item["scaffold_confidence"],
                "cut_quality": item["cut_quality"],
                "cut_quality_score": item.get("cut_quality_score", np.nan),
                "primary_membership": bool(item["primary_membership"]),
                "secondary_membership": bool(item["secondary_membership"]),
                "ambiguity_score": float(item["ambiguity_score"]),
                "raw_candidate_series_count": raw_candidate_series_count,
                "deduplicated_series_count": deduplicated_series_count,
                "independent_scaffold_families": independent_scaffold_families,
                "record_id": item["record_id"],
                "compound_id": item["compound_id"],
                "assay_id": item.get("assay_id", ""),
                "name": item["name"],
                "canonical_smiles": item["canonical_smiles"],
                "molecular_formula": item["molecular_formula"],
                "property_value": item["property_value"],
            })

    memberships = pd.DataFrame(rows)
    if memberships.empty:
        return memberships

    memberships = memberships.drop_duplicates(
        ["series_id", "record_id", "variable_smiles"]
    ).reset_index(drop=True)

    size_id_col = "record_id" if "record_id" in memberships.columns else "compound_id"
    counts = (
        memberships.groupby("series_id", dropna=False)[size_id_col]
        .nunique()
        .to_dict()
    )
    memberships["series_size"] = memberships["series_id"].map(counts)
    return memberships[
        memberships["series_size"] >= int(min_series_points)
    ].reset_index(drop=True)


def saod3_aggregate_series_points(memberships):
    if memberships.empty:
        return pd.DataFrame()

    rows = []
    ordered = memberships[memberships["series_coordinate"].notna()].copy()
    optional_defaults = {
        "coordinate_metric_role": "",
        "experimental_context_id": "CTX_DEFAULT",
        "record_id": "",
    }
    for col, default in optional_defaults.items():
        if col not in ordered.columns:
            ordered[col] = default
    for keys, group in ordered.groupby(
        [
            "series_source",
            "series_id",
            "series_domain",
            "coordinate_type",
            "series_coordinate",
        ],
        dropna=False,
    ):
        values = pd.to_numeric(
            group["property_value"], errors="coerce"
        ).dropna()
        if values.empty:
            continue
        rows.append({
            "series_source": keys[0],
            "series_id": keys[1],
            "series_domain": keys[2],
            "coordinate_type": keys[3],
            "series_coordinate": float(keys[4]),
            "series_complexity": float(
                pd.to_numeric(
                    group["series_complexity"], errors="coerce"
                ).median()
            ),
            "property_value": float(values.median()),
            "property_spread": float(values.max() - values.min()),
            "n_replicates": int(len(values)),
            "coordinate_metric_role": str(group["coordinate_metric_role"].iloc[0]),
            "experimental_context_id": str(group["experimental_context_id"].iloc[0]),
            "record_ids": "; ".join(
                sorted(set(group["record_id"].astype(str)))
            ),
            "compound_ids": "; ".join(
                sorted(set(group["compound_id"].astype(str)))
            ),
            "names": "; ".join(
                sorted(set(group["name"].astype(str)))
            ),
            "smiles": "; ".join(
                sorted(set(group["canonical_smiles"].astype(str)))
            ),
        })
    return pd.DataFrame(rows)


def saod3_series_quality_summary(memberships, own_series_summary, min_series_points=3):
    if memberships is None or memberships.empty:
        metrics = {
            "series_total": 0,
            "raw_candidate_series": 0,
            "deduplicated_series": 0,
            "independent_scaffold_families": 0,
            "full_series": 0,
            "small_series": 0,
            "secondary_series": 0,
            "overlapping_series": 0,
            "low_confidence_series": 0,
            "consistent_series": 0,
            "consistent_series_compounds": 0,
            "median_series_size": np.nan,
            "weighted_consistent_coverage": np.nan,
        }
        return pd.DataFrame([metrics]), pd.DataFrame()

    optional_defaults = {
        "scaffold_id": "",
        "scaffold_confidence": "",
        "cut_quality": "",
        "coordinate_metric_role": "",
        "experimental_context_id": "CTX_DEFAULT",
        "primary_membership": False,
        "secondary_membership": False,
        "ambiguity_score": 0.0,
    }
    memberships = memberships.copy()
    for col, default in optional_defaults.items():
        if col not in memberships.columns:
            memberships[col] = default

    per_series = (
        memberships.groupby("series_id", dropna=False)
        .agg(
            series_source=("series_source", "first"),
            series_domain=("series_domain", "first"),
            series_size=("record_id", "nunique"),
            scaffold_id=("scaffold_id", "first"),
            experimental_context_id=("experimental_context_id", "first"),
            coordinate_metric_role=("coordinate_metric_role", "first"),
            scaffold_confidence=("scaffold_confidence", "first"),
            cut_quality=("cut_quality", "first"),
            primary_membership=("primary_membership", "max"),
            secondary_membership=("secondary_membership", "max"),
            max_ambiguity_score=("ambiguity_score", "max"),
        )
        .reset_index()
    )
    raw_candidate_value = pd.to_numeric(
        memberships.get("raw_candidate_series_count", pd.Series(dtype=float)),
        errors="coerce",
    ).max()
    raw_candidate_series = (
        int(raw_candidate_value) if pd.notna(raw_candidate_value) else 0
    )
    deduplicated_value = pd.to_numeric(
        memberships.get("deduplicated_series_count", pd.Series(dtype=float)),
        errors="coerce",
    ).max()
    deduplicated_series = (
        int(deduplicated_value) if pd.notna(deduplicated_value) else len(per_series)
    )
    if "scaffold_id" in memberships.columns:
        scaffold_values = memberships["scaffold_id"].fillna("").astype(str)
        independent_scaffold_families = int(
            scaffold_values[scaffold_values.ne("")].nunique(dropna=True)
        )
    else:
        independent_scaffold_families = 0
    compound_series_counts = (
        memberships.groupby("compound_id", dropna=False)["series_id"]
        .nunique()
        .to_dict()
    )
    per_series["overlapping_compounds"] = per_series["series_id"].map(
        lambda sid: int(
            memberships.loc[
                memberships["series_id"].eq(sid),
                "compound_id",
            ].drop_duplicates().map(compound_series_counts).gt(1).sum()
        )
    )
    per_series["series_role"] = np.where(
        per_series["series_source"].astype(str).eq("graph_profile")
        | _saod_coerce_boolean_series(
            per_series.get("primary_membership", False)
        ),
        "primary",
        "secondary",
    )
    per_series["size_class"] = np.where(
        per_series["series_size"] >= int(min_series_points),
        "full",
        "small",
    )

    if own_series_summary is not None and not own_series_summary.empty:
        own_cols = [
            "series_id",
            "internally_consistent",
            "own_series_confidence_level",
            "n_points",
        ]
        own_cols = [col for col in own_cols if col in own_series_summary.columns]
        per_series = per_series.merge(
            own_series_summary[own_cols].drop_duplicates("series_id"),
            on="series_id",
            how="left",
        )
    else:
        per_series["internally_consistent"] = False
        per_series["own_series_confidence_level"] = ""
        per_series["n_points"] = np.nan

    per_series["internally_consistent"] = _saod_coerce_boolean_series(
        per_series["internally_consistent"]
    )
    confidence = per_series["own_series_confidence_level"].fillna("").astype(str)
    per_series["low_confidence"] = (
        confidence.isin(["", "low_short_series", "not_consistent"])
        | per_series["size_class"].eq("small")
    )
    per_series["overlapping"] = per_series["overlapping_compounds"] > 0

    consistent_series = per_series[per_series["internally_consistent"]].copy()
    consistent_compounds = set()
    for sid in consistent_series["series_id"].astype(str):
        consistent_compounds.update(
            memberships.loc[
                memberships["series_id"].astype(str).eq(sid),
                "compound_id",
            ].astype(str)
        )
    all_compounds = set(memberships["compound_id"].astype(str))
    weighted_coverage = (
        len(consistent_compounds) / len(all_compounds)
        if all_compounds
        else np.nan
    )

    metrics = {
        "series_total": int(len(per_series)),
        "raw_candidate_series": raw_candidate_series,
        "deduplicated_series": deduplicated_series,
        "independent_scaffold_families": independent_scaffold_families,
        "full_series": int(per_series["size_class"].eq("full").sum()),
        "small_series": int(per_series["size_class"].eq("small").sum()),
        "secondary_series": int(per_series["series_role"].eq("secondary").sum()),
        "overlapping_series": int(per_series["overlapping"].sum()),
        "low_confidence_series": int(per_series["low_confidence"].sum()),
        "consistent_series": int(per_series["internally_consistent"].sum()),
        "consistent_series_compounds": int(len(consistent_compounds)),
        "median_series_size": (
            float(per_series["series_size"].median())
            if not per_series.empty
            else np.nan
        ),
        "weighted_consistent_coverage": float(weighted_coverage),
    }
    return pd.DataFrame([metrics]), per_series


def saod3_local_expected(x, y, index):
    if len(y) < 3:
        return np.nan, "insufficient"
    if 0 < index < len(y) - 1:
        x0, x1 = x[index - 1], x[index + 1]
        y0, y1 = y[index - 1], y[index + 1]
        if abs(x1 - x0) < 1e-12:
            return np.nan, "insufficient"
        expected = y0 + (x[index] - x0) * (y1 - y0) / (x1 - x0)
        return float(expected), "interpolation"
    mask = np.ones(len(y), dtype=bool)
    mask[index] = False
    x_train = np.asarray(x, dtype=float)[mask]
    y_train = np.asarray(y, dtype=float)[mask]
    if len(y_train) >= 3 and len(np.unique(x_train)) >= 2:
        try:
            model = TheilSenRegressor(random_state=0)
            model.fit(x_train.reshape(-1, 1), y_train)
            return float(model.predict(np.asarray([[float(x[index])]]))[0]), "theil_sen_endpoint_extrapolation"
        except Exception:
            pass
    if index == 0:
        x0, x1 = x[1], x[2]
        y0, y1 = y[1], y[2]
    else:
        x0, x1 = x[-3], x[-2]
        y0, y1 = y[-3], y[-2]
    if abs(x1 - x0) < 1e-12:
        return np.nan, "insufficient"
    expected = y0 + (x[index] - x0) * (y1 - y0) / (x1 - x0)
    return float(expected), "two_neighbor_endpoint_extrapolation"


def saod3_linear_fit_values(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(np.unique(x)) < 2:
        return np.repeat(np.nanmedian(y), len(y)), 0.0, float(np.nanmedian(y))
    try:
        slope, intercept = np.polyfit(x, y, 1)
        return slope * x + intercept, float(slope), float(intercept)
    except Exception:
        return np.repeat(np.nanmedian(y), len(y)), 0.0, float(np.nanmedian(y))


def saod3_leave_one_out_expected(x, y, index):
    if len(y) < 4:
        return np.nan, "insufficient_loo"
    mask = np.ones(len(y), dtype=bool)
    mask[index] = False
    if len(np.unique(np.asarray(x)[mask])) < 2:
        return np.nan, "insufficient_loo"
    _, slope, intercept = saod3_linear_fit_values(np.asarray(x)[mask], np.asarray(y)[mask])
    return float(slope * x[index] + intercept), "linear_loo"


def saod3_series_influence_diagnostics(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    fitted, slope, _ = saod3_linear_fit_values(x, y)
    residuals = y - fitted
    sse = float(np.nansum(residuals ** 2))
    mse = sse / max(n - 2, 1)
    x_centered = x - float(np.nanmean(x))
    sxx = float(np.nansum(x_centered ** 2))
    leverage = (
        (1.0 / n) + (x_centered ** 2 / sxx)
        if n > 0 and sxx > 1e-12
        else np.zeros(n, dtype=float)
    )
    cooks = np.zeros(n, dtype=float)
    if mse > 1e-12:
        denom = 2.0 * mse * np.maximum((1.0 - leverage) ** 2, 1e-12)
        cooks = (residuals ** 2 / denom) * leverage
    base_mad = float(np.nanmedian(np.abs(residuals - np.nanmedian(residuals))))
    rows = []
    for index in range(n):
        mask = np.ones(n, dtype=bool)
        mask[index] = False
        _, loo_slope, _ = saod3_linear_fit_values(x[mask], y[mask])
        loo_fitted, _, _ = saod3_linear_fit_values(x[mask], y[mask])
        loo_resid = y[mask] - loo_fitted
        loo_mad = float(np.nanmedian(np.abs(loo_resid - np.nanmedian(loo_resid))))
        rows.append({
            "linear_leverage": float(leverage[index]) if index < len(leverage) else np.nan,
            "cooks_distance": float(cooks[index]) if index < len(cooks) else np.nan,
            "slope_full_series": slope,
            "slope_without_point": loo_slope,
            "slope_change_without_point": float(loo_slope - slope),
            "mad_full_series": base_mad,
            "mad_without_point": loo_mad,
            "mad_change_without_point": float(loo_mad - base_mad),
        })
    return rows


def saod3_analyze_own_series(series_points, min_points=3):
    summaries = []
    details = []
    edges = []
    if series_points.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for series_id, group in series_points.groupby("series_id", dropna=False):
        group = group.sort_values("series_coordinate").copy()
        duplicate_coordinate_aggregation = False
        duplicate_coordinate_values = []
        if group["series_coordinate"].duplicated(keep=False).any():
            duplicate_coordinate_aggregation = True
            duplicate_coordinate_values = (
                group.loc[
                    group["series_coordinate"].duplicated(keep=False),
                    "series_coordinate",
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            group["_compound_ids_text"] = group.get("compound_ids", "").astype(str)
            agg_map = {col: "first" for col in group.columns if col != "series_coordinate"}
            agg_map["property_value"] = "mean"
            agg_map["_compound_ids_text"] = (
                lambda values: ";".join(
                    sorted({str(value) for value in values if str(value).strip()})
                )
            )
            group = (
                group
                .groupby("series_coordinate", as_index=False, sort=True, dropna=False)
                .agg(agg_map)
            )
            if "compound_ids" in group.columns:
                group["compound_ids"] = group["_compound_ids_text"]
            group = group.drop(columns=["_compound_ids_text"], errors="ignore")
        x = group["series_coordinate"].astype(float).values
        y = group["property_value"].astype(float).values
        if len(np.unique(x)) < min_points:
            continue

        step = np.diff(x)
        delta = np.diff(y)
        valid_step = np.abs(step) > 1e-12
        if not np.any(valid_step):
            continue
        normalized_delta = delta[valid_step] / step[valid_step]
        normalized_delta_full = np.full(len(step), np.nan, dtype=float)
        normalized_delta_full[valid_step] = normalized_delta
        median_step = float(np.nanmedian(normalized_delta))
        delta_sigma = max(
            saod2_robust_sigma(normalized_delta - median_step),
            max(float(np.ptp(y)) * 0.02, 1e-6),
        )
        delta_scores_valid = np.abs(
            normalized_delta - median_step
        ) / delta_sigma
        delta_scores = np.full(len(step), np.nan, dtype=float)
        delta_scores[valid_step] = delta_scores_valid
        signs = np.sign(normalized_delta)
        main_sign = np.sign(median_step)
        flat_series = bool(np.all(np.abs(normalized_delta) <= delta_sigma))
        if flat_series:
            sign_consistency = 1.0
            trend_class = "stable_flat_series"
        else:
            sign_consistency = (
                float(np.mean(signs[signs != 0] == main_sign))
                if np.any(signs != 0) and main_sign != 0
                else 0.0
            )
            trend_class = "directional_series"

        residual_values = []
        local_rows = []
        influence_rows = saod3_series_influence_diagnostics(x, y)
        for index, (_, row) in enumerate(group.iterrows()):
            expected, method = saod3_local_expected(x, y, index)
            loo_expected, loo_method = saod3_leave_one_out_expected(x, y, index)
            residual = y[index] - expected if np.isfinite(expected) else np.nan
            loo_residual = (
                y[index] - loo_expected
                if np.isfinite(loo_expected)
                else np.nan
            )
            residual_values.append(residual)
            local_rows.append({
                **row.to_dict(),
                "expected_from_own_series": expected,
                "own_series_residual": residual,
                "loo_expected_from_own_series": loo_expected,
                "loo_own_series_residual": loo_residual,
                "loo_prediction_method": loo_method,
                "prediction_method": method,
                **influence_rows[index],
                "is_series_endpoint": bool(
                    index == 0 or index == len(group) - 1
                ),
            })

        finite_residuals = np.asarray(
            [v for v in residual_values if np.isfinite(v)]
        )
        residual_sigma = max(
            saod2_robust_sigma(finite_residuals),
            max(float(np.ptp(y)) * 0.02, 1e-6),
        )
        broken_points = 0
        max_score = 0.0

        for point_index, row in enumerate(local_rows):
            residual = row["own_series_residual"]
            residual_score = (
                abs(residual) / residual_sigma
                if (
                    np.isfinite(residual)
                    and not row["is_series_endpoint"]
                )
                else np.nan
            )
            endpoint_residual = row.get("loo_own_series_residual", np.nan)
            endpoint_residual_score = (
                abs(endpoint_residual) / residual_sigma
                if (
                    row["is_series_endpoint"]
                    and np.isfinite(endpoint_residual)
                    and len(group) >= 4
                )
                else np.nan
            )
            adjacent_scores = []
            if point_index > 0:
                adjacent_scores.append(delta_scores[point_index - 1])
            if point_index < len(delta_scores):
                adjacent_scores.append(delta_scores[point_index])
            two_sided_break = (
                len(adjacent_scores) == 2
                and all(score >= 3 for score in adjacent_scores)
            )
            endpoint_break = (
                row["is_series_endpoint"]
                and len(adjacent_scores) == 1
                and np.isfinite(adjacent_scores[0])
                and adjacent_scores[0] >= 3
            )
            score = max(
                [value for value in [residual_score, endpoint_residual_score, *adjacent_scores]
                 if np.isfinite(value)],
                default=np.nan,
            )
            row["endpoint_residual_score"] = endpoint_residual_score
            row["endpoint_step_break"] = bool(endpoint_break)
            row["own_series_score"] = score
            row["rule_vs_point_diagnosis"] = (
                "series_model_may_be_inadequate"
                if sign_consistency < 0.75
                else (
                    "point_deviates_from_stable_series"
                    if np.isfinite(score) and score >= 3
                    else "no_individual_point_alarm"
                )
            )
            row["own_series_status"] = (
                "нарушает собственный ряд"
                if (
                    (np.isfinite(residual_score) and residual_score >= 3)
                    or (np.isfinite(endpoint_residual_score) and endpoint_residual_score >= 3)
                    or two_sided_break
                    or endpoint_break
                )
                else "согласовано"
            )
            broken_points += int(
                row["own_series_status"] != "согласовано"
            )
            if np.isfinite(score):
                max_score = max(max_score, float(score))
            details.append(row)

        for index in range(1, len(group)):
            score = delta_scores[index - 1]
            edges.append({
                "series_id": series_id,
                "series_domain": group.iloc[0]["series_domain"],
                "coordinate_from": x[index - 1],
                "coordinate_to": x[index],
                "compound_from_ids": group.iloc[index - 1]["compound_ids"],
                "compound_to_ids": group.iloc[index]["compound_ids"],
                "property_from": y[index - 1],
                "property_to": y[index],
                "delta_property": delta[index - 1],
                "normalized_delta": normalized_delta_full[index - 1],
                "expected_normalized_delta": median_step,
                "delta_score": score,
                "edge_status": (
                    "нарушение шага ряда" if score >= 3 else "согласовано"
                ),
            })

        coordinate_type_value = str(group.iloc[0]["coordinate_type"])
        coordinate_metric_role = str(group.iloc[0].get("coordinate_metric_role", ""))
        heuristic_coordinate = (
            coordinate_type_value.endswith("_heuristic")
            or coordinate_metric_role == "ordinal_category_not_metric_distance"
        )
        internally_consistent = (
            len(group) >= min_points
            and sign_consistency >= 0.75
            and broken_points == 0
            and not heuristic_coordinate
        )
        if len(group) >= 10 and internally_consistent:
            confidence_level = "high"
        elif len(group) >= 5 and internally_consistent:
            confidence_level = "moderate"
        elif internally_consistent:
            confidence_level = "low_short_series"
        else:
            confidence_level = "not_consistent"
        summaries.append({
            "series_source": group.iloc[0]["series_source"],
            "series_id": series_id,
            "series_domain": group.iloc[0]["series_domain"],
            "coordinate_type": group.iloc[0]["coordinate_type"],
            "coordinate_metric_role": coordinate_metric_role,
            "heuristic_coordinate_axis": bool(heuristic_coordinate),
            "duplicate_coordinate_aggregation": bool(duplicate_coordinate_aggregation),
            "duplicate_coordinate_values": ";".join(duplicate_coordinate_values),
            "coordinate_semantics": (
                "heuristic ordinal coordinate; not treated as universal linear SAR axis"
                if heuristic_coordinate
                else (
                    "duplicate coordinates were aggregated; coordinate is not a unique structural context"
                    if duplicate_coordinate_aggregation
                    else "numeric series coordinate"
                )
            ),
            "series_complexity": float(
                group["series_complexity"].median()
            ),
            "n_points": int(len(group)),
            "coordinate_min": float(np.min(x)),
            "coordinate_max": float(np.max(x)),
            "median_property_step": median_step,
            "step_sign_consistency": sign_consistency,
            "series_trend_class": trend_class,
            "broken_own_points": broken_points,
            "max_own_series_score": max_score,
            "internally_consistent": internally_consistent,
            "own_series_confidence_level": confidence_level,
            "own_series_residual_sigma": residual_sigma,
            "own_series_delta_sigma": delta_sigma,
            "own_series_criterion": "stable_flat_series or sign_consistency>=0.75; endpoints use one-sided step/extrapolation checks",
            "loo_diagnostic_available": bool(len(group) >= 4),
            "internal_status": (
                "внутренне согласованный ряд"
                if internally_consistent
                else "есть внутренние нарушения"
            ),
        })

    return (
        pd.DataFrame(summaries),
        pd.DataFrame(details),
        pd.DataFrame(edges),
    )


def saod3_compare_ordered_series(candidate, reference, min_points=3):
    merged = candidate.merge(
        reference,
        on="series_coordinate",
        suffixes=("_candidate", "_reference"),
    ).sort_values("series_coordinate")
    if len(merged) < min_points:
        return None, pd.DataFrame()

    x = merged["series_coordinate"].astype(float).values
    delta = (
        merged["property_value_candidate"].astype(float).values
        - merged["property_value_reference"].astype(float).values
    )
    expected, method = saod2_fit_edge_trend(x, delta)
    residual = delta - expected
    sigma = max(
        saod2_robust_sigma(residual),
        max(float(np.ptp(delta)) * 0.05, 1e-6),
    )
    scores = np.abs(residual) / sigma
    broken = scores >= 3

    detail = pd.DataFrame({
        "series_coordinate": x,
        "candidate_series_id": merged["series_id_candidate"],
        "reference_series_id": merged["series_id_reference"],
        "candidate_compound_ids": merged["compound_ids_candidate"],
        "reference_compound_ids": merged["compound_ids_reference"],
        "candidate_names": merged["names_candidate"],
        "reference_names": merged["names_reference"],
        "candidate_value": merged["property_value_candidate"],
        "reference_value": merged["property_value_reference"],
        "delta_candidate_minus_reference": delta,
        "expected_delta": expected,
        "delta_residual": residual,
        "combined_score": scores,
        "comparison_status": np.where(
            broken, "нарушает референтное сравнение", "согласовано"
        ),
    })
    summary = {
        "candidate_series_id": merged["series_id_candidate"].iloc[0],
        "reference_series_id": merged["series_id_reference"].iloc[0],
        "n_common_points": int(len(merged)),
        "trend_method": method,
        "median_delta": float(np.median(delta)),
        "max_comparison_score": float(np.max(scores)),
        "broken_points": int(np.sum(broken)),
        "comparison_passed": bool(np.sum(broken) == 0),
    }
    return summary, detail


def saod3_build_reference_hierarchy(series_points, own_summary, min_points=3):
    hierarchy = []
    comparisons = []
    detail_tables = []
    if series_points.empty or own_summary.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    own_map = {
        row["series_id"]: row.to_dict()
        for _, row in own_summary.iterrows()
    }
    for domain, domain_points in series_points.groupby("series_domain", dropna=False):
        candidates = []
        for series_id, group in domain_points.groupby("series_id", dropna=False):
            info = own_map.get(series_id)
            if info:
                candidates.append(info)
        candidates.sort(key=lambda item: (
            not bool(item["internally_consistent"]),
            float(item["series_complexity"]),
            -int(item["n_points"]),
            item["series_id"],
        ))

        references = []
        for info in candidates:
            series_id = info["series_id"]
            if not bool(info["internally_consistent"]):
                hierarchy.append({
                    "series_domain": domain,
                    "series_id": series_id,
                    "reference_level": np.nan,
                    "reference_status": "отклонён по собственному ряду",
                    "references_tested": 0,
                    "references_passed": 0,
                    "reference_series": "",
                })
                continue

            if not references:
                references.append(series_id)
                hierarchy.append({
                    "series_domain": domain,
                    "series_id": series_id,
                    "reference_level": 1,
                    "reference_status": "базовый референтный ряд",
                    "references_tested": 0,
                    "references_passed": 0,
                    "reference_series": "",
                })
                continue

            candidate_points = domain_points[
                domain_points["series_id"] == series_id
            ]
            tested = passed = 0
            tested_ids = []
            for reference_id in references:
                reference_points = domain_points[
                    domain_points["series_id"] == reference_id
                ]
                summary, detail = saod3_compare_ordered_series(
                    candidate_points, reference_points, min_points
                )
                if summary is None:
                    continue
                tested += 1
                passed += int(summary["comparison_passed"])
                tested_ids.append(reference_id)
                comparisons.append(summary)
                detail_tables.append(detail)

            confirmed = tested > 0 and passed >= max(1, int(np.ceil(tested / 2)))
            if confirmed:
                references.append(series_id)
                status = "подтверждённый референтный ряд"
                level = len(references)
            elif tested == 0:
                status = "нет достаточного перекрытия с референтами"
                level = np.nan
            else:
                status = "не подтверждён референтными рядами"
                level = np.nan

            hierarchy.append({
                "series_domain": domain,
                "series_id": series_id,
                "reference_level": level,
                "reference_status": status,
                "references_tested": tested,
                "references_passed": passed,
                "reference_series": "; ".join(tested_ids),
            })

    details = (
        pd.concat(detail_tables, ignore_index=True)
        if detail_tables else pd.DataFrame()
    )
    return pd.DataFrame(hierarchy), pd.DataFrame(comparisons), details


def saod3_build_transformation_table(memberships):
    """
    Repeated R-group transformations provide cross-series checks when the
    variable fragments have no chemically defensible scalar ordering.
    """
    rows = []
    scaffold = memberships[
        memberships["series_source"] == "automatic_scaffold"
    ].copy()
    if scaffold.empty:
        return pd.DataFrame()

    for series_id, group in scaffold.groupby("series_id", dropna=False):
        records = group.drop_duplicates(
            ["compound_id", "variable_smiles"]
        ).to_dict("records")
        for a, b in itertools.combinations(records, 2):
            va, vb = a["variable_smiles"], b["variable_smiles"]
            if va == vb:
                continue
            if va > vb:
                a, b = b, a
                va, vb = vb, va
            rows.append({
                "transformation_id": f"{va}>>{vb}",
                "series_id": series_id,
                "core_smiles": a["core_smiles"],
                "variable_a": va,
                "variable_b": vb,
                "compound_a_id": a["compound_id"],
                "compound_b_id": b["compound_id"],
                "name_a": a["name"],
                "name_b": b["name"],
                "value_a": a["property_value"],
                "value_b": b["property_value"],
                "delta_a_minus_b": (
                    float(a["property_value"]) - float(b["property_value"])
                ),
            })
    return pd.DataFrame(rows)


def saod3_analyze_transformations(transformations, min_contexts=3):
    rules = []
    details = []
    if transformations.empty:
        return pd.DataFrame(), pd.DataFrame()

    for transformation_id, group in transformations.groupby(
        "transformation_id",
        dropna=False,
    ):
        contexts = group["core_smiles"].nunique()
        if contexts < min_contexts:
            continue
        delta = group["delta_a_minus_b"].astype(float).values
        median = float(np.median(delta))
        sigma = max(
            saod2_robust_sigma(delta - median),
            max(float(np.ptp(delta)) * 0.05, 1e-6),
        )
        scores = np.abs(delta - median) / sigma
        for (_, row), score in zip(group.iterrows(), scores):
            details.append({
                **row.to_dict(),
                "expected_delta": median,
                "transformation_score": float(score),
                "transformation_status": (
                    "нарушает повторяемую трансформацию"
                    if score >= 3 else "согласовано"
                ),
            })
        rules.append({
            "transformation_id": transformation_id,
            "n_contexts": int(contexts),
            "n_pairs": int(len(group)),
            "median_delta": median,
            "max_transformation_score": float(np.max(scores)),
            "broken_contexts": int(np.sum(scores >= 3)),
            "can_be_used_for_checking": True,
        })
    return pd.DataFrame(rules), pd.DataFrame(details)


def saod3_build_hierarchical_suspicion(
    own_details,
    reference_details,
    transformation_details,
    threshold_config=None,
):
    threshold_config = saod2_get_threshold_config(
        threshold_config.get("mode", "standard") if isinstance(threshold_config, dict) else "standard",
        threshold_config if isinstance(threshold_config, dict) else None,
    )
    evidence = defaultdict(lambda: {
        "own_series_breaks": 0,
        "reference_breaks": 0,
        "transformation_breaks": 0,
        "max_hierarchical_score": 0.0,
        "hierarchical_evidence": [],
    })

    def add(compound_ids, kind, score, text):
        for compound_id in str(compound_ids).split(";"):
            compound_id = compound_id.strip()
            if not compound_id:
                continue
            evidence[compound_id][kind] += 1
            if np.isfinite(score):
                evidence[compound_id]["max_hierarchical_score"] = max(
                    evidence[compound_id]["max_hierarchical_score"],
                    float(score),
                )
            evidence[compound_id]["hierarchical_evidence"].append(text)

    if not own_details.empty:
        for _, row in own_details[
            own_details["own_series_status"] == "нарушает собственный ряд"
        ].iterrows():
            add(
                row["compound_ids"], "own_series_breaks",
                row["own_series_score"],
                f"собственный ряд {row['series_id']}",
            )

    if not reference_details.empty:
        for _, row in reference_details[
            reference_details["comparison_status"]
            == "нарушает референтное сравнение"
        ].iterrows():
            add(
                row["candidate_compound_ids"], "reference_breaks",
                row["combined_score"],
                (
                    f"{row['candidate_series_id']} относительно "
                    f"{row['reference_series_id']}"
                ),
            )

    if not transformation_details.empty:
        for _, row in transformation_details[
            transformation_details["transformation_status"]
            == "нарушает повторяемую трансформацию"
        ].iterrows():
            # A transformation alone cannot identify which side is wrong.
            for column in ["compound_a_id", "compound_b_id"]:
                add(
                    row[column], "transformation_breaks",
                    row["transformation_score"],
                    f"трансформация {row['transformation_id']}",
                )

    rows = []
    for compound_id, item in evidence.items():
        independent_types = sum([
            item["own_series_breaks"] > 0,
            item["reference_breaks"] > 0,
            item["transformation_breaks"] > 0,
        ])
        total = (
            item["own_series_breaks"]
            + item["reference_breaks"]
            + item["transformation_breaks"]
        )
        if independent_types >= threshold_config["hierarchical_critical_independent_types"] or (
            item["own_series_breaks"] > 0
            and item["reference_breaks"] >= threshold_config["hierarchical_critical_reference_breaks"]
        ):
            status = SAOD_STATUS_PRIORITY_REVIEW
        elif (
            independent_types >= threshold_config["hierarchical_strong_independent_types"]
            or total >= threshold_config["hierarchical_strong_total_breaks"]
        ):
            status = SAOD_STATUS_STRUCTURAL_MISMATCH
        else:
            status = SAOD_STATUS_NEEDS_CHECK
        rows.append({
            "compound_id": compound_id,
            **item,
            "hierarchical_breaks_total": total,
            "independent_evidence_types": independent_types,
            "hierarchical_status": status,
            "hierarchical_evidence": "; ".join(
                sorted(set(item["hierarchical_evidence"]))
            ),
        })
    return pd.DataFrame(rows)


def saod3_build_hierarchical_suspicion(
    own_details,
    reference_details,
    transformation_details,
    threshold_config=None,
):
    threshold_config = saod2_get_threshold_config(
        threshold_config.get("mode", "standard") if isinstance(threshold_config, dict) else "standard",
        threshold_config if isinstance(threshold_config, dict) else None,
    )
    evidence = defaultdict(lambda: {
        "own_series_breaks": 0,
        "reference_breaks": 0,
        "transformation_breaks": 0,
        "max_hierarchical_score": 0.0,
        "median_hierarchical_score": 0.0,
        "max_standardized_deviation": 0.0,
        "median_deviation": 0.0,
        "evidence_strength": 0.0,
        "evidence_count": 0,
        "hierarchical_evidence": [],
        "_scores": [],
        "_lineage": set(),
    })

    def add(compound_ids, kind, score, text, lineage_key=""):
        for compound_id in str(compound_ids).split(";"):
            compound_id = compound_id.strip()
            if not compound_id:
                continue
            evidence[compound_id][kind] += 1
            if np.isfinite(score):
                score = float(score)
                evidence[compound_id]["_scores"].append(score)
                evidence[compound_id]["max_hierarchical_score"] = max(
                    evidence[compound_id]["max_hierarchical_score"],
                    score,
                )
            if lineage_key:
                evidence[compound_id]["_lineage"].add(str(lineage_key))
            evidence[compound_id]["hierarchical_evidence"].append(str(text))

    if not own_details.empty:
        own_breaks = own_details[
            own_details["own_series_status"].astype(str).str.contains("наруш|РЅР°СЂСѓС€", na=False, regex=True)
        ]
        for _, row in own_breaks.iterrows():
            add(
                row.get("compound_ids", ""),
                "own_series_breaks",
                row.get("own_series_score", np.nan),
                f"own_series {row.get('series_id', '')}",
                lineage_key=f"own::{row.get('series_id', '')}::{row.get('compound_ids', '')}",
            )

    if not reference_details.empty:
        reference_breaks = reference_details[
            reference_details["comparison_status"].astype(str).str.contains("наруш|РЅР°СЂСѓС€", na=False, regex=True)
        ]
        for _, row in reference_breaks.iterrows():
            add(
                row.get("candidate_compound_ids", ""),
                "reference_breaks",
                row.get("combined_score", np.nan),
                f"reference {row.get('candidate_series_id', '')} vs {row.get('reference_series_id', '')}",
                lineage_key=(
                    f"reference::{row.get('candidate_series_id', '')}"
                    f"::{row.get('reference_series_id', '')}"
                    f"::{row.get('candidate_compound_ids', '')}"
                ),
            )

    if not transformation_details.empty:
        transformation_breaks = transformation_details[
            transformation_details["transformation_status"].astype(str).str.contains("наруш|РЅР°СЂСѓС€", na=False, regex=True)
        ]
        for _, row in transformation_breaks.iterrows():
            for column in ["compound_a_id", "compound_b_id"]:
                add(
                    row.get(column, ""),
                    "transformation_breaks",
                    row.get("transformation_score", np.nan),
                    f"transformation {row.get('transformation_id', '')}",
                    lineage_key=(
                        f"transformation::{row.get('transformation_id', '')}"
                        f"::{row.get('core_smiles', '')}"
                        f"::{row.get('compound_a_id', '')}-{row.get('compound_b_id', '')}"
                    ),
                )

    rows = []
    for compound_id, item in evidence.items():
        independent_types = sum([
            item["own_series_breaks"] > 0,
            item["reference_breaks"] > 0,
            item["transformation_breaks"] > 0,
        ])
        total = (
            item["own_series_breaks"]
            + item["reference_breaks"]
            + item["transformation_breaks"]
        )
        scores = [float(value) for value in item["_scores"] if np.isfinite(value)]
        item["evidence_count"] = int(total)
        item["independent_evidence_lineages"] = int(len(item["_lineage"]))
        item["max_standardized_deviation"] = float(np.max(scores)) if scores else 0.0
        item["median_deviation"] = float(np.median(scores)) if scores else 0.0
        item["evidence_strength"] = item["max_standardized_deviation"]
        item["median_hierarchical_score"] = item["median_deviation"]

        if independent_types >= threshold_config["hierarchical_critical_independent_types"] or (
            item["own_series_breaks"] > 0
            and item["reference_breaks"] >= threshold_config["hierarchical_critical_reference_breaks"]
        ):
            status = SAOD_STATUS_PRIORITY_REVIEW
        elif (
            independent_types >= threshold_config["hierarchical_strong_independent_types"]
            or total >= threshold_config["hierarchical_strong_total_breaks"]
        ):
            status = SAOD_STATUS_STRUCTURAL_MISMATCH
        else:
            status = SAOD_STATUS_NEEDS_CHECK

        public_item = {
            key: value
            for key, value in item.items()
            if not str(key).startswith("_")
        }
        rows.append({
            "compound_id": compound_id,
            **public_item,
            "hierarchical_breaks_total": total,
            "independent_evidence_types": independent_types,
            "hierarchical_status": status,
            "hierarchical_evidence": "; ".join(
                sorted(set(item["hierarchical_evidence"]))
            ),
        })
    return pd.DataFrame(rows)


def saod3_merge_suspicion(base, hierarchical, processed, checkability):
    identity_columns = [
        "record_id",
        "compound_id",
        "measurement_id",
        "assay_id",
        "name",
        "canonical_smiles",
        "inchikey",
        "molecular_formula",
        "property_value",
    ]
    identity = processed[
        [col for col in identity_columns if col in processed.columns]
    ].copy()
    if "record_id" in identity.columns:
        identity = identity.drop_duplicates("record_id")
    else:
        identity = identity.drop_duplicates()

    if base.empty:
        out = identity.copy()
    else:
        out = identity.merge(base, on="compound_id", how="left", suffixes=("", "_old"))
        merge_conflicts = _saod_conflicting_old_columns(out)
        out.attrs["merge_conflicts"] = merge_conflicts
        out["merge_conflict_count"] = int(sum(item["n_conflicts"] for item in merge_conflicts))
        out["merge_conflict_columns"] = "; ".join(item["column"] for item in merge_conflicts)
    if "merge_conflict_count" not in out.columns:
        out.attrs["merge_conflicts"] = []
        out["merge_conflict_count"] = 0
        out["merge_conflict_columns"] = ""

    if not hierarchical.empty:
        out = out.merge(hierarchical, on="compound_id", how="left")
    if not checkability.empty:
        checkability_columns = [
            "compound_id",
            "overall_checkability",
            "trusted_edges_total",
            "supported_comparisons_count",
            "consistent_comparisons_count",
            "series_count",
            "independent_series_families",
            "independent_structural_contexts",
            "independent_reference_contexts",
            "structural_support",
            "data_noise_evidence",
            "structural_uniqueness",
            "property_inconsistency",
            "model_residual_outlier",
            "data_quality_issue",
        ]
        out = out.merge(
            checkability[[col for col in checkability_columns if col in checkability.columns]],
            on="compound_id",
            how="left",
        )

    def status_priority(status):
        text = str(status or "")
        if text in SAOD_STATUS_PRIORITY:
            return SAOD_STATUS_PRIORITY[text]
        return SAOD_STATUS_UNKNOWN_PRIORITY

    def safe_float(value, default=0.0):
        try:
            if pd.isna(value):
                return float(default)
        except (TypeError, ValueError):
            pass
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def choose_status(row, return_source=False):
        candidates = []
        for column in ["final_status", "hierarchical_status"]:
            value = row.get(column, "")
            if pd.isna(value):
                value = ""
            value = str(value)
            candidates.append((
                status_priority(value),
                SAOD_STATUS_SOURCE_PRIORITY.get(column, 0),
                value,
                column,
            ))
        selected = max(
            candidates,
            key=lambda item: (item[0], item[1], item[2]),
        )
        if selected[2]:
            return selected[3] if return_source else selected[2]
        if (
            row.get("overall_checkability") in [
                "хорошо проверяемое",
                "умеренно проверяемое",
            ]
            or False
        ):
            return "fallback_checkability_no_breaks" if return_source else SAOD_STATUS_UNCHECKABLE
        if safe_float(row.get("supported_comparisons_count", row.get("trusted_edges_total", 0)), default=0.0) > 0:
            return "fallback_supported_no_breaks" if return_source else SAOD_STATUS_UNCHECKABLE
        return "fallback_uncheckable" if return_source else SAOD_STATUS_UNCHECKABLE

    out["final_status_source"] = out.apply(
        lambda row: choose_status(row, return_source=True),
        axis=1,
    )
    out["final_status"] = out.apply(choose_status, axis=1)
    out["recommendation"] = out["final_status"].map({
        SAOD_STATUS_AGREED: "Значение согласовано с доступными химическими рядами.",
        SAOD_STATUS_UNCHECKABLE: "Недостаточно структурных связей для проверки.",
        SAOD_STATUS_NEEDS_CHECK: "Проверить значение, структуру, единицы и источник.",
        SAOD_STATUS_STRUCTURAL_MISMATCH: "Есть сильное структурное несоответствие; это эвристический сигнал для ручной проверки, а не доказанная ошибка.",
        SAOD_STATUS_PRIORITY_REVIEW: "Множественные несогласованности: нужна приоритетная ручная проверка значения, структуры, единиц и источника.",
    })
    return out


def saod3_universal_checkability(processed, memberships, hierarchy):
    rows = []
    membership_counts = (
        memberships.groupby("compound_id", dropna=False)["series_id"].nunique().to_dict()
        if not memberships.empty else {}
    )
    reference_series = set()
    if not hierarchy.empty:
        reference_series = set(
            hierarchy[
                hierarchy["reference_status"].isin([
                    "базовый референтный ряд",
                    "подтверждённый референтный ряд",
                ])
            ]["series_id"]
        )

    for _, row in processed.iterrows():
        compound_id = str(row.get("compound_id", ""))
        member_rows = (
            memberships[memberships["compound_id"] == compound_id]
            if not memberships.empty else pd.DataFrame()
        )
        n_series = int(membership_counts.get(compound_id, 0))
        n_reference = (
            int(member_rows["series_id"].isin(reference_series).sum())
            if not member_rows.empty else 0
        )
        if n_reference >= 2:
            overall = "хорошо проверяемое"
            score = 7
        elif n_reference == 1:
            overall = "умеренно проверяемое"
            score = 5
        elif n_series >= 2:
            overall = "слабо проверяемое"
            score = 3
        else:
            overall = "почти непроверяемое"
            score = 1
        rows.append({
            "compound_id": compound_id,
            "name": row.get("name", ""),
            "canonical_smiles": row.get("canonical_smiles", ""),
            "molecular_formula": row.get("molecular_formula", ""),
            "carbon_count": row.get("carbon_count", np.nan),
            "exact_pattern": row.get("exact_pattern", ""),
            "property_value": row.get("property_value", np.nan),
            "series_size": int(
                member_rows["series_size"].max()
            ) if not member_rows.empty else 0,
            "formula_group_size": 0,
            "raw_edges_total": n_series,
            "trusted_edges_total": n_reference,
            "supported_comparisons_count": n_reference,
            "consistent_comparisons_count": 0,
            "broken_trusted_edges": 0,
            "has_same_pattern_neighbors": n_series > 0,
            "has_formula_isomers": False,
            "has_positional_analogs": n_series > 1,
            "series_checkability": f"найдено рядов: {n_series}",
            "formula_checkability": "универсальная структурная проверка",
            "network_checkability": f"референтных рядов: {n_reference}",
            "overall_checkability": overall,
            "universal_series_checkability": overall,
            "universal_series_checkability_code": (
                "CHECKABILITY_HIGH" if score >= 7 else
                "CHECKABILITY_MODERATE" if score >= 5 else
                "CHECKABILITY_LOW" if score >= 3 else
                "CHECKABILITY_UNCHECKABLE"
            ),
            "legacy_alkane_checkability": "",
            "final_checkability": overall,
            "final_checkability_code": (
                "CHECKABILITY_HIGH" if score >= 7 else
                "CHECKABILITY_MODERATE" if score >= 5 else
                "CHECKABILITY_LOW" if score >= 3 else
                "CHECKABILITY_UNCHECKABLE"
            ),
            "final_checkability_source": "universal_series",
            "checkability_score": score,
            "checkability_level": overall,
            "dataset_noise_risk": (
                "низкий" if score >= 7 else
                "умеренный" if score >= 5 else
                "повышенный" if score >= 3 else "высокий"
            ),
            "checkability_comment": (
                f"Соединение входит в {n_series} автоматически найденных "
                f"рядов; референтных рядов: {n_reference}."
            ),
            "checkability_recommendation": (
                "Использовать найденные собственные и перекрёстные проверки."
                if n_series else
                "Недостаточно структурных аналогов для автоматической проверки."
            ),
        })
    return pd.DataFrame(rows)


def saod3_universal_checkability_independent(processed, memberships, hierarchy):
    rows = []
    membership_counts = (
        memberships.groupby("compound_id", dropna=False)["series_id"].nunique().to_dict()
        if not memberships.empty else {}
    )
    reference_series = set()
    if not hierarchy.empty:
        reference_series = set(
            hierarchy[
                hierarchy["reference_status"].isin([
                    "Р±Р°Р·РѕРІС‹Р№ СЂРµС„РµСЂРµРЅС‚РЅС‹Р№ СЂСЏРґ",
                    "РїРѕРґС‚РІРµСЂР¶РґС‘РЅРЅС‹Р№ СЂРµС„РµСЂРµРЅС‚РЅС‹Р№ СЂСЏРґ",
                ])
            ]["series_id"]
        )

    for _, row in processed.iterrows():
        compound_id = str(row.get("compound_id", ""))
        member_rows = (
            memberships[memberships["compound_id"] == compound_id].copy()
            if not memberships.empty else pd.DataFrame()
        )
        n_series = int(membership_counts.get(compound_id, 0))
        n_reference = (
            int(member_rows["series_id"].isin(reference_series).sum())
            if not member_rows.empty else 0
        )
        independent_families = set()
        independent_contexts = set()
        reference_contexts = set()
        independent_member_sets = set()
        reference_member_sets = set()
        if not member_rows.empty:
            series_member_map = (
                memberships.groupby("series_id", dropna=False)["compound_id"]
                .apply(lambda values: tuple(sorted(set(values.astype(str)))))
                .to_dict()
                if not memberships.empty else {}
            )
            for _, item in member_rows.iterrows():
                series_id = str(item.get("series_id", ""))
                series_domain = str(item.get("series_domain", ""))
                core_smiles = str(item.get("core_smiles", ""))
                member_key = series_member_map.get(series_id, tuple())
                if series_id:
                    independent_families.add(series_id.split("::", 1)[0])
                if member_key:
                    independent_member_sets.add(member_key)
                if series_domain:
                    independent_contexts.add(series_domain)
                if core_smiles:
                    independent_contexts.add(f"core::{core_smiles}")
            ref_rows = member_rows[member_rows["series_id"].isin(reference_series)]
            for _, item in ref_rows.iterrows():
                series_domain = str(item.get("series_domain", ""))
                core_smiles = str(item.get("core_smiles", ""))
                series_id = str(item.get("series_id", ""))
                member_key = series_member_map.get(series_id, tuple())
                if member_key:
                    reference_member_sets.add(member_key)
                if series_domain:
                    reference_contexts.add(series_domain)
                if core_smiles:
                    reference_contexts.add(f"core::{core_smiles}")

        n_independent_families = int(len(independent_families))
        n_independent_contexts = int(len(independent_contexts))
        n_reference_contexts = int(len(reference_contexts))
        n_independent_member_sets = int(len(independent_member_sets))
        n_reference_member_sets = int(len(reference_member_sets))
        effective_reference_contexts = min(
            n_reference_contexts,
            max(n_reference_member_sets, 0),
        )

        if effective_reference_contexts >= 2:
            overall = "С…РѕСЂРѕС€Рѕ РїСЂРѕРІРµСЂСЏРµРјРѕРµ"
            score = 7
        elif effective_reference_contexts == 1:
            overall = "СѓРјРµСЂРµРЅРЅРѕ РїСЂРѕРІРµСЂСЏРµРјРѕРµ"
            score = 5
        elif n_independent_contexts >= 2:
            overall = "СЃР»Р°Р±Рѕ РїСЂРѕРІРµСЂСЏРµРјРѕРµ"
            score = 3
        else:
            overall = "РїРѕС‡С‚Рё РЅРµРїСЂРѕРІРµСЂСЏРµРјРѕРµ"
            score = 1

        structural_support = (
            "high" if score >= 7 else
            "moderate" if score >= 5 else
            "low" if score >= 3 else "very_low"
        )
        overall_code = (
            "CHECKABILITY_HIGH" if score >= 7 else
            "CHECKABILITY_MODERATE" if score >= 5 else
            "CHECKABILITY_LOW" if score >= 3 else
            "CHECKABILITY_UNCHECKABLE"
        )
        structural_uniqueness = (
            "isolated" if n_independent_contexts == 0 else
            "low_context" if n_independent_contexts == 1 else "supported"
        )

        rows.append({
            "compound_id": compound_id,
            "name": row.get("name", ""),
            "canonical_smiles": row.get("canonical_smiles", ""),
            "molecular_formula": row.get("molecular_formula", ""),
            "carbon_count": row.get("carbon_count", np.nan),
            "exact_pattern": row.get("exact_pattern", ""),
            "property_value": row.get("property_value", np.nan),
            "series_size": int(member_rows["series_size"].max()) if not member_rows.empty else 0,
            "formula_group_size": 0,
            "raw_edges_total": n_series,
            "trusted_edges_total": n_reference,
            "supported_comparisons_count": n_reference,
            "consistent_comparisons_count": 0,
            "series_count": n_series,
            "independent_series_families": n_independent_families,
            "independent_structural_contexts": n_independent_contexts,
            "independent_reference_contexts": n_reference_contexts,
            "independent_member_sets": n_independent_member_sets,
            "independent_reference_member_sets": n_reference_member_sets,
            "effective_reference_contexts": effective_reference_contexts,
            "series_deduplication_policy": "series_domain + core_smiles + member_set",
            "broken_trusted_edges": 0,
            "has_same_pattern_neighbors": n_series > 0,
            "has_formula_isomers": False,
            "has_positional_analogs": n_independent_contexts > 1,
            "series_checkability": f"РЅР°Р№РґРµРЅРѕ СЂСЏРґРѕРІ: {n_series}",
            "formula_checkability": "СѓРЅРёРІРµСЂСЃР°Р»СЊРЅР°СЏ СЃС‚СЂСѓРєС‚СѓСЂРЅР°СЏ РїСЂРѕРІРµСЂРєР°",
            "network_checkability": f"РЅРµР·Р°РІРёСЃРёРјС‹С… РєРѕРЅС‚РµРєСЃС‚РѕРІ: {n_reference_contexts}",
            "overall_checkability": overall,
            "universal_series_checkability": overall,
            "universal_series_checkability_code": overall_code,
            "legacy_alkane_checkability": "",
            "final_checkability": overall,
            "final_checkability_code": overall_code,
            "final_checkability_source": "universal_series",
            "checkability_score": score,
            "checkability_level": overall,
            "structural_support": structural_support,
            "data_noise_evidence": "none",
            "dataset_noise_risk": "not_inferred_from_support",
            "structural_uniqueness": structural_uniqueness,
            "property_inconsistency": "not_evaluated",
            "model_residual_outlier": "not_evaluated",
            "data_quality_issue": "none",
            "checkability_comment": (
                f"РЎРѕРµРґРёРЅРµРЅРёРµ РІС…РѕРґРёС‚ РІ {n_series} Р°РІС‚РѕРјР°С‚РёС‡РµСЃРєРё РЅР°Р№РґРµРЅРЅС‹С… "
                f"СЂСЏРґРѕРІ; РЅРµР·Р°РІРёСЃРёРјС‹С… СЃРµРјРµР№СЃС‚РІ: {n_independent_families}; "
                f"РЅРµР·Р°РІРёСЃРёРјС‹С… СЃС‚СЂСѓРєС‚СѓСЂРЅС‹С… РєРѕРЅС‚РµРєСЃС‚РѕРІ: {n_independent_contexts}."
            ),
            "checkability_recommendation": (
                "РСЃРїРѕР»СЊР·РѕРІР°С‚СЊ РЅРµР·Р°РІРёСЃРёРјС‹Рµ СЃС‚СЂСѓРєС‚СѓСЂРЅС‹Рµ РєРѕРЅС‚РµРєСЃС‚С‹ РґР»СЏ РїСЂРѕРІРµСЂРєРё."
                if n_independent_contexts else
                "РЎС‚СЂСѓРєС‚СѓСЂР° РїР»РѕС…Рѕ РїСЂРѕРІРµСЂСЏРµРјР° РІРЅСѓС‚СЂРё СЌС‚РѕРіРѕ РґР°С‚Р°СЃРµС‚Р°; СЌС‚Рѕ РЅРµ РґРѕРєР°Р·Р°С‚РµР»СЊСЃС‚РІРѕ С€СѓРјР° РґР°РЅРЅС‹С…."
            ),
        })
    return pd.DataFrame(rows)


def saod2_inject_artificial_errors(
    input_df,
    smiles_col,
    property_col,
    fraction=0.05,
    error_type="property_increase_percent",
    percent=20.0,
    random_state=42,
):
    """
    Creates a labelled SAOD stress-test dataset by injecting controlled errors.

    The returned labels table uses compound_id so detection quality can be
    measured after SAOD standardisation and aggregation.
    """
    work = input_df.copy().reset_index(drop=True)
    if "compound_id" not in work.columns:
        work["compound_id"] = work.index.astype(str)
    else:
        work["compound_id"] = work["compound_id"].astype(str)

    rng = np.random.default_rng(int(random_state))
    n_rows = len(work)
    if n_rows == 0:
        return work, pd.DataFrame()

    n_select = max(1, int(round(float(fraction) * n_rows)))
    n_select = min(n_select, n_rows)
    selected = rng.choice(np.arange(n_rows), size=n_select, replace=False)
    labels = []

    def add_label(idx, original_value, mutated_value, original_smiles=None, mutated_smiles=None):
        labels.append({
            "row_index": int(idx),
            "compound_id": str(work.loc[idx, "compound_id"]),
            "error_type": error_type,
            "original_value": original_value,
            "mutated_value": mutated_value,
            "original_smiles": original_smiles,
            "mutated_smiles": mutated_smiles,
        })

    if error_type == "property_increase_percent":
        factor = 1.0 + float(percent) / 100.0
        values = pd.to_numeric(work[property_col], errors="coerce")
        for idx in selected:
            old = values.iloc[idx]
            if not np.isfinite(old):
                continue
            new = float(old) * factor
            work.loc[idx, property_col] = new
            add_label(idx, old, new)

    elif error_type == "swap_values":
        if len(selected) >= 2:
            values = work.loc[selected, property_col].to_numpy(copy=True)
            rotated = np.roll(values, 1)
            for idx, old, new in zip(selected, values, rotated):
                work.loc[idx, property_col] = new
                add_label(idx, old, new)

    elif error_type == "replace_smiles":
        smiles_values = work[smiles_col].astype(str).to_numpy(copy=True)
        donors = np.arange(n_rows)
        for idx in selected:
            donor_pool = donors[donors != idx]
            if len(donor_pool) == 0:
                continue
            donor = int(rng.choice(donor_pool))
            old = smiles_values[idx]
            new = smiles_values[donor]
            work.loc[idx, smiles_col] = new
            add_label(idx, work.loc[idx, property_col], work.loc[idx, property_col], old, new)

    elif error_type == "conflicting_duplicate":
        factor = 1.0 + float(percent) / 100.0
        duplicates = []
        values = pd.to_numeric(work[property_col], errors="coerce")
        for idx in selected:
            old = values.iloc[idx]
            if not np.isfinite(old):
                continue
            duplicate = work.loc[idx].copy()
            new = float(old) * factor
            duplicate[property_col] = new
            duplicates.append(duplicate)
            add_label(idx, old, new, work.loc[idx, smiles_col], work.loc[idx, smiles_col])
        if duplicates:
            work = pd.concat([work, pd.DataFrame(duplicates)], ignore_index=True)

    else:
        raise ValueError(f"Unknown artificial SAOD error type: {error_type}")

    return work, pd.DataFrame(labels)


def _saod2_rank_auc(y_true, scores):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    for group_id, count in enumerate(counts):
        if count > 1:
            mask = inverse == group_id
            ranks[mask] = ranks[mask].mean()
    rank_sum_pos = float(ranks[pos].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def saod2_evaluate_artificial_error_detection(
    suspicion,
    injected_labels,
    detected_statuses=None,
):
    if detected_statuses is None:
        detected_statuses = {
            SAOD_STATUS_STRUCTURAL_MISMATCH,
            SAOD_STATUS_PRIORITY_REVIEW,
        }
    if not isinstance(suspicion, pd.DataFrame) or suspicion.empty:
        return pd.DataFrame()
    if not isinstance(injected_labels, pd.DataFrame) or injected_labels.empty:
        return pd.DataFrame()

    scored = suspicion.copy()
    scored["compound_id"] = scored["compound_id"].astype(str)
    injected_ids = set(injected_labels["compound_id"].astype(str))
    status_score = {
        SAOD_STATUS_UNCHECKABLE: 0.0,
        SAOD_STATUS_AGREED: 0.0,
        SAOD_STATUS_NEEDS_CHECK: 1.0,
        SAOD_STATUS_STRUCTURAL_MISMATCH: 2.0,
        SAOD_STATUS_PRIORITY_REVIEW: 3.0,
    }
    scored["_truth"] = scored["compound_id"].isin(injected_ids).astype(int)
    scored["_detected"] = scored["final_status"].isin(detected_statuses).astype(int)
    scored["_score"] = scored["final_status"].map(status_score).fillna(0.0)
    if "max_edge_score" in scored.columns:
        scored["_score"] = scored["_score"] + pd.to_numeric(
            scored["max_edge_score"],
            errors="coerce",
        ).fillna(0.0) / 100.0

    y_true = scored["_truth"].to_numpy(dtype=int)
    y_pred = scored["_detected"].to_numpy(dtype=int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = sensitivity
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    auc = _saod2_rank_auc(y_true, scored["_score"].to_numpy(dtype=float))

    return pd.DataFrame([{
        "injected_compounds": int(len(injected_ids)),
        "evaluated_compounds": int(len(scored)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
        "roc_auc": auc,
    }])


# ------------------------------------------------------------------
# Main runner

def run_saod2_analysis(
    input_df,
    smiles_col,
    property_col,
    min_rule_points=3,
    min_series_points=None,
    min_own_series_points=None,
    min_reference_points=None,
    min_transformation_contexts=None,
    min_legacy_alkane_rule_points=None,
    threshold_mode="standard",
    threshold_config=None,
    trend_method="linear",
    measurement_uncertainty_col=None,
    typical_measurement_uncertainty=None,
    duplicate_key_policy="canonical_smiles",
    duplicate_abs_tolerance=1e-6,
    duplicate_rel_tolerance=1e-6,
):
    """
    Universal SAOD analysis.

    Backward-compatible result keys are preserved. New keys expose automatic
    scaffold series, own-series checks, hierarchical reference validation,
    and repeated R-group transformation checks.
    """
    errors = []
    warnings = []
    threshold_config = saod2_get_threshold_config(threshold_mode, threshold_config)
    base_min_rule_points = int(min_rule_points)
    saod_config = SAODConfig(
        min_series_points=int(
            base_min_rule_points if min_series_points is None else min_series_points
        ),
        min_own_series_points=int(
            base_min_rule_points if min_own_series_points is None else min_own_series_points
        ),
        min_reference_points=int(
            base_min_rule_points if min_reference_points is None else min_reference_points
        ),
        min_transformation_pairs=int(
            base_min_rule_points
            if min_transformation_contexts is None
            else min_transformation_contexts
        ),
        min_independent_contexts=int(
            base_min_rule_points
            if min_transformation_contexts is None
            else min_transformation_contexts
        ),
        min_legacy_alkane_rule_points=int(
            base_min_rule_points
            if min_legacy_alkane_rule_points is None
            else min_legacy_alkane_rule_points
        ),
    )

    try:
        processed = saod2_prepare_structures(
            input_df=input_df,
            smiles_col=smiles_col,
            property_col=property_col,
            duplicate_key_policy=duplicate_key_policy,
            duplicate_abs_tolerance=duplicate_abs_tolerance,
            duplicate_rel_tolerance=duplicate_rel_tolerance,
            duplicate_measurement_uncertainty=typical_measurement_uncertainty,
        )

    except Exception as e:
        failed_df = input_df.copy()
        failed_df = failed_df.reset_index(drop=False).rename(columns={"index": "source_row"})
        if "record_id" not in failed_df.columns:
            failed_df["record_id"] = [f"record_{i + 1:06d}" for i in range(len(failed_df))]
        if "compound_id" not in failed_df.columns:
            failed_df["compound_id"] = failed_df["record_id"].astype(str)
        if "name" not in failed_df.columns:
            failed_df["name"] = ""
        failed_df["input_smiles"] = failed_df.get(smiles_col, "")
        failed_df["canonical_smiles"] = ""
        failed_df["property_value"] = np.nan
        failed_status = saod2_encode_standardization_status({
            "parse": "failed",
            "largest_fragment": "not_run",
            "cleanup": "not_run",
            "uncharging": "not_run",
            "inchikey": "not_run",
            "overall": "failed",
            "code": "STANDARDIZATION_FAILED",
            "details": {"exception_type": type(e).__name__},
        })
        failed_df["structure_status"] = failed_status
        failed_df["structure_status_code"] = "STANDARDIZATION_FAILED"
        failed_df["structure_standardization_overall"] = "failed"
        failed_df["valid_structure"] = False
        failed_df["series_supported"] = False
        failed_df["series_reason"] = ""
        analysis_coverage, analysis_inclusion_table, exclusion_table = (
            saod2_build_analysis_inclusion_tables(
                failed_df,
                pd.DataFrame(),
                pd.DataFrame(),
                min_rule_points=saod_config.min_series_points,
                errors=[str(e)],
            )
        )
        return {
            "status": "failed",
            "failure_stage": "prepare_structures",
            "failed_stage": "PREPARATION",
            "completed_stages": [],
            "algorithm_version": SAOD_VERSION,
            "processed": pd.DataFrame(),
            "checkability": pd.DataFrame(),
            "analysis_coverage": analysis_coverage,
            "analysis_inclusion_table": analysis_inclusion_table,
            "exclusion_table": exclusion_table,
            "raw_edge_table": pd.DataFrame(),
            "edge_table": pd.DataFrame(),
            "rules": pd.DataFrame(),
            "edge_details": pd.DataFrame(),
            "broken_edges": pd.DataFrame(),
            "suspicion": pd.DataFrame(),
            "summary": pd.DataFrame(),
            "series_memberships": pd.DataFrame(),
            "series_points": pd.DataFrame(),
            "series_quality_summary": pd.DataFrame(),
            "series_quality_table": pd.DataFrame(),
            "own_series_summary": pd.DataFrame(),
            "own_series_details": pd.DataFrame(),
            "own_series_edges": pd.DataFrame(),
            "reference_hierarchy": pd.DataFrame(),
            "reference_comparisons": pd.DataFrame(),
            "reference_comparison_details": pd.DataFrame(),
            "transformation_table": pd.DataFrame(),
            "transformation_rules": pd.DataFrame(),
            "transformation_details": pd.DataFrame(),
            "hierarchical_suspicion": pd.DataFrame(),
            "measurement_replicates": pd.DataFrame(),
            "measurement_uncertainty_used": None,
            "threshold_config": threshold_config,
            "saod_config": asdict(saod_config),
            "errors": [str(e)],
            "warnings": warnings
        }

    measurement_uncertainty_used = saod2_measurement_uncertainty_summary(
        processed,
        uncertainty_col=measurement_uncertainty_col,
        typical_uncertainty=typical_measurement_uncertainty,
    )
    if measurement_uncertainty_col and measurement_uncertainty_col in processed.columns:
        processed["measurement_uncertainty"] = pd.to_numeric(
            processed[measurement_uncertainty_col],
            errors="coerce",
        )
    elif measurement_uncertainty_used is not None:
        processed["measurement_uncertainty"] = float(measurement_uncertainty_used)

    measurement_replicates = saod2_replicate_measurement_report(
        processed,
        uncertainty_col=(
            "measurement_uncertainty"
            if "measurement_uncertainty" in processed.columns
            else None
        ),
    )

    invalid_count = int((processed["valid_structure"] == False).sum())

    if invalid_count > 0:
        warnings.append(f"Некорректных SMILES: {invalid_count}")

    missing_property = int(processed["property_value"].isna().sum())

    if missing_property > 0:
        warnings.append(
            f"Пропущенных или нечисловых значений свойства: {missing_property}"
        )

    duplicate_conflicts = int(processed["duplicate_conflict"].sum())

    if duplicate_conflicts > 0:
        warnings.append(f"Конфликтов дубликатов: {duplicate_conflicts}")

    completed_stages = ["PREPARATION"]
    series_memberships = pd.DataFrame()
    series_points = pd.DataFrame()
    own_series_summary = pd.DataFrame()
    own_series_details = pd.DataFrame()
    own_series_edges = pd.DataFrame()
    reference_hierarchy = pd.DataFrame()
    reference_comparisons = pd.DataFrame()
    reference_comparison_details = pd.DataFrame()
    transformation_table = pd.DataFrame()
    transformation_rules = pd.DataFrame()
    transformation_details = pd.DataFrame()
    hierarchical_suspicion = pd.DataFrame()
    series_quality_summary = pd.DataFrame()
    series_quality_table = pd.DataFrame()
    raw_edge_table = pd.DataFrame()
    edge_table = pd.DataFrame()
    rules = pd.DataFrame()
    edge_details = pd.DataFrame()
    broken_edges = pd.DataFrame()
    checkability = pd.DataFrame()
    analysis_coverage = pd.DataFrame()
    analysis_inclusion_table = pd.DataFrame()
    exclusion_table = pd.DataFrame()
    suspicion = pd.DataFrame()
    summary = pd.DataFrame()

    def _partial_result(failed_stage, exc):
        stage_error = f"{failed_stage}: {exc}"
        partial_errors = errors + [stage_error]
        partial_checkability = checkability
        if partial_checkability.empty:
            try:
                partial_checkability = saod3_universal_checkability_independent(
                    processed,
                    series_memberships,
                    reference_hierarchy,
                )
            except Exception:
                partial_checkability = pd.DataFrame()
        try:
            partial_coverage, partial_inclusion, partial_exclusion = (
                saod2_build_analysis_inclusion_tables(
                    processed,
                    series_memberships,
                    partial_checkability,
                    min_rule_points=saod_config.min_series_points,
                    errors=partial_errors,
                )
            )
        except Exception:
            partial_coverage = pd.DataFrame()
            partial_inclusion = pd.DataFrame()
            partial_exclusion = pd.DataFrame()
        return {
            "status": "partial_success",
            "failure_stage": failed_stage,
            "failed_stage": failed_stage,
            "completed_stages": list(completed_stages),
            "algorithm_version": SAOD_VERSION,
            "processed": processed,
            "series_memberships": series_memberships,
            "analysis_coverage": partial_coverage,
            "analysis_inclusion_table": partial_inclusion,
            "exclusion_table": partial_exclusion,
            "series_points": series_points,
            "own_series_summary": own_series_summary,
            "own_series_details": own_series_details,
            "own_series_edges": own_series_edges,
            "reference_hierarchy": reference_hierarchy,
            "reference_comparisons": reference_comparisons,
            "reference_comparison_details": reference_comparison_details,
            "transformation_table": transformation_table,
            "transformation_rules": transformation_rules,
            "transformation_details": transformation_details,
            "series_quality_summary": series_quality_summary,
            "series_quality_table": series_quality_table,
            "hierarchical_suspicion": hierarchical_suspicion,
            "measurement_replicates": measurement_replicates,
            "measurement_uncertainty_used": measurement_uncertainty_used,
            "checkability": partial_checkability,
            "raw_edge_table": raw_edge_table,
            "edge_table": edge_table,
            "rules": rules,
            "edge_details": edge_details,
            "broken_edges": broken_edges,
            "suspicion": suspicion,
            "summary": summary,
            "threshold_config": threshold_config,
            "saod_config": asdict(saod_config),
            "errors": partial_errors,
            "warnings": warnings,
        }

    # Universal automatic series layer.
    try:
        series_memberships = saod3_discover_series_memberships(
            processed,
            min_series_points=saod_config.min_series_points,
        )
        series_points = saod3_aggregate_series_points(series_memberships)
        completed_stages.append("SERIES_DISCOVERY")
    except Exception as exc:
        return _partial_result("SERIES_DISCOVERY", exc)
    try:
        (
            own_series_summary,
            own_series_details,
            own_series_edges,
        ) = saod3_analyze_own_series(
            series_points,
            min_points=saod_config.min_own_series_points,
        )
        series_quality_summary, series_quality_table = saod3_series_quality_summary(
            series_memberships,
            own_series_summary,
            min_series_points=saod_config.min_series_points,
        )
        completed_stages.append("OWN_SERIES")
    except Exception as exc:
        return _partial_result("OWN_SERIES", exc)
    try:
        (
            reference_hierarchy,
            reference_comparisons,
            reference_comparison_details,
        ) = saod3_build_reference_hierarchy(
            series_points,
            own_series_summary,
            min_points=saod_config.min_reference_points,
        )
        completed_stages.append("REFERENCE_HIERARCHY")
    except Exception as exc:
        return _partial_result("REFERENCE_HIERARCHY", exc)
    try:
        transformation_table = saod3_build_transformation_table(
            series_memberships
        )
        (
            transformation_rules,
            transformation_details,
        ) = saod3_analyze_transformations(
            transformation_table,
            min_contexts=saod_config.min_independent_contexts,
        )
        completed_stages.append("TRANSFORMATIONS")
    except Exception as exc:
        return _partial_result("TRANSFORMATIONS", exc)

    try:
        hierarchical_suspicion = saod3_build_hierarchical_suspicion(
            own_series_details,
            reference_comparison_details,
            transformation_details,
            threshold_config=threshold_config,
        )
    except Exception as exc:
        return _partial_result("FINAL_MERGE", exc)

    # Original alkane edge layer remains available and supplies additional
    # evidence for existing reports.
    try:
        raw_edge_table = saod2_build_edge_table(processed)

        edge_table = saod2_aggregate_edge_table_by_formula(
            raw_edge_table=raw_edge_table
        )

        rules, edge_details = saod2_discover_rules(
            edge_table=edge_table,
            min_points=saod_config.min_legacy_alkane_rule_points,
            trend_method=trend_method,
            measurement_uncertainty=measurement_uncertainty_used,
        )

        broken_edges = saod2_broken_edges(edge_details)

        base_suspicion = saod2_compound_suspicion(
            edge_table=edge_table,
            edge_detail_table=edge_details,
            broken_edges=broken_edges,
            threshold_config=threshold_config,
        )
        checkability = saod3_universal_checkability_independent(
            processed,
            series_memberships,
            reference_hierarchy,
        )
        analysis_coverage, analysis_inclusion_table, exclusion_table = (
            saod2_build_analysis_inclusion_tables(
                processed,
                series_memberships,
                checkability,
                min_rule_points=saod_config.min_series_points,
                errors=errors,
            )
        )
        suspicion = saod3_merge_suspicion(
            base_suspicion,
            hierarchical_suspicion,
            processed,
            checkability,
        )

        summary = saod2_dataset_summary(
            processed=processed,
            checkability=checkability,
            rules=rules,
            broken_edges=broken_edges,
            suspicion=suspicion,
            threshold_config=threshold_config,
        )
        completed_stages.append("FINAL_MERGE")
    except Exception as exc:
        return _partial_result("FINAL_MERGE", exc)
    if not analysis_coverage.empty:
        coverage_summary = pd.DataFrame([
            {
                "Показатель": row.get("label", row.get("metric", "")),
                "Значение": row.get("value", 0),
            }
            for _, row in analysis_coverage.iterrows()
            if row.get("metric") in {"loaded", "analyzed", "excluded"}
        ])
        summary = pd.concat(
            [coverage_summary, summary],
            ignore_index=True,
        )
    universal_summary = pd.DataFrame([
        {
            "Показатель": "Автоматически найденных химических рядов",
            "Значение": int(
                series_memberships["series_id"].nunique()
            ) if not series_memberships.empty else 0,
        },
        {
            "Показатель": "Внутренне согласованных рядов",
            "Значение": int(
                own_series_summary["internally_consistent"].sum()
            ) if not own_series_summary.empty else 0,
        },
        {
            "Показатель": "Референтных рядов",
            "Значение": int(
                reference_hierarchy["reference_status"].isin([
                    "базовый референтный ряд",
                    "подтверждённый референтный ряд",
                ]).sum()
            ) if not reference_hierarchy.empty else 0,
        },
        {
            "Показатель": "Повторяемых R-групповых трансформаций",
            "Значение": int(len(transformation_rules)),
        },
    ])
    summary = pd.concat(
        [summary, universal_summary],
        ignore_index=True,
    )
    if not summary.empty and len(summary.columns) >= 2:
        metric_col, value_col = summary.columns[:2]
    else:
        metric_col, value_col = "metric", "value"
    series_quality_metrics = (
        series_quality_summary.iloc[0].to_dict()
        if isinstance(series_quality_summary, pd.DataFrame)
        and not series_quality_summary.empty
        else {}
    )
    series_quality_diagnostics = pd.DataFrame([
        {metric_col: "Сырых рядов-кандидатов", value_col: int(series_quality_metrics.get("raw_candidate_series", 0) or 0)},
        {metric_col: "Рядов после удаления дублей", value_col: int(series_quality_metrics.get("deduplicated_series", 0) or 0)},
        {metric_col: "Независимых семейств каркасов", value_col: int(series_quality_metrics.get("independent_scaffold_families", 0) or 0)},
        {metric_col: "Полноразмерных найденных рядов", value_col: int(series_quality_metrics.get("full_series", 0) or 0)},
        {metric_col: "Малых найденных рядов", value_col: int(series_quality_metrics.get("small_series", 0) or 0)},
        {metric_col: "Вторичных/производных рядов", value_col: int(series_quality_metrics.get("secondary_series", 0) or 0)},
        {metric_col: "Перекрывающихся рядов", value_col: int(series_quality_metrics.get("overlapping_series", 0) or 0)},
        {metric_col: "Низконадёжных рядов", value_col: int(series_quality_metrics.get("low_confidence_series", 0) or 0)},
        {metric_col: "Веществ, покрытых внутренне согласованными рядами", value_col: int(series_quality_metrics.get("consistent_series_compounds", 0) or 0)},
        {metric_col: "Медианный размер найденного ряда", value_col: series_quality_metrics.get("median_series_size", np.nan)},
        {metric_col: "Взвешенное покрытие датасета согласованными рядами", value_col: series_quality_metrics.get("weighted_consistent_coverage", np.nan)},
    ])
    summary = pd.concat(
        [summary, series_quality_diagnostics],
        ignore_index=True,
    )
    diagnostics_summary = pd.DataFrame([
        {
            metric_col: "Запрошенный метод тренда SAOD",
            value_col: trend_method,
        },
        {
            metric_col: "Использованная неопределённость измерения",
            value_col: (
                measurement_uncertainty_used
                if measurement_uncertainty_used is not None
                else "не задана"
            ),
        },
        {
            metric_col: "Повторяющиеся структуры",
            value_col: int(len(measurement_replicates)),
        },
    ])
    summary = pd.concat(
        [summary, diagnostics_summary],
        ignore_index=True,
    )

    has_series_evidence = any(
        not frame.empty
        for frame in [
            series_memberships,
            own_series_summary,
            reference_comparisons,
            transformation_rules,
            rules,
        ]
        if isinstance(frame, pd.DataFrame)
    )

    return {
        "status": "success" if has_series_evidence else "success_no_series",
        "failure_stage": "",
        "failed_stage": "",
        "completed_stages": list(completed_stages),
        "algorithm_version": SAOD_VERSION,
        "processed": processed,
        "series_memberships": series_memberships,
        "analysis_coverage": analysis_coverage,
        "analysis_inclusion_table": analysis_inclusion_table,
        "exclusion_table": exclusion_table,
        "series_points": series_points,
        "own_series_summary": own_series_summary,
        "own_series_details": own_series_details,
        "own_series_edges": own_series_edges,
        "reference_hierarchy": reference_hierarchy,
        "reference_comparisons": reference_comparisons,
        "reference_comparison_details": reference_comparison_details,
        "transformation_table": transformation_table,
        "transformation_rules": transformation_rules,
        "transformation_details": transformation_details,
        "series_quality_summary": series_quality_summary,
        "series_quality_table": series_quality_table,
        "hierarchical_suspicion": hierarchical_suspicion,
        "measurement_replicates": measurement_replicates,
        "measurement_uncertainty_used": measurement_uncertainty_used,
        "checkability": checkability,
        "raw_edge_table": raw_edge_table,
        "edge_table": edge_table,
        "rules": rules,
        "edge_details": edge_details,
        "broken_edges": broken_edges,
        "suspicion": suspicion,
        "summary": summary,
        "threshold_config": threshold_config,
        "saod_config": asdict(saod_config),
        "errors": errors,
        "warnings": warnings
    }
