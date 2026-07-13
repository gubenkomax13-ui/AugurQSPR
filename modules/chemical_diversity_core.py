# -*- coding: utf-8 -*-
"""Chemical diversity diagnostics for a QSPR dataset."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, rdFingerprintGenerator
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.ML.Cluster import Butina
except Exception:  # pragma: no cover - depends on optional local RDKit install
    Chem = None
    DataStructs = None
    AllChem = None
    rdFingerprintGenerator = None
    rdMolStandardize = None
    Butina = None

try:
    from sklearn.cluster import DBSCAN
    from sklearn.manifold import MDS, TSNE
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - depends on optional sklearn install
    DBSCAN = None
    MDS = None
    TSNE = None
    PCA = None
    NearestNeighbors = None
    StandardScaler = None

try:
    from modules.saod2_core import saod2_classify_alkane
except Exception:  # pragma: no cover - SAOD module is optional for this diagnostic
    saod2_classify_alkane = None


CHEMICAL_SPACE_ALGORITHM_VERSION = "chemical_space_1.3"


def coerce_boolean_series(values, default: bool = False) -> pd.Series:
    """Coerce bool-like values without treating non-empty strings as True."""
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


@dataclass
class _MoleculeRecord:
    row_index: int
    label: str
    smiles: str
    original_smiles: str
    fingerprint_smiles: str
    fingerprint_structure_source: str
    mol: object
    fingerprint: object


def _safe_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _invalid_structure_row(row, row_index, smiles, reason_code):
    reason_labels = {
        "empty_smiles": "пустой SMILES",
        "invalid_smiles": "невалидный SMILES",
        "mixture": "смесь",
    }
    return {
        "record_id": row.get("record_id", ""),
        "compound_id": row.get("compound_id", ""),
        "source_row": row.get("source_row", int(row_index)),
        "row_index": int(row_index),
        "SMILES": smiles,
        "status_code": "excluded",
        "status_label": "исключено",
        "reason_code": reason_code,
        "reason": reason_labels.get(reason_code, reason_code),
    }


def _standardized_parent_mol(mol):
    if mol is None or rdMolStandardize is None:
        return mol, "original_smiles"
    try:
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
    except Exception:
        pass
    try:
        mol = rdMolStandardize.Cleanup(mol)
    except Exception:
        pass
    try:
        mol = rdMolStandardize.Uncharger().uncharge(mol)
    except Exception:
        pass
    return mol, "standardized_parent"


def _morgan_fingerprint(mol, radius: int = 2, n_bits: int = 2048):
    if rdFingerprintGenerator is not None:
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=int(radius),
            fpSize=int(n_bits),
        )
        return generator.GetFingerprint(mol)
    if AllChem is not None:
        return AllChem.GetMorganFingerprintAsBitVect(mol, int(radius), nBits=int(n_bits))
    raise RuntimeError("RDKit Morgan fingerprint generator недоступен.")


def canonicalize_smiles(smiles: str) -> str:
    if Chem is None:
        return ""
    smiles = _safe_text(smiles)
    if not smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _build_records(
    data: pd.DataFrame,
    smiles_col: str,
    label_col: Optional[str],
    radius: int,
    n_bits: int,
    fingerprint_structure_source: str = "standardized_parent",
) -> tuple[list[_MoleculeRecord], pd.DataFrame]:
    records: list[_MoleculeRecord] = []
    invalid_rows = []

    if Chem is None or (rdFingerprintGenerator is None and AllChem is None):
        raise RuntimeError("RDKit недоступен: нельзя рассчитать Morgan/Tanimoto диагностику.")

    for row_index, row in data.iterrows():
        smiles = _safe_text(row.get(smiles_col, ""))
        if not smiles:
            invalid_rows.append(_invalid_structure_row(row, row_index, smiles, "empty_smiles"))
            continue

        use_parent = str(fingerprint_structure_source or "standardized_parent") == "standardized_parent"
        if "." in smiles and not use_parent:
            invalid_rows.append(_invalid_structure_row(row, row_index, smiles, "mixture"))
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_rows.append(_invalid_structure_row(row, row_index, smiles, "invalid_smiles"))
            continue
        fingerprint_mol = mol
        structure_source_used = "original_smiles"
        if use_parent:
            fingerprint_mol, structure_source_used = _standardized_parent_mol(mol)
            if fingerprint_mol is None:
                invalid_rows.append(_invalid_structure_row(row, row_index, smiles, "invalid_smiles"))
                continue
        fingerprint_smiles = Chem.MolToSmiles(
            fingerprint_mol,
            canonical=True,
            isomericSmiles=True,
        )

        label = _safe_text(row.get(label_col, "")) if label_col and label_col in data.columns else ""
        if not label:
            label = f"row {int(row_index) + 1}"

        records.append(_MoleculeRecord(
            row_index=int(row_index),
            label=label,
            smiles=fingerprint_smiles,
            original_smiles=smiles,
            fingerprint_smiles=fingerprint_smiles,
            fingerprint_structure_source=structure_source_used,
            mol=fingerprint_mol,
            fingerprint=_morgan_fingerprint(fingerprint_mol, radius=radius, n_bits=n_bits),
        ))

    return records, pd.DataFrame(invalid_rows)


def _push_top_pair(heap, max_size: int, similarity: float, i: int, j: int):
    item = (float(similarity), int(i), int(j))
    if len(heap) < max_size:
        heapq.heappush(heap, item)
    elif similarity > heap[0][0]:
        heapq.heapreplace(heap, item)


def _pairwise_full(records: list[_MoleculeRecord], top_n: int):
    n = len(records)
    sims = []
    top_heap = []
    max_similarity = np.zeros(n, dtype=float)
    nearest_index = np.full(n, -1, dtype=int)

    for i in range(1, n):
        values = DataStructs.BulkTanimotoSimilarity(
            records[i].fingerprint,
            [record.fingerprint for record in records[:i]],
        )
        for j, sim in enumerate(values):
            sim = float(sim)
            sims.append(sim)
            _push_top_pair(top_heap, top_n, sim, i, j)
            if sim > max_similarity[i]:
                max_similarity[i] = sim
                nearest_index[i] = j
            if sim > max_similarity[j]:
                max_similarity[j] = sim
                nearest_index[j] = i

    return np.asarray(sims, dtype=float), top_heap, max_similarity, nearest_index


def _pairwise_sample(records: list[_MoleculeRecord], top_n: int, sample_pairs: int, random_state: int):
    n = len(records)
    rng = np.random.default_rng(int(random_state))
    total_pairs = n * (n - 1) // 2
    sample_pairs = min(int(sample_pairs), total_pairs)

    sims = []
    top_heap = []
    max_similarity = np.full(n, np.nan, dtype=float)
    nearest_index = np.full(n, -1, dtype=int)
    seen = set()

    while len(seen) < sample_pairs:
        i = int(rng.integers(1, n))
        j = int(rng.integers(0, i))
        key = (i, j)
        if key in seen:
            continue
        seen.add(key)
        sim = float(DataStructs.TanimotoSimilarity(records[i].fingerprint, records[j].fingerprint))
        sims.append(sim)
        _push_top_pair(top_heap, top_n, sim, i, j)
        if np.isnan(max_similarity[i]) or sim > max_similarity[i]:
            max_similarity[i] = sim
            nearest_index[i] = j
        if np.isnan(max_similarity[j]) or sim > max_similarity[j]:
            max_similarity[j] = sim
            nearest_index[j] = i

    return np.asarray(sims, dtype=float), top_heap, max_similarity, nearest_index


def _bootstrap_mean_interval(values, random_state: int, n_bootstrap: int = 500, alpha: float = 0.05):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(int(random_state))
    means = []
    n = len(values)
    for _ in range(int(n_bootstrap)):
        sample = values[rng.integers(0, n, size=n)]
        means.append(float(np.mean(sample)))
    lower = float(np.quantile(means, alpha / 2.0))
    upper = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lower, upper


def _top_pairs_table(records: list[_MoleculeRecord], top_heap) -> pd.DataFrame:
    rows = []
    for similarity, i, j in sorted(top_heap, reverse=True):
        left = records[i]
        right = records[j]
        rows.append({
            "row_1": left.row_index + 1,
            "label_1": left.label,
            "smiles_1": left.smiles,
            "row_2": right.row_index + 1,
            "label_2": right.label,
            "smiles_2": right.smiles,
            "tanimoto": round(float(similarity), 4),
        })
    return pd.DataFrame(rows)


def _pair_table_from_indices(records: list[_MoleculeRecord], pair_items) -> pd.DataFrame:
    rows = []
    for similarity, i, j in sorted(pair_items, reverse=True):
        left = records[int(i)]
        right = records[int(j)]
        rows.append({
            "row_1": left.row_index + 1,
            "label_1": left.label,
            "smiles_1": left.smiles,
            "row_2": right.row_index + 1,
            "label_2": right.label,
            "smiles_2": right.smiles,
            "tanimoto": round(float(similarity), 4),
        })
    return pd.DataFrame(rows)


def _threshold_pair_tables(
    records: list[_MoleculeRecord],
    duplicate_threshold: float,
    analogue_threshold: float,
    max_rows: int,
    max_full_molecules: int,
    max_sample_pairs: int,
    random_state: int,
):
    n = len(records)
    if n < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    duplicate_items = []
    analogue_items = []
    network_items = []

    def add_pair(similarity, i, j):
        item = (float(similarity), int(i), int(j))
        if similarity >= float(analogue_threshold):
            network_items.append(item)
            analogue_items.append(item)
        if similarity >= float(duplicate_threshold):
            duplicate_items.append(item)

    if n <= int(max_full_molecules):
        fingerprints = [record.fingerprint for record in records]
        for i in range(1, n):
            sims = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
            for j, sim in enumerate(sims):
                add_pair(float(sim), i, j)
    else:
        rng = np.random.default_rng(int(random_state))
        total_pairs = n * (n - 1) // 2
        sample_pairs = min(int(max_sample_pairs), total_pairs)
        seen = set()
        while len(seen) < sample_pairs:
            i = int(rng.integers(1, n))
            j = int(rng.integers(0, i))
            key = (i, j)
            if key in seen:
                continue
            seen.add(key)
            sim = float(DataStructs.TanimotoSimilarity(records[i].fingerprint, records[j].fingerprint))
            add_pair(sim, i, j)

    duplicate_items = sorted(duplicate_items, reverse=True)[:int(max_rows)]
    analogue_items = sorted(analogue_items, reverse=True)[:int(max_rows)]
    network_items = sorted(network_items, reverse=True)[:int(max_rows)]

    return (
        _pair_table_from_indices(records, duplicate_items),
        _pair_table_from_indices(records, analogue_items),
        _network_edges_table(records, network_items),
    )


def _unique_table(records: list[_MoleculeRecord], max_similarity, nearest_index, limit: int) -> pd.DataFrame:
    rows = []
    for i, record in enumerate(records):
        nearest = int(nearest_index[i])
        nearest_record = records[nearest] if 0 <= nearest < len(records) else None
        rows.append({
            "row": record.row_index + 1,
            "label": record.label,
            "smiles": record.smiles,
            "max_similarity": (
                round(float(max_similarity[i]), 4)
                if np.isfinite(max_similarity[i])
                else np.nan
            ),
            "nearest_row": nearest_record.row_index + 1 if nearest_record else "",
            "nearest_label": nearest_record.label if nearest_record else "",
            "nearest_smiles": nearest_record.smiles if nearest_record else "",
        })

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values("max_similarity", na_position="last").head(int(limit)).reset_index(drop=True)


def _fingerprint_bit_matrix(records: list[_MoleculeRecord], n_bits: int) -> np.ndarray:
    matrix = np.zeros((len(records), int(n_bits)), dtype=np.uint8)
    for i, record in enumerate(records):
        arr = np.zeros((int(n_bits),), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(record.fingerprint, arr)
        matrix[i, :] = arr
    return matrix


def _cluster_lookup(cluster_assignments: pd.DataFrame) -> dict[int, int]:
    if not isinstance(cluster_assignments, pd.DataFrame) or cluster_assignments.empty:
        return {}
    lookup = {}
    for _, row in cluster_assignments.iterrows():
        try:
            lookup[int(row["row"])] = int(row["cluster_id"])
        except Exception:
            continue
    return lookup


def _fingerprint_pca_map(
    records: list[_MoleculeRecord],
    cluster_assignments: pd.DataFrame,
    data: pd.DataFrame,
    target_col: Optional[str],
    n_bits: int,
) -> pd.DataFrame:
    if PCA is None or len(records) < 3:
        return pd.DataFrame()

    fp_matrix = _fingerprint_bit_matrix(records, int(n_bits)).astype(float)
    coords = PCA(n_components=2, random_state=42).fit_transform(fp_matrix)
    cluster_by_row = _cluster_lookup(cluster_assignments)

    rows = []
    for i, record in enumerate(records):
        item = {
            "row": record.row_index + 1,
            "label": record.label,
            "smiles": record.smiles,
            "PC1": float(coords[i, 0]),
            "PC2": float(coords[i, 1]),
            "cluster_id": cluster_by_row.get(record.row_index + 1, -1),
        }
        if target_col and target_col in data.columns:
            item["target"] = pd.to_numeric(
                pd.Series([data.loc[record.row_index, target_col]]),
                errors="coerce",
            ).iloc[0]
        rows.append(item)
    return pd.DataFrame(rows)


def _similarity_matrix_for_records(records: list[_MoleculeRecord]) -> np.ndarray:
    n = len(records)
    matrix = np.eye(n, dtype=float)
    fingerprints = [record.fingerprint for record in records]
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
        for j, sim in enumerate(sims):
            matrix[i, j] = float(sim)
            matrix[j, i] = float(sim)
    return matrix


CSA_DENSE_LABEL = "DENSE"
CSA_MODERATE_LABEL = "MODERATE"
CSA_SPARSE_LABEL = "SPARSE"
CSA_ISOLATED_LABEL = "ISOLATED"


def classify_chemical_space_score(score: float) -> str:
    try:
        score = float(score)
    except (TypeError, ValueError):
        return CSA_ISOLATED_LABEL
    if score >= 0.85:
        return CSA_DENSE_LABEL
    if score >= 0.70:
        return CSA_MODERATE_LABEL
    if score >= 0.50:
        return CSA_SPARSE_LABEL
    return CSA_ISOLATED_LABEL


def _compute_projection(distance_matrix: np.ndarray, method: str, random_state: int) -> tuple[np.ndarray, str]:
    n = int(distance_matrix.shape[0])
    if n == 1:
        return np.zeros((1, 2), dtype=float), "single-point"

    method_aliases = {
        "AUTO": "AUTO",
        "UMAP": "UMAP",
        "MDS": "MDS",
        "TSNE": "TSNE",
        "T-SNE": "TSNE",
        "T_SNE": "TSNE",
    }
    method_normalized = method_aliases.get(
        str(method or "AUTO").strip().upper(),
        "AUTO",
    )
    if method_normalized in {"AUTO", "UMAP"}:
        try:
            import umap  # type: ignore

            n_neighbors = min(15, max(2, n - 1))
            coords = umap.UMAP(
                n_components=2,
                metric="precomputed",
                n_neighbors=n_neighbors,
                min_dist=0.08,
                random_state=int(random_state),
            ).fit_transform(distance_matrix)
            return np.asarray(coords, dtype=float), "UMAP"
        except Exception:
            if method_normalized == "UMAP":
                method_normalized = "MDS"

    if method_normalized == "TSNE" and TSNE is not None and n >= 3:
        perplexity = max(1, min(30, (n - 1) // 3))
        coords = TSNE(
            n_components=2,
            metric="precomputed",
            init="random",
            perplexity=float(perplexity),
            random_state=int(random_state),
            learning_rate="auto",
        ).fit_transform(distance_matrix)
        return np.asarray(coords, dtype=float), "TSNE"

    if MDS is not None:
        try:
            mds = MDS(
                n_components=2,
                dissimilarity="precomputed",
                random_state=int(random_state),
                normalized_stress="auto",
            )
        except TypeError:
            mds = MDS(
                n_components=2,
                dissimilarity="precomputed",
                random_state=int(random_state),
            )
        coords = mds.fit_transform(distance_matrix)
        return np.asarray(coords, dtype=float), "MDS"

    if PCA is not None:
        coords = PCA(n_components=2, random_state=int(random_state)).fit_transform(distance_matrix)
        return np.asarray(coords, dtype=float), "PCA fallback"

    return np.column_stack([np.arange(n, dtype=float), np.zeros(n, dtype=float)]), "linear fallback"


def _projection_quality(distance_matrix: np.ndarray, coords: np.ndarray, method: str) -> dict:
    quality = {
        "projection_method": str(method),
        "trustworthiness": np.nan,
        "distance_correlation": np.nan,
        "stress": np.nan,
    }
    try:
        n = int(distance_matrix.shape[0])
        if n < 3:
            return quality
        projected = np.sqrt(
            np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
        )
        tri = np.triu_indices(n, k=1)
        original_values = np.asarray(distance_matrix[tri], dtype=float)
        projected_values = np.asarray(projected[tri], dtype=float)
        finite = np.isfinite(original_values) & np.isfinite(projected_values)
        if finite.sum() >= 2:
            quality["distance_correlation"] = float(
                np.corrcoef(original_values[finite], projected_values[finite])[0, 1]
            )
            numerator = float(np.sum((original_values[finite] - projected_values[finite]) ** 2))
            denominator = float(np.sum(original_values[finite] ** 2))
            quality["stress"] = float(np.sqrt(numerator / denominator)) if denominator > 0 else np.nan
        try:
            from sklearn.manifold import trustworthiness

            quality["trustworthiness"] = float(
                trustworthiness(
                    distance_matrix,
                    coords,
                    n_neighbors=max(1, min(5, n - 1)),
                    metric="precomputed",
                )
            )
        except Exception:
            pass
    except Exception:
        pass
    return quality


def build_similarity_edges(
    similarity_matrix: np.ndarray,
    threshold: float = 0.75,
    top_k: int = 5,
    max_edges: int = 3000,
) -> pd.DataFrame:
    n = int(similarity_matrix.shape[0])
    rows = []
    seen = set()
    for i in range(n):
        candidates = [
            (float(similarity_matrix[i, j]), int(j))
            for j in range(n)
            if j != i and float(similarity_matrix[i, j]) >= float(threshold)
        ]
        candidates.sort(reverse=True)
        for sim, j in candidates[: max(1, int(top_k))]:
            key = tuple(sorted((i, j)))
            if key in seen:
                continue
            seen.add(key)
            rows.append({"source": key[0], "target": key[1], "tanimoto": round(sim, 4)})
    if not rows:
        return pd.DataFrame(columns=["source", "target", "tanimoto"])
    return pd.DataFrame(rows).sort_values("tanimoto", ascending=False).head(int(max_edges)).reset_index(drop=True)


def compute_connected_components(edges: pd.DataFrame, n_nodes: int) -> tuple[list[int], int, int]:
    parent = list(range(int(n_nodes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(int(a))
        rb = find(int(b))
        if ra != rb:
            parent[rb] = ra

    if isinstance(edges, pd.DataFrame) and not edges.empty:
        for _, edge in edges.iterrows():
            union(int(edge["source"]), int(edge["target"]))

    roots = [find(i) for i in range(int(n_nodes))]
    root_to_id = {root: idx + 1 for idx, root in enumerate(sorted(set(roots)))}
    component_ids = [root_to_id[root] for root in roots]
    counts = pd.Series(component_ids).value_counts()
    return component_ids, int(len(counts)), int(counts.max()) if not counts.empty else 0


def _component_sizes(labels: Iterable[int]) -> dict[int, int]:
    counts = pd.Series([int(label) for label in labels]).value_counts().to_dict()
    return {int(key): int(value) for key, value in counts.items()}


def _internal_similarity(similarity_matrix: np.ndarray, indices: list[int]) -> float:
    if len(indices) < 2:
        return np.nan
    values = []
    for pos, i in enumerate(indices[:-1]):
        for j in indices[pos + 1:]:
            values.append(float(similarity_matrix[int(i), int(j)]))
    return float(np.mean(values)) if values else np.nan


def _groups_table_from_nodes(
    nodes: pd.DataFrame,
    similarity_matrix: np.ndarray,
    method: str,
    group_type: str,
    small_group_limit: int,
) -> pd.DataFrame:
    rows = []
    if not isinstance(nodes, pd.DataFrame) or nodes.empty:
        return pd.DataFrame()
    for group_id, group in nodes.groupby("group_id", dropna=False):
        indices = [int(i) for i in group["node_index"].tolist()]
        representative = group.sort_values(
            ["degree", "nearest_neighbor_tanimoto"],
            ascending=[False, False],
        ).iloc[0]
        group_size = int(len(group))
        rows.append({
            "method": method,
            "group_id": int(group_id) if pd.notna(group_id) else -1,
            "group_type": group_type,
            "group_size": group_size,
            "representative_molecule": representative.get("name", ""),
            "representative_smiles": representative.get("SMILES", ""),
            "members_preview": "; ".join(group["name"].astype(str).head(8).tolist()),
            "average_internal_similarity": round(_internal_similarity(similarity_matrix, indices), 4)
            if group_size > 1 else np.nan,
            "singleton_count_in_group": int(coerce_boolean_series(group["is_singleton_selected"]).sum()),
            "notes": "small isolated group" if group_size <= int(small_group_limit) else "",
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["group_size", "group_id"], ascending=[False, True]).reset_index(drop=True)


def _singletons_table_from_nodes(nodes: pd.DataFrame, method: str) -> pd.DataFrame:
    if not isinstance(nodes, pd.DataFrame) or nodes.empty:
        return pd.DataFrame()
    singles = nodes[coerce_boolean_series(nodes["is_singleton_selected"])].copy()
    if singles.empty:
        return pd.DataFrame(columns=[
            "molecule_name",
            "smiles",
            "method",
            "singleton_reason",
            "nearest_neighbor",
            "nearest_neighbor_tanimoto",
            "component_id",
            "cluster_id",
            "degree",
            "csa_class",
        ])
    return pd.DataFrame({
        "molecule_name": singles["name"],
        "smiles": singles["SMILES"],
        "method": method,
        "singleton_reason": singles["singleton_reason"],
        "nearest_neighbor": singles["nearest_neighbor"],
        "nearest_neighbor_tanimoto": singles["nearest_neighbor_tanimoto"],
        "component_id": singles.get("component_id", singles["group_id"]),
        "cluster_id": singles["group_id"],
        "degree": singles["degree"],
        "csa_class": singles["csa_class"],
    }).reset_index(drop=True)


def _small_groups_table(groups_table: pd.DataFrame, small_group_limit: int) -> pd.DataFrame:
    if not isinstance(groups_table, pd.DataFrame) or groups_table.empty:
        return pd.DataFrame()
    small = groups_table[
        (pd.to_numeric(groups_table["group_size"], errors="coerce") > 1)
        & (pd.to_numeric(groups_table["group_size"], errors="coerce") <= int(small_group_limit))
    ].copy()
    if small.empty:
        return pd.DataFrame()
    return small.rename(columns={
        "group_size": "size",
        "members_preview": "members",
        "average_internal_similarity": "mean_similarity",
    })[["group_id", "size", "members", "method", "mean_similarity", "notes"]].reset_index(drop=True)


def _community_interpretation(summary: dict, method: str) -> str:
    n_groups = int(summary.get("n_groups", 0) or 0)
    n_singletons = int(summary.get("n_singletons", 0) or 0)
    largest_fraction = float(summary.get("largest_group_fraction", 0.0) or 0.0)
    small_groups = int(summary.get("n_small_groups", 0) or 0)
    if n_groups <= 1 and largest_fraction >= 0.8:
        return "Большинство веществ входит в одну крупную структурную группу; одиночные или малые изолированные области выражены слабо."
    if n_singletons >= max(3, int(0.25 * max(summary.get("n_nodes", 0), 1))):
        return "При выбранном пороге сходства датасет заметно фрагментируется: много одиночных веществ без близких аналогов."
    if small_groups > 0 and largest_fraction >= 0.45:
        return "Большая часть веществ входит в крупное сообщество, но присутствуют малые изолированные группы и отдельные структуры."
    if method == "Butina clustering":
        return "Butina clustering выделяет компактные химически близкие группы и singleton-кластеры по fingerprint-дистанции."
    return "Датасет распадается на несколько структурных сообществ; одиночные вещества и малые группы можно рассматривать как зоны слабого покрытия аналогами."


def _methyl_prefix(count: int) -> str:
    return {
        1: "methyl",
        2: "dimethyl",
        3: "trimethyl",
        4: "tetramethyl",
        5: "pentamethyl",
        6: "hexamethyl",
    }.get(int(count), f"{int(count)}-methyl")


def _exact_pattern_label(pattern: str) -> str:
    pattern = _safe_text(pattern)
    if not pattern:
        return "unclassified"
    if pattern == "n-alkane":
        return "n-alkanes"
    parts = [part.strip() for part in pattern.split(";") if part.strip()]
    parsed = []
    for part in parts:
        if "-" not in part:
            parsed.append((part, ""))
            continue
        pos, substituent = part.split("-", 1)
        parsed.append((pos.strip(), substituent.strip()))
    substituents = [sub for _, sub in parsed if sub]
    positions = [pos for pos, _ in parsed if pos]
    if substituents and all(sub == "methyl" for sub in substituents):
        return f"{','.join(positions)}-{_methyl_prefix(len(substituents))}alkanes"
    if substituents and all(sub == "ethyl" for sub in substituents):
        prefix = "ethyl" if len(substituents) == 1 else f"{len(substituents)}-ethyl"
        return f"{','.join(positions)}-{prefix}alkanes"
    cleaned = pattern.replace("; ", ",")
    return f"{cleaned}-substituted alkanes"


def _broad_series_from_exact(pattern: str) -> str:
    pattern = _safe_text(pattern)
    if not pattern:
        return "other / unclassified"
    if pattern == "n-alkane":
        return "n-alkanes"
    parts = [part.strip() for part in pattern.split(";") if part.strip()]
    substituents = []
    for part in parts:
        if "-" in part:
            substituents.append(part.split("-", 1)[1].strip())
        else:
            substituents.append(part)
    methyl_count = sum(1 for item in substituents if item == "methyl")
    ethyl_count = sum(1 for item in substituents if item == "ethyl")
    if methyl_count == len(substituents):
        return {
            1: "methylalkanes",
            2: "dimethylalkanes",
            3: "trimethylalkanes",
            4: "tetramethylalkanes",
        }.get(methyl_count, "mixed-substituted alkanes")
    if ethyl_count > 0 and methyl_count == 0:
        return "ethyl-substituted alkanes"
    if ethyl_count > 0 or methyl_count > 0:
        return "mixed-substituted alkanes"
    return "other / unclassified"


def _pattern_interpretation(group_df: pd.DataFrame) -> str:
    if not isinstance(group_df, pd.DataFrame) or group_df.empty:
        return "Точные структурные паттерны не выделены: нет распознанных ациклических алканов."
    top = group_df.sort_values("group_size", ascending=False).head(3)
    top_names = ", ".join(top["exact_pattern"].astype(str).tolist())
    weak = group_df[coerce_boolean_series(group_df["rare_group"])].sort_values("group_size").head(3)
    weak_names = ", ".join(weak["exact_pattern"].astype(str).tolist())
    unclassified = int(coerce_boolean_series(group_df["unclassified_group"]).sum())
    text = f"Наиболее крупные точные паттерны представлены группами: {top_names}."
    if weak_names:
        text += f" Слабопредставленные паттерны: {weak_names}."
    if unclassified:
        text += f" Нераспознанных групп: {unclassified}; они сохранены как other / unclassified."
    return text


def _exact_pattern_hierarchy(
    records: list[_MoleculeRecord],
    data: pd.DataFrame,
    target_col: Optional[str],
    small_group_threshold: int = 2,
    rare_group_threshold: int = 3,
) -> dict:
    rows = []
    if saod2_classify_alkane is None:
        return {
            "groups": pd.DataFrame(),
            "rare_groups": pd.DataFrame(),
            "members": pd.DataFrame(),
            "interpretation": "SAOD-классификатор точных паттернов недоступен.",
        }

    for record in records:
        classification = saod2_classify_alkane(record.mol)
        is_alkane = bool(classification.get("is_acyclic_alkane"))
        raw_pattern = classification.get("exact_pattern", "") if is_alkane else ""
        exact_pattern = _exact_pattern_label(raw_pattern)
        broad_series = _broad_series_from_exact(raw_pattern)
        if not is_alkane:
            exact_pattern = "unclassified"
            broad_series = "other / unclassified"
        item = {
            "row": record.row_index + 1,
            "molecule_name": record.label,
            "smiles": record.smiles,
            "broad_series": broad_series,
            "exact_pattern": exact_pattern,
            "raw_exact_pattern": raw_pattern,
            "is_acyclic_alkane": is_alkane,
        }
        if target_col and target_col in data.columns:
            item["property"] = pd.to_numeric(
                pd.Series([data.loc[record.row_index, target_col]]),
                errors="coerce",
            ).iloc[0]
        rows.append(item)

    members = pd.DataFrame(rows)
    if members.empty:
        return {"groups": pd.DataFrame(), "rare_groups": pd.DataFrame(), "members": members, "interpretation": ""}

    total = max(len(members), 1)
    group_rows = []
    for (broad, exact), group in members.groupby(["broad_series", "exact_pattern"], dropna=False):
        size = int(len(group))
        group_rows.append({
            "broad_series": broad,
            "exact_pattern": exact,
            "group_size": size,
            "dataset_fraction": round(size / total, 4),
            "parent_group": broad,
            "rare_group": bool(size <= int(rare_group_threshold)),
            "small_group": bool(size <= int(small_group_threshold)),
            "singleton_group": bool(size == 1),
            "unclassified_group": bool(broad == "other / unclassified" or exact == "unclassified"),
            "representative_examples": "; ".join(group["molecule_name"].astype(str).head(5).tolist()),
            "member_smiles": "; ".join(group["smiles"].astype(str).head(5).tolist()),
            "mean_property": float(pd.to_numeric(group.get("property"), errors="coerce").mean())
            if "property" in group.columns else np.nan,
        })
    groups = pd.DataFrame(group_rows).sort_values(["group_size", "exact_pattern"], ascending=[False, True]).reset_index(drop=True)

    rare_groups = groups[
        coerce_boolean_series(groups["rare_group"])
        | coerce_boolean_series(groups["singleton_group"])
        | coerce_boolean_series(groups["unclassified_group"])
    ].copy()
    if not rare_groups.empty:
        rare_groups["reason_flag"] = np.select(
            [
                coerce_boolean_series(rare_groups["unclassified_group"]),
                coerce_boolean_series(rare_groups["singleton_group"]),
                coerce_boolean_series(rare_groups["small_group"]),
                coerce_boolean_series(rare_groups["rare_group"]),
            ],
            ["unclassified", "singleton", "small", "rare"],
            default="rare",
        )
        rare_groups["member_names"] = rare_groups["representative_examples"]

    return {
        "groups": groups,
        "rare_groups": rare_groups.reset_index(drop=True),
        "members": members,
        "interpretation": _pattern_interpretation(groups),
    }


def _normalize_community_method(method: str) -> str:
    value = str(method or "").strip().lower()
    aliases = {
        "connected_components": "Connected components",
        "connected components": "Connected components",
        "component": "Connected components",
        "butina": "Butina clustering",
        "butina_clustering": "Butina clustering",
        "butina clustering": "Butina clustering",
        "dbscan": "DBSCAN",
        "similarity_network": "Similarity network",
        "similarity network": "Similarity network",
        "singletons_only": "Singletons only",
        "singletons only": "Singletons only",
    }
    return aliases.get(value, "Connected components")


def _normalize_singleton_criterion(criterion: str) -> str:
    value = str(criterion or "").strip().lower()
    aliases = {
        "combined": "combined",
        "component_size_1": "component size == 1",
        "component size == 1": "component size == 1",
        "no_neighbors": "no neighbors above threshold",
        "no neighbors above threshold": "no neighbors above threshold",
        "cluster_size_lte_n": "cluster size <= N",
        "cluster size <= n": "cluster size <= N",
        "dbscan_noise": "DBSCAN noise",
        "dbscan noise": "DBSCAN noise",
    }
    return aliases.get(value, "combined")


def analyze_structural_communities(
    map_df: pd.DataFrame,
    similarity_matrix: np.ndarray,
    method: str = "Connected components",
    threshold: float = 0.75,
    top_k: int = 5,
    min_cluster_size: int = 3,
    butina_cutoff: float = 0.20,
    dbscan_eps: float = 0.25,
    dbscan_min_samples: int = 2,
    singleton_criterion: str = "combined",
    max_edges: int = 3000,
) -> dict:
    """Build structural community labels and tables from an existing CSA map."""
    if not isinstance(map_df, pd.DataFrame) or map_df.empty:
        return {"nodes": pd.DataFrame(), "edges": pd.DataFrame(), "groups": pd.DataFrame(), "singletons": pd.DataFrame(), "small_groups": pd.DataFrame(), "summary": {}}

    nodes = map_df.reset_index(drop=True).copy()
    n = len(nodes)
    nodes["node_index"] = np.arange(n)
    method = _normalize_community_method(method)
    singleton_criterion = _normalize_singleton_criterion(singleton_criterion)
    threshold = float(threshold)
    edges = build_similarity_edges(similarity_matrix, threshold=threshold, top_k=int(top_k), max_edges=int(max_edges))
    degree = np.zeros(n, dtype=int)
    if isinstance(edges, pd.DataFrame) and not edges.empty:
        for _, edge in edges.iterrows():
            degree[int(edge["source"])] += 1
            degree[int(edge["target"])] += 1
    nodes["degree"] = degree
    nodes["has_no_close_neighbors"] = degree == 0

    group_type = "component"
    is_noise = np.zeros(n, dtype=bool)
    if method == "Butina clustering" and Butina is not None:
        distances = []
        for i in range(1, n):
            for j in range(i):
                distances.append(1.0 - float(similarity_matrix[i, j]))
        clusters = Butina.ClusterData(distances, n, float(butina_cutoff), isDistData=True, reordering=True)
        labels = np.zeros(n, dtype=int)
        for cluster_id, cluster in enumerate(clusters, start=1):
            for idx in cluster:
                labels[int(idx)] = int(cluster_id)
        group_type = "Butina cluster"
    elif method == "DBSCAN" and DBSCAN is not None:
        distance_matrix = np.clip(1.0 - np.asarray(similarity_matrix, dtype=float), 0.0, 1.0)
        np.fill_diagonal(distance_matrix, 0.0)
        raw_labels = DBSCAN(
            eps=float(dbscan_eps),
            min_samples=int(dbscan_min_samples),
            metric="precomputed",
        ).fit_predict(distance_matrix)
        is_noise = raw_labels == -1
        labels = np.where(is_noise, -(np.arange(n) + 1), raw_labels + 1)
        group_type = "DBSCAN cluster"
    else:
        labels, _, _ = compute_connected_components(edges, n)
        labels = np.asarray(labels, dtype=int)
        if method == "Similarity network":
            group_type = "network component"
        elif method == "Singletons only":
            group_type = "singleton component"
        else:
            group_type = "component"

    nodes["group_id"] = labels.astype(int)
    nodes["cluster_id"] = nodes["group_id"]
    nodes["component_id"] = nodes["group_id"]
    nodes["method"] = method
    nodes["is_noise"] = is_noise
    sizes = _component_sizes(nodes["group_id"])
    nodes["group_size"] = nodes["group_id"].map(sizes).fillna(1).astype(int)
    nodes["is_singleton_graph"] = nodes["group_size"] == 1
    nodes["is_singleton_butina"] = (nodes["group_size"] == 1) if method == "Butina clustering" else False
    nodes["is_noise_dbscan"] = coerce_boolean_series(nodes["is_noise"])
    nodes["is_small_isolated_group"] = nodes["group_size"] <= int(min_cluster_size)

    if singleton_criterion == "component size == 1":
        selected_singleton = coerce_boolean_series(nodes["is_singleton_graph"])
        reason = "component size == 1"
    elif singleton_criterion == "no neighbors above threshold":
        selected_singleton = coerce_boolean_series(nodes["has_no_close_neighbors"])
        reason = "no neighbors above threshold"
    elif singleton_criterion == "cluster size <= N":
        selected_singleton = coerce_boolean_series(nodes["is_small_isolated_group"])
        reason = f"cluster size <= {int(min_cluster_size)}"
    elif singleton_criterion == "DBSCAN noise":
        selected_singleton = coerce_boolean_series(nodes["is_noise_dbscan"])
        reason = "DBSCAN noise"
    else:
        selected_singleton = (
            coerce_boolean_series(nodes["has_no_close_neighbors"])
            | coerce_boolean_series(nodes["is_singleton_graph"])
            | coerce_boolean_series(nodes["is_singleton_butina"])
            | coerce_boolean_series(nodes["is_noise_dbscan"])
        )
        reason = "combined criterion"

    nodes["is_singleton_selected"] = selected_singleton
    nodes["singleton_reason"] = np.where(selected_singleton, reason, "")
    if method == "Singletons only":
        nodes = nodes[coerce_boolean_series(nodes["is_singleton_selected"])].copy()
        visible = set(nodes["node_index"].astype(int).tolist())
        if isinstance(edges, pd.DataFrame) and not edges.empty:
            edges = edges[
                edges["source"].astype(int).isin(visible)
                & edges["target"].astype(int).isin(visible)
            ].copy()

    groups_table = _groups_table_from_nodes(nodes, similarity_matrix, method, group_type, int(min_cluster_size))
    singletons_table = _singletons_table_from_nodes(nodes, method)
    small_groups = _small_groups_table(groups_table, int(min_cluster_size))

    n_groups = int(nodes["group_id"].nunique()) if not nodes.empty else 0
    largest_group_size = int(nodes["group_size"].max()) if not nodes.empty else 0
    summary = {
        "method": method,
        "n_nodes": int(len(nodes)),
        "n_groups": n_groups,
        "n_singletons": int(coerce_boolean_series(nodes["is_singleton_selected"]).sum()) if not nodes.empty else 0,
        "n_small_groups": int(len(small_groups)) if isinstance(small_groups, pd.DataFrame) else 0,
        "largest_group_size": largest_group_size,
        "largest_group_fraction": float(largest_group_size / max(len(nodes), 1)),
        "noise_points": int(coerce_boolean_series(nodes["is_noise"]).sum()) if not nodes.empty else 0,
        "mean_degree": float(np.mean(nodes["degree"])) if not nodes.empty else np.nan,
        "mean_cluster_size": float(np.mean(nodes["group_size"])) if not nodes.empty else np.nan,
        "no_close_neighbors": int(coerce_boolean_series(nodes["has_no_close_neighbors"]).sum()) if not nodes.empty else 0,
    }
    summary["interpretation"] = _community_interpretation(summary, method)
    return {
        "nodes": nodes.reset_index(drop=True),
        "edges": edges.reset_index(drop=True) if isinstance(edges, pd.DataFrame) else pd.DataFrame(),
        "groups": groups_table,
        "singletons": singletons_table,
        "small_groups": small_groups,
        "summary": summary,
    }


def detect_exact_duplicates(canonical_smiles: Iterable[str]) -> pd.DataFrame:
    rows = []
    groups: dict[str, list[int]] = {}
    for idx, smiles in enumerate(canonical_smiles):
        smiles = _safe_text(smiles)
        if smiles:
            groups.setdefault(smiles, []).append(int(idx))
    for smiles, indices in groups.items():
        if len(indices) < 2:
            continue
        for pos, i in enumerate(indices[:-1]):
            for j in indices[pos + 1:]:
                rows.append({"source": i, "target": j, "canonical_smiles": smiles})
    return pd.DataFrame(rows)


def detect_near_duplicates(similarity_matrix: np.ndarray, threshold: float = 0.95) -> pd.DataFrame:
    rows = []
    n = int(similarity_matrix.shape[0])
    for i in range(1, n):
        for j in range(i):
            sim = float(similarity_matrix[i, j])
            if sim >= float(threshold):
                rows.append({"source": i, "target": j, "tanimoto": round(sim, 4)})
    return pd.DataFrame(rows)


def _duplicate_detail_table(
    records: list[_MoleculeRecord],
    similarity_matrix: np.ndarray,
    canonical_smiles: list[str],
    duplicate_threshold: float,
    analogue_threshold: float,
    max_rows: int,
) -> pd.DataFrame:
    rows = []
    exact_keys = set()
    exact_df = detect_exact_duplicates(canonical_smiles)
    if isinstance(exact_df, pd.DataFrame) and not exact_df.empty:
        for _, pair in exact_df.iterrows():
            i = int(pair["source"])
            j = int(pair["target"])
            exact_keys.add(tuple(sorted((i, j))))
            left = records[i]
            right = records[j]
            rows.append({
                "molecule_1": left.label,
                "smiles_1": left.smiles,
                "molecule_2": right.label,
                "smiles_2": right.smiles,
                "tanimoto": 1.0,
                "duplicate_type": "exact duplicate",
            })

    n = len(records)
    for i in range(1, n):
        for j in range(i):
            key = tuple(sorted((i, j)))
            if key in exact_keys:
                continue
            sim = float(similarity_matrix[i, j])
            duplicate_type = ""
            if sim >= float(duplicate_threshold):
                duplicate_type = "near duplicate"
            elif sim >= float(analogue_threshold):
                duplicate_type = "strong analogue"
            if duplicate_type:
                left = records[i]
                right = records[j]
                rows.append({
                    "molecule_1": left.label,
                    "smiles_1": left.smiles,
                    "molecule_2": right.label,
                    "smiles_2": right.smiles,
                    "tanimoto": round(sim, 4),
                    "duplicate_type": duplicate_type,
                })

    if not rows:
        return pd.DataFrame(columns=["molecule_1", "smiles_1", "molecule_2", "smiles_2", "tanimoto", "duplicate_type"])
    return pd.DataFrame(rows).sort_values("tanimoto", ascending=False).head(int(max_rows)).reset_index(drop=True)


def _final_chemical_space(
    records: list[_MoleculeRecord],
    data: pd.DataFrame,
    target_col: Optional[str],
    similarity_matrix: np.ndarray,
    projection_method: str,
    edge_threshold: float,
    edge_top_k: int,
    duplicate_threshold: float,
    analogue_threshold: float,
    random_state: int,
    max_edges: int,
    max_table_pairs: int,
) -> dict:
    n = len(records)
    if n == 0:
        return {
            "map": pd.DataFrame(),
            "edges": pd.DataFrame(),
            "nearest_neighbors": pd.DataFrame(),
            "duplicates": pd.DataFrame(),
            "projection_method": "",
            "projection_quality": {},
            "random_seed": int(random_state),
            "n_components": 0,
            "largest_component_size": 0,
        }

    distance_matrix = np.clip(1.0 - np.asarray(similarity_matrix, dtype=float), 0.0, 1.0)
    np.fill_diagonal(distance_matrix, 0.0)
    coords, actual_method = _compute_projection(distance_matrix, projection_method, int(random_state))
    projection_quality = _projection_quality(distance_matrix, coords, actual_method)
    edges = build_similarity_edges(
        similarity_matrix,
        threshold=float(edge_threshold),
        top_k=int(edge_top_k),
        max_edges=int(max_edges),
    )
    components, n_components, largest_component_size = compute_connected_components(edges, n)

    masked = similarity_matrix.copy()
    np.fill_diagonal(masked, -np.inf)
    nearest_idx = np.argmax(masked, axis=1)
    nearest_score = np.max(masked, axis=1)
    nearest_score = np.where(np.isfinite(nearest_score), nearest_score, np.nan)
    close_counts = np.sum(masked >= float(edge_threshold), axis=1)
    local_density = []
    k = max(1, int(edge_top_k))
    for i in range(n):
        vals = sorted([float(v) for v in masked[i] if np.isfinite(v)], reverse=True)[:k]
        local_density.append(float(np.mean(vals)) if vals else np.nan)

    canonical = [Chem.MolToSmiles(record.mol, canonical=True, isomericSmiles=True) for record in records]
    duplicate_pairs = _duplicate_detail_table(
        records=records,
        similarity_matrix=similarity_matrix,
        canonical_smiles=canonical,
        duplicate_threshold=float(duplicate_threshold),
        analogue_threshold=float(analogue_threshold),
        max_rows=int(max_table_pairs),
    )
    exact_patterns = _exact_pattern_hierarchy(
        records=records,
        data=data,
        target_col=target_col,
    )

    rows = []
    nearest_rows = []
    for i, record in enumerate(records):
        nearest = int(nearest_idx[i]) if np.isfinite(nearest_score[i]) and nearest_idx[i] >= 0 else -1
        nearest_record = records[nearest] if 0 <= nearest < n else None
        csa_class = classify_chemical_space_score(nearest_score[i])
        item = {
            "row": record.row_index + 1,
            "name": record.label,
            "SMILES": record.original_smiles,
            "fingerprint_smiles": record.fingerprint_smiles,
            "fingerprint_structure_source": record.fingerprint_structure_source,
            "canonical_smiles": canonical[i],
            "csa_x": float(coords[i, 0]),
            "csa_y": float(coords[i, 1]),
            "nearest_neighbor": nearest_record.label if nearest_record else "",
            "nearest_neighbor_smiles": nearest_record.original_smiles if nearest_record else "",
            "nearest_neighbor_tanimoto": round(float(nearest_score[i]), 4) if np.isfinite(nearest_score[i]) else np.nan,
            "close_analog_count": int(close_counts[i]),
            "local_density": round(float(local_density[i]), 4) if np.isfinite(local_density[i]) else np.nan,
            "connected_component": int(components[i]),
            "csa_class": csa_class,
            "is_singleton": bool(close_counts[i] == 0),
            "is_structural_outlier": bool(csa_class == CSA_ISOLATED_LABEL),
        }
        if target_col and target_col in data.columns:
            item["experimental_value"] = pd.to_numeric(
                pd.Series([data.loc[record.row_index, target_col]]),
                errors="coerce",
            ).iloc[0]
        rows.append(item)
        nearest_rows.append({
            "molecule": record.label,
            "SMILES": record.original_smiles,
            "fingerprint_smiles": record.fingerprint_smiles,
            "nearest_neighbor": item["nearest_neighbor"],
            "nearest_neighbor_smiles": item["nearest_neighbor_smiles"],
            "nearest_neighbor_tanimoto": item["nearest_neighbor_tanimoto"],
            "close_analog_count": item["close_analog_count"],
            "csa_class": csa_class,
        })

    map_df = pd.DataFrame(rows)
    return {
        "map": map_df,
        "edges": edges,
        "nearest_neighbors": pd.DataFrame(nearest_rows),
        "duplicates": duplicate_pairs,
        "similarity_matrix": similarity_matrix,
        "exact_patterns": exact_patterns,
        "projection_method": actual_method,
        "projection_quality": projection_quality,
        "random_seed": int(random_state),
        "n_components": int(n_components),
        "largest_component_size": int(largest_component_size),
    }


def _heatmap_payload(
    records: list[_MoleculeRecord],
    cluster_assignments: pd.DataFrame,
    max_molecules: int,
    random_state: int,
) -> dict:
    if not records:
        return {"matrix": pd.DataFrame(), "molecules": pd.DataFrame(), "sampled": False}

    cluster_by_row = _cluster_lookup(cluster_assignments)
    sortable = []
    for i, record in enumerate(records):
        sortable.append((
            cluster_by_row.get(record.row_index + 1, 10**9),
            record.row_index + 1,
            i,
        ))
    sortable = sorted(sortable)

    sampled = len(sortable) > int(max_molecules)
    if sampled:
        rng = np.random.default_rng(int(random_state))
        selected_positions = sorted(rng.choice(
            np.arange(len(sortable)),
            size=int(max_molecules),
            replace=False,
        ).tolist())
        sortable = [sortable[pos] for pos in selected_positions]
        sortable = sorted(sortable)

    selected_records = [records[i] for _, _, i in sortable]
    matrix = _similarity_matrix_for_records(selected_records)
    labels = [
        f"C{cluster_by_row.get(record.row_index + 1, -1)} | {record.row_index + 1}"
        for record in selected_records
    ]
    molecules = pd.DataFrame({
        "matrix_index": list(range(1, len(selected_records) + 1)),
        "row": [record.row_index + 1 for record in selected_records],
        "label": [record.label for record in selected_records],
        "smiles": [record.smiles for record in selected_records],
        "cluster_id": [cluster_by_row.get(record.row_index + 1, -1) for record in selected_records],
    })
    if saod2_classify_alkane is not None:
        exact_patterns = []
        for record in selected_records:
            try:
                classification = saod2_classify_alkane(record.mol)
                exact_patterns.append(str(classification.get("exact_pattern", "")))
            except Exception:
                exact_patterns.append("")
        molecules["exact_pattern"] = exact_patterns
    return {
        "matrix": pd.DataFrame(matrix, index=labels, columns=labels),
        "molecules": molecules,
        "sampled": bool(sampled),
    }


def _network_edges_table(records: list[_MoleculeRecord], pair_items) -> pd.DataFrame:
    rows = []
    for similarity, i, j in pair_items:
        left = records[int(i)]
        right = records[int(j)]
        rows.append({
            "source_row": left.row_index + 1,
            "source_label": left.label,
            "source_smiles": left.smiles,
            "target_row": right.row_index + 1,
            "target_label": right.label,
            "target_smiles": right.smiles,
            "tanimoto": round(float(similarity), 4),
        })
    return pd.DataFrame(rows)


def _cluster_records(records: list[_MoleculeRecord], similarity_threshold: float):
    n = len(records)
    if n == 0 or Butina is None:
        return pd.DataFrame(), pd.DataFrame()

    if n == 1:
        clusters = ((0,),)
    else:
        distances = []
        fingerprints = [record.fingerprint for record in records]
        for i in range(1, n):
            sims = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
            distances.extend([1.0 - float(value) for value in sims])
        clusters = Butina.ClusterData(
            distances,
            n,
            1.0 - float(similarity_threshold),
            isDistData=True,
            reordering=True,
        )

    assignment_rows = []
    summary_rows = []
    for cluster_id, cluster in enumerate(clusters, start=1):
        cluster = tuple(int(i) for i in cluster)
        members = [records[i] for i in cluster]
        representative = members[0]
        summary_rows.append({
            "cluster_id": cluster_id,
            "n": len(members),
            "percent": round(len(members) / max(n, 1) * 100.0, 2),
            "representative_row": representative.row_index + 1,
            "representative_smiles": representative.smiles,
            "examples": "; ".join(member.smiles for member in members[:5]),
        })
        for member in members:
            assignment_rows.append({
                "row": member.row_index + 1,
                "label": member.label,
                "smiles": member.smiles,
                "cluster_id": cluster_id,
                "cluster_size": len(members),
            })

    return (
        pd.DataFrame(summary_rows).sort_values("n", ascending=False).reset_index(drop=True),
        pd.DataFrame(assignment_rows).sort_values(["cluster_id", "row"]).reset_index(drop=True),
    )


def _cluster_threshold_sensitivity(records: list[_MoleculeRecord], thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90]
    rows = []
    for threshold in thresholds:
        cluster_summary, _ = _cluster_records(records, float(threshold))
        if cluster_summary.empty:
            rows.append({
                "tanimoto_threshold": float(threshold),
                "n_clusters": 0,
                "n_singletons": 0,
                "largest_cluster_size": 0,
            })
            continue
        rows.append({
            "tanimoto_threshold": float(threshold),
            "n_clusters": int(len(cluster_summary)),
            "n_singletons": int((cluster_summary["n"] == 1).sum()),
            "largest_cluster_size": int(cluster_summary["n"].max()),
        })
    return pd.DataFrame(rows)


def _status_from_summary(summary: dict) -> tuple[str, str]:
    n = int(summary.get("valid_structures", 0) or 0)
    if n < 3:
        return "недостаточно данных", "Для устойчивой оценки нужно хотя бы 3 валидные структуры."

    mean_sim = summary.get("mean_tanimoto", np.nan)
    singleton_clusters = int(summary.get("singleton_clusters", 0) or 0)
    n_clusters = int(summary.get("n_clusters", 0) or 0)
    largest_pct = float(summary.get("largest_cluster_percent", 0.0) or 0.0)
    very_close = int(summary.get("pairs_gt_0_95", 0) or 0)

    reasons = []
    if np.isfinite(mean_sim):
        if mean_sim > 0.70 and n_clusters <= max(3, int(n * 0.10)):
            status = "низкое разнообразие"
            reasons.append("среднее Tanimoto-сходство высокое")
        elif mean_sim < 0.40 and n_clusters >= max(4, int(n * 0.10)):
            status = "высокое разнообразие"
            reasons.append("среднее Tanimoto-сходство низкое и кластеров много")
        else:
            status = "умеренное разнообразие"
            reasons.append("структуры образуют несколько областей без экстремального сходства")
    else:
        status = "не рассчитано"

    if largest_pct >= 45.0 and singleton_clusters >= max(3, int(n_clusters * 0.25)):
        status = "неоднородный датасет"
        reasons.append("есть крупный кластер и заметная доля одиночных кластеров")
    elif very_close > 0:
        reasons.append("есть почти дублирующиеся или очень близкие пары")

    return status, "; ".join(reasons) if reasons else "явных причин не выделено"


def _status_code_from_status(status: str) -> str:
    text = str(status or "").strip()
    status_codes = {
        "низкое разнообразие": "LOW_DIVERSITY",
        "неоднородный датасет": "HETEROGENEOUS_DATASET",
        "высокое разнообразие": "HIGH_DIVERSITY",
        "умеренное разнообразие": "MODERATE_DIVERSITY",
        "не рассчитано": "NOT_CALCULATED",
        "недостаточно данных": "INSUFFICIENT_DATA",
        "LOW_DIVERSITY": "LOW_DIVERSITY",
        "HETEROGENEOUS_DATASET": "HETEROGENEOUS_DATASET",
        "HIGH_DIVERSITY": "HIGH_DIVERSITY",
        "MODERATE_DIVERSITY": "MODERATE_DIVERSITY",
        "NOT_CALCULATED": "NOT_CALCULATED",
        "INSUFFICIENT_DATA": "INSUFFICIENT_DATA",
    }
    if text in status_codes:
        return status_codes[text]
    if text in {"РЅРёР·РєРѕРµ СЂР°Р·РЅРѕРѕР±СЂР°Р·РёРµ", "LOW_DIVERSITY"}:
        return "LOW_DIVERSITY"
    if text in {"РЅРµРѕРґРЅРѕСЂРѕРґРЅС‹Р№ РґР°С‚Р°СЃРµС‚", "HETEROGENEOUS_DATASET"}:
        return "HETEROGENEOUS_DATASET"
    if text in {"РІС‹СЃРѕРєРѕРµ СЂР°Р·РЅРѕРѕР±СЂР°Р·РёРµ", "HIGH_DIVERSITY"}:
        return "HIGH_DIVERSITY"
    if text in {"СѓРјРµСЂРµРЅРЅРѕРµ СЂР°Р·РЅРѕРѕР±СЂР°Р·РёРµ", "MODERATE_DIVERSITY"}:
        return "MODERATE_DIVERSITY"
    return "NOT_CALCULATED"


def _descriptor_space_diagnostics(
    descriptor_df: Optional[pd.DataFrame],
    max_rows: int = 5000,
) -> dict:
    if descriptor_df is None or not isinstance(descriptor_df, pd.DataFrame) or descriptor_df.empty:
        return {}
    if StandardScaler is None or NearestNeighbors is None:
        return {"status": "sklearn недоступен"}

    numeric = descriptor_df.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.dropna(axis=1, how="all")
    if numeric.empty or len(numeric) < 2:
        return {"status": "недостаточно числовых дескрипторов"}

    numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0.0)
    if len(numeric) > int(max_rows):
        numeric = numeric.sample(n=int(max_rows), random_state=42)

    values = StandardScaler().fit_transform(numeric.values.astype(float))
    k = 2 if len(values) > 1 else 1
    distances = NearestNeighbors(n_neighbors=k).fit(values).kneighbors(values)[0][:, -1]
    result = {
        "n_descriptor_rows": int(len(numeric)),
        "n_descriptor_columns": int(numeric.shape[1]),
        "median_nearest_distance": float(np.median(distances)),
        "mean_nearest_distance": float(np.mean(distances)),
        "max_nearest_distance": float(np.max(distances)),
    }

    if PCA is not None and numeric.shape[1] >= 2 and len(numeric) >= 3:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(values)
        result["pca_explained_variance_1"] = float(pca.explained_variance_ratio_[0])
        result["pca_explained_variance_2"] = float(pca.explained_variance_ratio_[1])
        result["pca_coordinates"] = pd.DataFrame({
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
        })

    return result


def analyze_chemical_diversity(
    data: pd.DataFrame,
    smiles_col: str,
    label_col: Optional[str] = None,
    target_col: Optional[str] = None,
    descriptor_df: Optional[pd.DataFrame] = None,
    radius: int = 2,
    n_bits: int = 2048,
    duplicate_threshold: float = 0.95,
    analogue_threshold: float = 0.85,
    unique_threshold: float = 0.30,
    cluster_similarity_threshold: float = 0.60,
    projection_method: str = "auto",
    fingerprint_structure_source: str = "standardized_parent",
    map_edge_threshold: float = 0.75,
    map_edge_top_k: int = 5,
    max_full_molecules: int = 2000,
    max_sample_pairs: int = 200000,
    max_table_pairs: int = 500,
    max_heatmap_molecules: int = 300,
    max_network_edges: int = 500,
    top_n_pairs: int = 30,
    top_n_unique: int = 30,
    random_state: int = 42,
) -> dict:
    """Analyze fingerprint and optional descriptor-space diversity."""
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Датасет пуст.")
    if smiles_col not in data.columns:
        raise ValueError(f"Колонка SMILES `{smiles_col}` не найдена.")

    records, invalid_df = _build_records(
        data=data,
        smiles_col=smiles_col,
        label_col=label_col,
        radius=radius,
        n_bits=n_bits,
        fingerprint_structure_source=fingerprint_structure_source,
    )
    n = len(records)
    total_pairs = n * (n - 1) // 2

    if n < 2:
        summary = {
            "total_rows": int(len(data)),
            "valid_structures": int(n),
            "invalid_structures": int(len(invalid_df)),
            "pairwise_mode": "not_enough_data",
            "total_pairs": int(total_pairs),
        }
        status, reasons = _status_from_summary(summary)
        summary["status"] = status
        summary["status_code"] = _status_code_from_status(status)
        summary["status_reasons"] = reasons
        return {
            "status": "success",
            "errors": [],
            "warnings": [],
            "summary": summary,
            "fingerprint_config": {
                "structure_source": str(fingerprint_structure_source),
                "fingerprint_type": "Morgan bit fingerprint",
                "radius": int(radius),
                "n_bits": int(n_bits),
                "duplicate_threshold": float(duplicate_threshold),
                "analogue_threshold": float(analogue_threshold),
                "cluster_similarity_threshold": float(cluster_similarity_threshold),
                "threshold_note": "Tanimoto thresholds depend on fingerprint settings and chemistry.",
            },
            "invalid_structures": invalid_df,
            "similarity_histogram": pd.DataFrame(),
            "top_similar_pairs": pd.DataFrame(),
            "unique_molecules": pd.DataFrame(),
            "cluster_summary": pd.DataFrame(),
            "cluster_assignments": pd.DataFrame(),
            "cluster_threshold_sensitivity": pd.DataFrame(),
            "fingerprint_pca": pd.DataFrame(),
            "similarity_heatmap": {"matrix": pd.DataFrame(), "molecules": pd.DataFrame(), "sampled": False},
            "duplicate_pairs": pd.DataFrame(),
            "analogue_pairs": pd.DataFrame(),
            "network_edges": pd.DataFrame(),
            "final_chemical_space": {
                "map": pd.DataFrame(),
                "edges": pd.DataFrame(),
                "nearest_neighbors": pd.DataFrame(),
                "duplicates": pd.DataFrame(),
                "projection_method": "",
                "n_components": 0,
                "largest_component_size": 0,
                "sampled": False,
            },
            "descriptor_space": _descriptor_space_diagnostics(descriptor_df),
        }

    if n <= int(max_full_molecules):
        pairwise_mode = "full"
        sims, top_heap, max_similarity, nearest_index = _pairwise_full(records, int(top_n_pairs))
        pair_sample_fraction = 1.0
        n_pairs_used = int(total_pairs)
    else:
        pairwise_mode = "sampled"
        sims, top_heap, max_similarity, nearest_index = _pairwise_sample(
            records,
            int(top_n_pairs),
            int(max_sample_pairs),
            int(random_state),
        )
        n_pairs_used = int(len(sims))
        pair_sample_fraction = n_pairs_used / max(total_pairs, 1)

    pairs_gt_dup = int(np.sum(sims > float(duplicate_threshold)))
    pairs_gt_analogue = int(np.sum(sims > float(analogue_threshold)))
    if pairwise_mode == "sampled":
        mean_ci_low, mean_ci_high = _bootstrap_mean_interval(sims, random_state=int(random_state))
    else:
        mean_ci_low, mean_ci_high = np.nan, np.nan
    if pairwise_mode == "sampled" and pair_sample_fraction > 0:
        pairs_gt_dup_est = int(round(pairs_gt_dup / pair_sample_fraction))
        pairs_gt_analogue_est = int(round(pairs_gt_analogue / pair_sample_fraction))
    else:
        pairs_gt_dup_est = pairs_gt_dup
        pairs_gt_analogue_est = pairs_gt_analogue

    cluster_sampled = n > int(max_full_molecules)
    cluster_records = records
    if cluster_sampled:
        rng = np.random.default_rng(int(random_state))
        sample_idx = sorted(rng.choice(np.arange(n), size=int(max_full_molecules), replace=False).tolist())
        cluster_records = [records[i] for i in sample_idx]

    cluster_summary, cluster_assignments = _cluster_records(
        cluster_records,
        similarity_threshold=float(cluster_similarity_threshold),
    )
    duplicate_pairs, analogue_pairs, network_edges = _threshold_pair_tables(
        records,
        duplicate_threshold=float(duplicate_threshold),
        analogue_threshold=float(analogue_threshold),
        max_rows=max(int(max_table_pairs), int(max_network_edges)),
        max_full_molecules=int(max_full_molecules),
        max_sample_pairs=int(max_sample_pairs),
        random_state=int(random_state),
    )
    if len(duplicate_pairs) > int(max_table_pairs):
        duplicate_pairs = duplicate_pairs.head(int(max_table_pairs)).copy()
    if len(analogue_pairs) > int(max_table_pairs):
        analogue_pairs = analogue_pairs.head(int(max_table_pairs)).copy()
    if len(network_edges) > int(max_network_edges):
        network_edges = network_edges.head(int(max_network_edges)).copy()

    n_clusters = int(len(cluster_summary))
    largest_cluster = int(cluster_summary["n"].max()) if not cluster_summary.empty else 0
    largest_pct = float(cluster_summary["percent"].max()) if not cluster_summary.empty else 0.0
    singleton_clusters = int((cluster_summary["n"] == 1).sum()) if not cluster_summary.empty else 0

    summary = {
        "total_rows": int(len(data)),
        "valid_structures": int(n),
        "invalid_structures": int(len(invalid_df)),
        "pairwise_mode": pairwise_mode,
        "total_pairs": int(total_pairs),
        "pairs_used": n_pairs_used,
        "pair_sample_fraction": float(pair_sample_fraction),
        "random_seed": int(random_state),
        "mean_tanimoto_bootstrap_ci95_low": float(mean_ci_low) if np.isfinite(mean_ci_low) else np.nan,
        "mean_tanimoto_bootstrap_ci95_high": float(mean_ci_high) if np.isfinite(mean_ci_high) else np.nan,
        "mean_tanimoto": float(np.mean(sims)) if len(sims) else np.nan,
        "median_tanimoto": float(np.median(sims)) if len(sims) else np.nan,
        "min_tanimoto": float(np.min(sims)) if len(sims) else np.nan,
        "max_tanimoto": float(np.max(sims)) if len(sims) else np.nan,
        "pairs_gt_0_95": pairs_gt_dup_est,
        "pairs_gt_0_85": pairs_gt_analogue_est,
        "pairs_gt_0_95_observed": pairs_gt_dup,
        "pairs_gt_0_85_observed": pairs_gt_analogue,
        "unique_molecules_lt_0_30": int(np.sum(max_similarity < float(unique_threshold))),
        "n_clusters": n_clusters,
        "largest_cluster_size": largest_cluster,
        "largest_cluster_percent": largest_pct,
        "singleton_clusters": singleton_clusters,
        "cluster_sampled": bool(cluster_sampled),
        "cluster_similarity_threshold": float(cluster_similarity_threshold),
        "fingerprint_structure_source": str(fingerprint_structure_source),
        "fingerprint_type": "Morgan bit fingerprint",
        "fingerprint_radius": int(radius),
        "fingerprint_bits": int(n_bits),
        "tanimoto_threshold_note": "Thresholds are fingerprint-configuration dependent, not universal.",
        "analogue_threshold": float(analogue_threshold),
        "duplicate_threshold": float(duplicate_threshold),
        "full_similarity_matrix_estimated_mb": float((n ** 2 * 8) / (1024 ** 2)),
        "large_dataset_mode": bool(n > int(max_full_molecules)),
        "large_dataset_note": (
            "Final map uses a sampled subset to avoid full NxN matrix growth."
            if n > int(max_full_molecules)
            else "Final map uses the full valid set."
        ),
    }
    status, reasons = _status_from_summary(summary)
    summary["status"] = status
    summary["status_code"] = _status_code_from_status(status)
    summary["status_reasons"] = reasons

    if n <= int(max_full_molecules):
        final_records = records
        final_similarity_matrix = _similarity_matrix_for_records(final_records)
        final_sampled = False
    else:
        rng = np.random.default_rng(int(random_state))
        sample_idx = sorted(rng.choice(np.arange(n), size=int(max_full_molecules), replace=False).tolist())
        final_records = [records[i] for i in sample_idx]
        final_similarity_matrix = _similarity_matrix_for_records(final_records)
        final_sampled = True
    final_space = _final_chemical_space(
        records=final_records,
        data=data,
        target_col=target_col,
        similarity_matrix=final_similarity_matrix,
        projection_method=projection_method,
        edge_threshold=float(map_edge_threshold),
        edge_top_k=int(map_edge_top_k),
        duplicate_threshold=float(duplicate_threshold),
        analogue_threshold=float(analogue_threshold),
        random_state=int(random_state),
        max_edges=int(max_network_edges),
        max_table_pairs=int(max_table_pairs),
    )
    final_space["sampled"] = bool(final_sampled)
    final_space["total_valid_structures"] = int(n)
    final_space["displayed_structures"] = int(len(final_records))
    final_classes = final_space.get("map", pd.DataFrame())
    if isinstance(final_classes, pd.DataFrame) and not final_classes.empty:
        class_counts = final_classes["csa_class"].value_counts()
        summary["csa_dense_area"] = int(class_counts.get(CSA_DENSE_LABEL, 0))
        summary["csa_moderate_area"] = int(class_counts.get(CSA_MODERATE_LABEL, 0))
        summary["csa_sparse_area"] = int(class_counts.get(CSA_SPARSE_LABEL, 0))
        summary["csa_singleton_outlier"] = int(class_counts.get(CSA_ISOLATED_LABEL, 0))
        summary["csa_connected_components"] = int(final_space.get("n_components", 0) or 0)
        summary["csa_largest_component_size"] = int(final_space.get("largest_component_size", 0) or 0)
        summary["csa_exact_duplicates"] = int(
            (final_space.get("duplicates", pd.DataFrame()).get("duplicate_type") == "exact duplicate").sum()
        ) if isinstance(final_space.get("duplicates"), pd.DataFrame) and not final_space.get("duplicates").empty else 0
        summary["csa_near_duplicates"] = int(
            (final_space.get("duplicates", pd.DataFrame()).get("duplicate_type") == "near duplicate").sum()
        ) if isinstance(final_space.get("duplicates"), pd.DataFrame) and not final_space.get("duplicates").empty else 0

    hist_counts, hist_edges = np.histogram(sims, bins=np.linspace(0.0, 1.0, 21))
    hist_df = pd.DataFrame({
        "similarity_from": hist_edges[:-1],
        "similarity_to": hist_edges[1:],
        "count": hist_counts,
    })

    return {
        "status": "success",
        "errors": [],
        "warnings": [],
        "algorithm_version": CHEMICAL_SPACE_ALGORITHM_VERSION,
        "fingerprint_config": {
            "structure_source": str(fingerprint_structure_source),
            "fingerprint_type": "Morgan bit fingerprint",
            "radius": int(radius),
            "n_bits": int(n_bits),
            "duplicate_threshold": float(duplicate_threshold),
            "analogue_threshold": float(analogue_threshold),
            "cluster_similarity_threshold": float(cluster_similarity_threshold),
            "threshold_note": "Tanimoto thresholds depend on fingerprint settings and chemistry.",
        },
        "summary": summary,
        "invalid_structures": invalid_df,
        "similarity_histogram": hist_df,
        "top_similar_pairs": _top_pairs_table(records, top_heap),
        "duplicate_pairs": duplicate_pairs,
        "analogue_pairs": analogue_pairs,
        "unique_molecules": _unique_table(records, max_similarity, nearest_index, int(top_n_unique)),
        "cluster_summary": cluster_summary,
        "cluster_assignments": cluster_assignments,
        "cluster_threshold_sensitivity": _cluster_threshold_sensitivity(cluster_records),
        "fingerprint_pca": _fingerprint_pca_map(
            records=records,
            cluster_assignments=cluster_assignments,
            data=data,
            target_col=target_col,
            n_bits=int(n_bits),
        ),
        "similarity_heatmap": _heatmap_payload(
            records=records,
            cluster_assignments=cluster_assignments,
            max_molecules=int(max_heatmap_molecules),
            random_state=int(random_state),
        ),
        "network_edges": network_edges,
        "final_chemical_space": final_space,
        "descriptor_space": _descriptor_space_diagnostics(descriptor_df),
    }
