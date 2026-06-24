# -*- coding: utf-8 -*-

"""
saod2_core.py

Ядро Universal SAOD v3 для GenQSPR:
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

import re
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from sklearn.linear_model import LinearRegression


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


def saod2_robust_sigma(values):
    """
    Робастная оценка sigma через MAD с fallback.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 1e-6

    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    sigma = 1.4826 * mad

    if not np.isfinite(sigma) or sigma <= 1e-12:
        sigma = np.nanstd(values)

    if not np.isfinite(sigma) or sigma <= 1e-12:
        q75, q25 = np.nanpercentile(values, [75, 25])
        iqr = q75 - q25
        sigma = iqr / 1.349 if iqr > 1e-12 else 1e-6

    return sigma


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
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())

        if mol is None:
            return None, "", "", "invalid_smiles"

        try:
            chooser = rdMolStandardize.LargestFragmentChooser()
            mol = chooser.choose(mol)
        except Exception:
            pass

        try:
            mol = rdMolStandardize.Cleanup(mol)
        except Exception:
            pass

        try:
            uncharger = rdMolStandardize.Uncharger()
            mol = uncharger.uncharge(mol)
        except Exception:
            pass

        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

        try:
            inchikey = Chem.MolToInchiKey(mol)
        except Exception:
            inchikey = ""

        return mol, canonical_smiles, inchikey, "ok"

    except Exception as e:
        return None, "", "", f"standardization_error: {e}"


def saod2_is_hydrocarbon(mol):
    """
    Проверяет, состоит ли молекула только из C и H.
    """
    return all(atom.GetAtomicNum() in [1, 6] for atom in mol.GetAtoms())


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
        queue = [(start, [start])]
        best_node = start
        best_path = [start]

        while queue:
            node, path = queue.pop(0)

            if len(path) > len(best_path):
                best_node = node
                best_path = path

            for nb in graph.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))

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

                norm_pos = min(pos_idx, chain_len + 1 - pos_idx)

                substituents.append({
                    "position": norm_pos,
                    "type": sub_type,
                    "size": size
                })

    substituents = sorted(
        substituents,
        key=lambda x: (x["position"], x["type"], x["size"])
    )

    return substituents


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
        "substituent_summary": ""
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
    main_chain = saod2_longest_path(graph)
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
        "substituent_summary": exact_pattern if not is_n_alkane else ""
    })

    return result


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


def saod2_detect_functional_class(mol):
    """
    Conservative single-class detector. Multifunctional or unsupported
    compounds are returned as unsupported instead of being forced into a row.
    """
    atoms = list(mol.GetAtoms())
    atomic_numbers = [atom.GetAtomicNum() for atom in atoms]
    ring_count = mol.GetRingInfo().NumRings()

    carboxyl = mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[OX2H1]"))
    aldehyde = mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3H1](=O)[#6]"))
    ketone = mol.GetSubstructMatches(Chem.MolFromSmarts("[#6][CX3](=O)[#6]"))
    alcohol = mol.GetSubstructMatches(Chem.MolFromSmarts("[#6;!$(C=O)][OX2H1]"))
    amine = mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H0,H1,H2;!$(N-C=O)]"))
    carbon_double = [
        bond for bond in mol.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetBeginAtom().GetAtomicNum() == 6
        and bond.GetEndAtom().GetAtomicNum() == 6
    ]
    halogens = [
        atom for atom in atoms
        if atom.GetAtomicNum() in [9, 17, 35, 53]
    ]

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

    features = [
        bool(carboxyl),
        bool(aldehyde),
        bool(ketone),
        bool(alcohol),
        bool(amine),
        bool(carbon_double),
        bool(halogens),
    ]

    if len(carboxyl) == 1 and sum(features) == 1 and ring_count == 0:
        return "carboxylic_acid", {"match": carboxyl[0]}
    if len(aldehyde) == 1 and sum(features) == 1 and ring_count == 0:
        return "aldehyde", {"match": aldehyde[0]}
    if len(ketone) == 1 and sum(features) == 1 and ring_count == 0:
        return "ketone", {"match": ketone[0]}
    if len(alcohol) == 1 and sum(features) == 1 and ring_count == 0:
        return "alcohol", {"match": alcohol[0]}
    if len(amine) == 1 and sum(features) == 1 and ring_count == 0:
        return "amine", {"match": amine[0]}
    if len(carbon_double) == 1 and sum(features) == 1 and ring_count == 0:
        return "alkene", {"bond": carbon_double[0]}
    if len(halogens) == 1 and sum(features) == 1 and ring_count == 0:
        return "haloalkane", {"atom": halogens[0]}

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
                "ring": benzene_rings[0],
                "attachment": attachments[0],
            }

    if (
        all(number in [1, 6] for number in atomic_numbers)
        and ring_count == 0
        and not carbon_double
        and all(bond.GetBondType() == Chem.BondType.SINGLE for bond in mol.GetBonds())
    ):
        return "alkane", {}

    return "unsupported", {}


def saod2_general_series_passport(mol):
    result = {
        "chemical_class": "unsupported",
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

    carbon_count = sum(
        atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()
    )
    result["series_coordinate"] = carbon_count

    if chemical_class == "unsupported":
        result["series_reason"] = (
            "unsupported, cyclic, or multifunctional structure"
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
        result.update({
            "series_pattern": "alkylbenzene|mono-n-alkyl",
            "series_family": "alkylbenzene",
            "series_complexity": 0,
            "series_supported": True,
            "series_reason": "monosubstituted alkylbenzene",
        })
        return result

    graph = saod2_carbon_graph(mol)
    main_chain = saod2_longest_path(graph)
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
    })
    return result


# ------------------------------------------------------------------
# Prepare structures

def saod2_prepare_structures(input_df, smiles_col, property_col):
    """
    Главная подготовка структур для SAOD v2.
    """
    df = input_df.copy()
    df.columns = df.columns.str.strip()

    if smiles_col not in df.columns:
        raise ValueError(f"Не найдена колонка SMILES: {smiles_col}")

    if property_col not in df.columns:
        raise ValueError(f"Не найдена колонка свойства: {property_col}")

    if "compound_id" not in df.columns:
        df["compound_id"] = df.index.astype(str)

    if "name" not in df.columns:
        df["name"] = ""

    df["property_value"] = saod2_to_numeric(df[property_col])

    rows = []

    for idx, row in df.iterrows():
        smiles = row.get(smiles_col, "")

        mol, canonical_smiles, inchikey, structure_status = saod2_standardize_smiles(
            smiles
        )

        base = {
            "row_index": idx,
            "input_smiles": smiles,
            "canonical_smiles": canonical_smiles,
            "inchikey": inchikey,
            "structure_status": structure_status,
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
            desc.drop(columns=["row_index"], errors="ignore")
        ],
        axis=1
    )

    out["duplicate_structure"] = False
    out["duplicate_conflict"] = False

    valid = out["canonical_smiles"].astype(str).str.len() > 0

    if valid.any():
        counts = (
            out.loc[valid]
            .groupby("canonical_smiles")["canonical_smiles"]
            .transform("count")
        )

        out.loc[valid, "duplicate_structure"] = counts > 1

        for smi, group in out.loc[valid].groupby("canonical_smiles"):
            vals = group["property_value"].dropna().astype(float).values

            if len(vals) > 1:
                if np.nanmax(vals) - np.nanmin(vals) > 1e-9:
                    out.loc[group.index, "duplicate_conflict"] = True

    return out


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
        .groupby("exact_pattern")["exact_pattern"]
        .transform("count")
    )

    alk["formula_group_size"] = (
        alk
        .groupby("molecular_formula")["molecular_formula"]
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

        if bool(row.get("duplicate_conflict", False)):
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

    for formula, g in alk.groupby("molecular_formula"):
        if len(g) < 2:
            continue

        records = g.to_dict("records")

        for a, b in itertools.combinations(records, 2):
            pattern_a = a["exact_pattern"]
            pattern_b = b["exact_pattern"]

            if pattern_a == pattern_b:
                continue

            # Фиксируем порядок паттернов, чтобы edge_label был одинаковым
            # по всем формулам.
            if pattern_a > pattern_b:
                a, b = b, a
                pattern_a = a["exact_pattern"]
                pattern_b = b["exact_pattern"]

            delta = a["property_value"] - b["property_value"]

            try:
                c_count = int(a["carbon_count"])
            except Exception:
                c_count = np.nan

            edge_label = f"{pattern_a} minus {pattern_b}"

            rows.append({
                "formula": formula,
                "carbon_count": c_count,
                "edge_label": edge_label,

                "pattern_a": pattern_a,
                "pattern_b": pattern_b,

                "family_a": saod2_pattern_family(pattern_a),
                "family_b": saod2_pattern_family(pattern_b),

                "compound_a_id": a.get("compound_id", ""),
                "compound_b_id": b.get("compound_id", ""),

                "name_a": a.get("name", ""),
                "name_b": b.get("name", ""),

                "smiles_a": a.get("canonical_smiles", ""),
                "smiles_b": b.get("canonical_smiles", ""),

                "value_a": a.get("property_value", np.nan),
                "value_b": b.get("property_value", np.nan),

                "delta_a_minus_b": delta,

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

            "mean_delta_within_formula": mean_delta,
            "min_delta_within_formula": min_delta,
            "max_delta_within_formula": max_delta,
            "delta_spread_within_formula": delta_spread,
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

def saod2_fit_edge_trend(x, y):
    """
    Тренд для ряда разностей.

    Если точек мало — медиана.
    Если точек 5+ — линейная модель.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(y) < 5:
        pred = np.repeat(np.nanmedian(y), len(y))
        method = "median"

    else:
        try:
            model = LinearRegression()
            model.fit(x.reshape(-1, 1), y)
            pred = model.predict(x.reshape(-1, 1))
            method = "linear"
        except Exception:
            pred = np.repeat(np.nanmedian(y), len(y))
            method = "median_fallback"

    return pred, method


def saod2_discover_rules(edge_table, min_points=3):
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

    for edge_label, g in edge_table.groupby("edge_label"):
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

        pred, trend_method = saod2_fit_edge_trend(x, y)

        residual = y - pred
        residual_sigma = saod2_robust_sigma(residual)
        residual_scores = np.abs(residual) / residual_sigma

        delta_change = np.full(len(y), np.nan)

        for i in range(1, len(y)):
            delta_change[i] = y[i] - y[i - 1]

        valid_changes = delta_change[np.isfinite(delta_change)]

        if len(valid_changes) >= 2:
            change_med = np.nanmedian(valid_changes)
            change_sigma = saod2_robust_sigma(valid_changes - change_med)
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
            "trend_method": trend_method,
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


def saod2_compound_suspicion(edge_table, edge_detail_table, broken_edges):
    """
    Определяет вещества, вокруг которых концентрируются поломки.
    """
    if edge_table.empty:
        return pd.DataFrame()

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
            status = "почти непроверяемое"
        elif info["edges_broken"] == 0:
            status = "согласовано"
        elif info["edges_broken"] >= 3 or info["max_edge_score"] >= 6:
            status = "критически подозрительно"
        elif info["edges_broken"] >= 2 or info["max_edge_score"] >= 4:
            status = "сильно подозрительно"
        else:
            status = "требует проверки"

        if status == "почти непроверяемое":
            recommendation = "Недостаточно структурных связей для проверки."
        elif status == "согласовано":
            recommendation = "Значение согласовано с найденными правилами."
        else:
            recommendation = (
                "Проверить экспериментальное значение, единицы измерения, "
                "структуру, название вещества и источник данных."
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

def saod2_dataset_summary(processed, checkability, rules, broken_edges, suspicion):
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
        if "checkability_level" in checkability.columns:
            rows.append({
                "Показатель": "Почти непроверяемых веществ",
                "Значение": int(
                    (checkability["checkability_level"] == "почти непроверяемое").sum()
                )
            })

            rows.append({
                "Показатель": "Хорошо проверяемых веществ",
                "Значение": int(
                    (checkability["checkability_level"] == "хорошо проверяемое").sum()
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
        if "final_status" in suspicion.columns:
            rows.append({
                "Показатель": "Критически подозрительных веществ",
                "Значение": int(
                    (suspicion["final_status"] == "критически подозрительно").sum()
                )
            })

            rows.append({
                "Показатель": "Сильно подозрительных веществ",
                "Значение": int(
                    (suspicion["final_status"] == "сильно подозрительно").sum()
                )
            })

    return pd.DataFrame(rows)


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

    if heavy_count == 1 and len(halogens) == 1:
        family = "halogen"
        coordinate = float({9: 1, 17: 2, 35: 3, 53: 4}[halogens[0]])
        coordinate_type = "halogen_order"
    elif ring_count == 0 and not unsaturated:
        family = "acyclic:" + ",".join(elements)
        coordinate = float(carbon_count)
        coordinate_type = "carbon_count"
    else:
        family = f"unordered:{','.join(elements)}:rings{ring_count}"
        coordinate = np.nan
        coordinate_type = "unordered"

    return {
        "variable_family": family,
        "variable_coordinate": coordinate,
        "coordinate_type": coordinate_type,
        "variable_carbon_count": carbon_count,
        "variable_heavy_count": heavy_count,
        "variable_elements": ",".join(elements),
        "variable_ring_count": ring_count,
        "variable_unsaturated": unsaturated,
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
            if total_heavy and core_heavy / total_heavy < 0.45:
                continue

            core_smiles = Chem.MolToSmiles(core, canonical=True)
            variable_smiles = Chem.MolToSmiles(variable, canonical=True)
            descriptor = saod3_fragment_descriptor(variable_smiles)
            if descriptor is None:
                continue

            candidates.append({
                "core_smiles": core_smiles,
                "variable_smiles": variable_smiles,
                "core_heavy_atoms": core_heavy,
                "variable_heavy_atoms": variable_heavy,
                **descriptor,
            })

    unique = {}
    for item in candidates:
        key = (item["core_smiles"], item["variable_smiles"])
        unique[key] = item
    return list(unique.values())


def saod3_discover_series_memberships(processed, min_series_points=3):
    rows = []

    # Profile-derived homologous rows preserve strong chemistry where it can
    # be inferred unambiguously, but use the same downstream series engine.
    for _, row in processed.iterrows():
        if (
            bool(row.get("series_supported", False))
            and pd.notna(row.get("property_value", np.nan))
        ):
            rows.append({
                "series_source": "graph_profile",
                "series_id": f"profile::{row['series_pattern']}",
                "series_domain": str(row.get("series_family", "")),
                "core_smiles": "",
                "variable_smiles": "",
                "variable_family": str(row.get("series_family", "")),
                "coordinate_type": "total_carbon_count",
                "series_coordinate": float(row["series_coordinate"]),
                "series_complexity": float(
                    row.get("series_complexity", 0)
                ),
                "compound_id": str(row.get("compound_id", "")),
                "name": str(row.get("name", "")),
                "canonical_smiles": str(
                    row.get("canonical_smiles", "")
                ),
                "molecular_formula": str(
                    row.get("molecular_formula", "")
                ),
                "property_value": float(row["property_value"]),
            })

    fragment_rows = []
    for _, row in processed.iterrows():
        if (
            not bool(row.get("valid_structure", False))
            or pd.isna(row.get("property_value", np.nan))
        ):
            continue
        mol = Chem.MolFromSmiles(str(row.get("canonical_smiles", "")))
        if mol is None:
            continue
        for fragment in saod3_single_cut_fragments(mol):
            fragment_rows.append({
                **fragment,
                "compound_id": str(row.get("compound_id", "")),
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
        fragments = fragments.drop_duplicates(
            ["core_smiles", "variable_smiles", "compound_id"]
        )

        core_sizes = (
            fragments.groupby("core_smiles")["compound_id"]
            .nunique()
            .to_dict()
        )
        fragments = fragments[
            fragments["core_smiles"].map(core_sizes)
            >= int(min_series_points)
        ].copy()

        # A core can yield several coherent variable families. Each becomes a
        # series; unordered families remain useful for transformation checks.
        for _, item in fragments.iterrows():
            series_id = (
                f"scaffold::{item['core_smiles']}::"
                f"{item['variable_family']}"
            )
            rows.append({
                "series_source": "automatic_scaffold",
                "series_id": series_id,
                "series_domain": (
                    f"scaffold::{item['core_smiles']}::"
                    f"{item['coordinate_type']}"
                ),
                "core_smiles": item["core_smiles"],
                "variable_smiles": item["variable_smiles"],
                "variable_family": item["variable_family"],
                "coordinate_type": item["coordinate_type"],
                "series_coordinate": item["variable_coordinate"],
                "series_complexity": float(item["core_heavy_atoms"]),
                "compound_id": item["compound_id"],
                "name": item["name"],
                "canonical_smiles": item["canonical_smiles"],
                "molecular_formula": item["molecular_formula"],
                "property_value": item["property_value"],
            })

    memberships = pd.DataFrame(rows)
    if memberships.empty:
        return memberships

    memberships = memberships.drop_duplicates(
        ["series_id", "compound_id", "variable_smiles"]
    ).reset_index(drop=True)

    counts = (
        memberships.groupby("series_id")["compound_id"]
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


def saod3_local_expected(x, y, index):
    if len(y) < 3:
        return np.nan, "insufficient"
    if 0 < index < len(y) - 1:
        x0, x1 = x[index - 1], x[index + 1]
        y0, y1 = y[index - 1], y[index + 1]
    elif index == 0:
        x0, x1 = x[1], x[2]
        y0, y1 = y[1], y[2]
    else:
        x0, x1 = x[-3], x[-2]
        y0, y1 = y[-3], y[-2]
    if abs(x1 - x0) < 1e-12:
        return np.nan, "insufficient"
    expected = y0 + (x[index] - x0) * (y1 - y0) / (x1 - x0)
    return float(expected), (
        "interpolation" if 0 < index < len(y) - 1 else "extrapolation"
    )


def saod3_analyze_own_series(series_points, min_points=3):
    summaries = []
    details = []
    edges = []
    if series_points.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for series_id, group in series_points.groupby("series_id"):
        group = group.sort_values("series_coordinate").copy()
        x = group["series_coordinate"].astype(float).values
        y = group["property_value"].astype(float).values
        if len(np.unique(x)) < min_points:
            continue

        step = np.diff(x)
        delta = np.diff(y)
        normalized_delta = delta / step
        median_step = float(np.nanmedian(normalized_delta))
        delta_sigma = max(
            saod2_robust_sigma(normalized_delta - median_step),
            max(float(np.ptp(y)) * 0.02, 1e-6),
        )
        delta_scores = np.abs(
            normalized_delta - median_step
        ) / delta_sigma
        signs = np.sign(normalized_delta)
        main_sign = np.sign(median_step)
        sign_consistency = (
            float(np.mean(signs[signs != 0] == main_sign))
            if np.any(signs != 0) and main_sign != 0
            else 0.0
        )

        residual_values = []
        local_rows = []
        for index, (_, row) in enumerate(group.iterrows()):
            expected, method = saod3_local_expected(x, y, index)
            residual = y[index] - expected if np.isfinite(expected) else np.nan
            residual_values.append(residual)
            local_rows.append({
                **row.to_dict(),
                "expected_from_own_series": expected,
                "own_series_residual": residual,
                "prediction_method": method,
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
            adjacent_scores = []
            if point_index > 0:
                adjacent_scores.append(delta_scores[point_index - 1])
            if point_index < len(delta_scores):
                adjacent_scores.append(delta_scores[point_index])
            two_sided_break = (
                len(adjacent_scores) == 2
                and all(score >= 3 for score in adjacent_scores)
            )
            score = max(
                [value for value in [residual_score, *adjacent_scores]
                 if np.isfinite(value)],
                default=np.nan,
            )
            row["own_series_score"] = score
            row["own_series_status"] = (
                "нарушает собственный ряд"
                if (
                    (np.isfinite(residual_score) and residual_score >= 3)
                    or two_sided_break
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
                "normalized_delta": normalized_delta[index - 1],
                "expected_normalized_delta": median_step,
                "delta_score": score,
                "edge_status": (
                    "нарушение шага ряда" if score >= 3 else "согласовано"
                ),
            })

        internally_consistent = (
            len(group) >= min_points
            and sign_consistency >= 0.75
            and broken_points == 0
        )
        summaries.append({
            "series_source": group.iloc[0]["series_source"],
            "series_id": series_id,
            "series_domain": group.iloc[0]["series_domain"],
            "coordinate_type": group.iloc[0]["coordinate_type"],
            "series_complexity": float(
                group["series_complexity"].median()
            ),
            "n_points": int(len(group)),
            "coordinate_min": float(np.min(x)),
            "coordinate_max": float(np.max(x)),
            "median_property_step": median_step,
            "step_sign_consistency": sign_consistency,
            "broken_own_points": broken_points,
            "max_own_series_score": max_score,
            "internally_consistent": internally_consistent,
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
    for domain, domain_points in series_points.groupby("series_domain"):
        candidates = []
        for series_id, group in domain_points.groupby("series_id"):
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

    for series_id, group in scaffold.groupby("series_id"):
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
        "transformation_id"
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
):
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
        if independent_types >= 3 or (
            item["own_series_breaks"] > 0
            and item["reference_breaks"] >= 2
        ):
            status = "критически подозрительно"
        elif independent_types >= 2 or total >= 3:
            status = "сильно подозрительно"
        else:
            status = "требует проверки"
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


def saod3_merge_suspicion(base, hierarchical, processed, checkability):
    identity = processed[[
        "compound_id", "name", "canonical_smiles",
        "molecular_formula", "property_value"
    ]].drop_duplicates("compound_id")

    if base.empty:
        out = identity.copy()
    else:
        out = identity.merge(base, on="compound_id", how="left", suffixes=("", "_old"))

    if not hierarchical.empty:
        out = out.merge(hierarchical, on="compound_id", how="left")
    if not checkability.empty:
        out = out.merge(
            checkability[[
                "compound_id",
                "overall_checkability",
                "trusted_edges_total",
            ]],
            on="compound_id",
            how="left",
        )

    priority = {
        "": -1,
        "согласовано": 0,
        "почти непроверяемое": 0,
        "требует проверки": 1,
        "сильно подозрительно": 2,
        "критически подозрительно": 3,
    }

    def choose_status(row):
        candidates = []
        for column in ["final_status", "hierarchical_status"]:
            value = row.get(column, "")
            if pd.isna(value):
                value = ""
            candidates.append(str(value))
        selected = max(
            candidates,
            key=lambda status: priority.get(status, -1),
        )
        if selected:
            return selected
        if (
            row.get("overall_checkability") in [
                "хорошо проверяемое",
                "умеренно проверяемое",
            ]
            or float(row.get("trusted_edges_total", 0) or 0) > 0
        ):
            return "согласовано"
        return "почти непроверяемое"

    out["final_status"] = out.apply(choose_status, axis=1)
    out["recommendation"] = out["final_status"].map({
        "согласовано": "Значение согласовано с доступными химическими рядами.",
        "почти непроверяемое": "Недостаточно структурных связей для проверки.",
        "требует проверки": "Проверить значение, структуру, единицы и источник.",
        "сильно подозрительно": "Несколько независимых химических проверок указывают на возможную ошибку.",
        "критически подозрительно": "Собственный ряд и перекрёстные проверки согласованно указывают на возможную ошибку.",
    })
    return out


def saod3_universal_checkability(processed, memberships, hierarchy):
    rows = []
    membership_counts = (
        memberships.groupby("compound_id")["series_id"].nunique().to_dict()
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
            "broken_trusted_edges": 0,
            "has_same_pattern_neighbors": n_series > 0,
            "has_formula_isomers": False,
            "has_positional_analogs": n_series > 1,
            "series_checkability": f"найдено рядов: {n_series}",
            "formula_checkability": "универсальная структурная проверка",
            "network_checkability": f"референтных рядов: {n_reference}",
            "overall_checkability": overall,
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


# ------------------------------------------------------------------
# Main runner

def run_saod2_analysis(input_df, smiles_col, property_col, min_rule_points=3):
    """
    Universal SAOD analysis.

    Backward-compatible result keys are preserved. New keys expose automatic
    scaffold series, own-series checks, hierarchical reference validation,
    and repeated R-group transformation checks.
    """
    errors = []
    warnings = []

    try:
        processed = saod2_prepare_structures(
            input_df=input_df,
            smiles_col=smiles_col,
            property_col=property_col
        )

    except Exception as e:
        return {
            "processed": pd.DataFrame(),
            "checkability": pd.DataFrame(),
            "raw_edge_table": pd.DataFrame(),
            "edge_table": pd.DataFrame(),
            "rules": pd.DataFrame(),
            "edge_details": pd.DataFrame(),
            "broken_edges": pd.DataFrame(),
            "suspicion": pd.DataFrame(),
            "summary": pd.DataFrame(),
            "series_memberships": pd.DataFrame(),
            "series_points": pd.DataFrame(),
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
            "errors": [str(e)],
            "warnings": warnings
        }

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

    # Universal automatic series layer.
    series_memberships = saod3_discover_series_memberships(
        processed,
        min_series_points=min_rule_points,
    )
    series_points = saod3_aggregate_series_points(series_memberships)
    (
        own_series_summary,
        own_series_details,
        own_series_edges,
    ) = saod3_analyze_own_series(
        series_points,
        min_points=min_rule_points,
    )
    (
        reference_hierarchy,
        reference_comparisons,
        reference_comparison_details,
    ) = saod3_build_reference_hierarchy(
        series_points,
        own_series_summary,
        min_points=min_rule_points,
    )
    transformation_table = saod3_build_transformation_table(
        series_memberships
    )
    (
        transformation_rules,
        transformation_details,
    ) = saod3_analyze_transformations(
        transformation_table,
        min_contexts=min_rule_points,
    )
    hierarchical_suspicion = saod3_build_hierarchical_suspicion(
        own_series_details,
        reference_comparison_details,
        transformation_details,
    )

    # Original alkane edge layer remains available and supplies additional
    # evidence for existing reports.
    raw_edge_table = saod2_build_edge_table(processed)

    edge_table = saod2_aggregate_edge_table_by_formula(
        raw_edge_table=raw_edge_table
    )

    rules, edge_details = saod2_discover_rules(
        edge_table=edge_table,
        min_points=min_rule_points
    )

    broken_edges = saod2_broken_edges(edge_details)

    base_suspicion = saod2_compound_suspicion(
        edge_table=edge_table,
        edge_detail_table=edge_details,
        broken_edges=broken_edges
    )
    checkability = saod3_universal_checkability(
        processed,
        series_memberships,
        reference_hierarchy,
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
        suspicion=suspicion
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

    return {
        "processed": processed,
        "series_memberships": series_memberships,
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
        "hierarchical_suspicion": hierarchical_suspicion,
        "checkability": checkability,
        "raw_edge_table": raw_edge_table,
        "edge_table": edge_table,
        "rules": rules,
        "edge_details": edge_details,
        "broken_edges": broken_edges,
        "suspicion": suspicion,
        "summary": summary,
        "errors": errors,
        "warnings": warnings
    }
