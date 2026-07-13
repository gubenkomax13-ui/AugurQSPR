# -*- coding: utf-8 -*-

"""
Structural filter core for Augur QSPR.

Функции структурной фильтрации датасета:
- фильтрация по элементам;
- фильтрация по функциональным группам;
- фильтрация по SMARTS;
- ароматичность;
- число атомов углерода;
- число гетероатомов.
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors


FUNCTIONAL_GROUP_SMARTS = {
    "Гидроксил / спиртовая OH": "[OX2H]",
    "Карбонил C=O": "[CX3]=[OX1]",
    "Альдегид": "[CX3H1](=O)[#6,#1]",
    "Кетон": "[#6][CX3](=O)[#6]",
    "Карбоксильная кислота COOH": "C(=O)[OX2H1]",
    "Сложный эфир": "C(=O)O[#6]",
    "Простой эфир C-O-C": "[OD2]([#6])[#6]",
    "Амид": "C(=O)N",
    "Амин": "[NX3;H2,H1,H0;!$(NC=O)]",
    "Нитро-группа": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "Нитрил C≡N": "C#N",
    "Алкен C=C": "C=C",
    "Алкин C≡C": "C#C",
    "Ароматический фрагмент": "a",
    "Бензольное кольцо": "c1ccccc1",
    "Галоген": "[F,Cl,Br,I]",
    "Тиол SH": "[SX2H]",
    "Сульфонил SO2": "S(=O)(=O)",
    "Фосфорсодержащий фрагмент": "P",
}

FUNCTIONAL_GROUP_SMARTS.update({
    "Alcohol OH (aliphatic)": "[OX2H][CX4;!$(C=[O,N,S])]",
    "Phenolic OH": "[OX2H][c]",
    "Carboxylic acid OH": "[OX2H][CX3](=O)",
    "Primary amine": "[NX3H2;!$(NC=O)]",
    "Secondary amine": "[NX3H1;!$(NC=O)]",
    "Tertiary amine": "[NX3H0;!$(NC=O)]",
    "Aromatic amine": "[NX3;!$(NC=O)][c]",
    "Aliphatic amine": "[NX3;!$(NC=O)][CX4]",
})

FUNCTIONAL_GROUP_NOTES = {
    "Alcohol OH (aliphatic)": "Specific aliphatic alcohol OH; phenols and carboxylic acids are separate groups.",
    "Phenolic OH": "Specific aromatic phenolic OH.",
    "Carboxylic acid OH": "Specific OH attached to carboxylic acid carbonyl.",
    "Primary amine": "Specific amine subclass; broad amine matches are not mutually exclusive.",
    "Secondary amine": "Specific amine subclass; broad amine matches are not mutually exclusive.",
    "Tertiary amine": "Specific amine subclass; broad amine matches are not mutually exclusive.",
    "Aromatic amine": "Amine nitrogen attached to an aromatic atom.",
    "Aliphatic amine": "Amine nitrogen attached to an aliphatic sp3 carbon.",
}


def structural_filter_normalize_combine_mode(combine_mode):
    text = str(combine_mode or "").strip().lower()
    if text in {"any", "or"} or "хотя" in text or "at least" in text:
        return "any"
    return "all"


def structural_filter_normalize_match_mode(value):
    text = str(value or "").strip().lower()
    if text in {"all", "and"} or "все" in text or "all selected" in text or "all smarts" in text:
        return "all"
    return "any"


def structural_filter_normalize_aromatic_mode(value):
    text = str(value or "").strip().lower()
    if text in {"aromatic", "only_aromatic"} or "только аромат" in text:
        return "only_aromatic"
    if text in {"non_aromatic", "only_non_aromatic"} or "неаромат" in text:
        return "only_non_aromatic"
    return "any"


def structural_filter_report(work, filtered, condition_names, combine_mode):
    return {
        "total_rows": len(work),
        "valid_structures": int(work["valid_mol"].sum()),
        "rows_after_filter": len(filtered),
        "conditions_applied": len(condition_names),
        "condition_list": ", ".join(condition_names),
        "smiles_validity_required": True,
        "combine_mode": structural_filter_normalize_combine_mode(combine_mode),
    }


def structural_filter_group_metadata_table():
    rows = []
    for name, smarts in FUNCTIONAL_GROUP_SMARTS.items():
        rows.append({
            "group": name,
            "smarts": smarts,
            "note": FUNCTIONAL_GROUP_NOTES.get(
                name,
                "Functional-group SMARTS are substructure matches and are not mutually exclusive.",
            ),
        })
    return pd.DataFrame(rows)


def qspr_guess_descriptor_source(desc_name):
    """
    Грубо определяет источник дескриптора по имени.
    """
    name = str(desc_name)

    if name.startswith("xtb_"):
        return "xTB"

    if name.startswith("SPEC_") or name.startswith("spectral_"):
        return "Spectra"

    if name.startswith("IR_") or name.startswith("Mass_"):
        return "Spectra"

    if name.startswith("MACCS") or name.startswith("PubchemFP") or name.startswith("SubFP"):
        return "PaDEL"

    return "Molecular"


def qspr_make_descriptor_meaning_table(desc_names, status_label=""):
    """
    Делает таблицу:
    дескриптор -> расшифровка -> источник -> статус.
    Использует descriptor_meanings.json через qspr_load_descriptor_meanings().
    """
    desc_names = list(desc_names or [])
    from modules.qspr_core import qspr_load_descriptor_meanings

    meanings = qspr_load_descriptor_meanings()

    rows = []

    for i, desc_name in enumerate(desc_names, start=1):
        desc_str = str(desc_name)

        rows.append({
            "№": i,
            "Дескриптор": desc_str,
            "Расшифровка": meanings.get(desc_str, "Нет расшифровки в descriptor_meanings.json"),
            "Источник": qspr_guess_descriptor_source(desc_str),
            "Статус": status_label,
        })

    return pd.DataFrame(rows)


def qspr_show_descriptor_meaning_table(
    desc_names,
    title="📖 Расшифровка дескрипторов",
    status_label="",
    expanded=False,
    key_prefix="descriptor_meanings"
):
    """
    Показывает таблицу расшифровки дескрипторов и кнопку скачивания CSV.
    """
    import streamlit as st

    desc_names = list(desc_names or [])

    if not desc_names:
        return

    meaning_df = qspr_make_descriptor_meaning_table(
        desc_names=desc_names,
        status_label=status_label
    )

    with st.expander(title, expanded=expanded):
        st.caption(
            "Расшифровка берётся из файла `descriptor_meanings.json`. "
            "Если написано «Нет расшифровки», значит этот дескриптор нужно добавить в JSON."
        )

        st.dataframe(
            meaning_df,
            width="stretch",
            hide_index=True
        )

        st.download_button(
            "📥 Скачать расшифровку дескрипторов CSV",
            meaning_df.to_csv(index=False).encode("utf-8-sig"),
            f"{key_prefix}.csv",
            "text/csv",
            key=f"download_{key_prefix}"
        )

def structural_filter_get_group_options():
    """
    Возвращает список доступных групп для интерфейса.
    """
    return list(FUNCTIONAL_GROUP_SMARTS.keys())


def structural_filter_mol_from_smiles(smiles):
    """
    Безопасно создаёт RDKit Mol из SMILES.
    """
    try:
        smiles = str(smiles).strip()

        if not smiles or smiles.lower() in ["nan", "none"]:
            return None

        return Chem.MolFromSmiles(smiles)

    except Exception:
        return None


def structural_filter_formula_from_mol(mol):
    """Return RDKit molecular formula, including implicit hydrogens."""
    if mol is None:
        return ""

    try:
        return rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        return ""


def structural_filter_validate_custom_smarts(custom_smarts_text):
    rows = []
    for line_no, raw in enumerate(str(custom_smarts_text).splitlines(), start=1):
        smarts = raw.strip()
        if not smarts:
            continue
        patt = Chem.MolFromSmarts(smarts)
        rows.append({
            "line": line_no,
            "SMARTS": smarts,
            "status": "ok" if patt is not None else "syntax_error",
            "message": "" if patt is not None else "RDKit could not parse this SMARTS.",
        })
    return pd.DataFrame(rows)


def structural_filter_analyze_mol(mol):
    """
    Рассчитывает простые структурные признаки молекулы.
    """
    if mol is None:
        return {
            "valid_mol": False,
            "formula_rdkit": "",
            "carbon_count_rdkit": 0,
            "heteroatom_count_rdkit": 0,
            "heavy_atom_count_rdkit": 0,
            "ring_count_rdkit": 0,
            "aromatic_atom_count_rdkit": 0,
            "aromatic_ring_count_rdkit": 0,
            "aliphatic_ring_count_rdkit": 0,
            "heteroaromatic_ring_count_rdkit": 0,
            "fused_ring_system_count_rdkit": 0,
            "formal_charge_rdkit": 0,
            "positive_atom_count_rdkit": 0,
            "negative_atom_count_rdkit": 0,
            "charge_class_rdkit": "invalid",
            "mol_weight_rdkit": 0.0,
            "rotatable_bond_count_rdkit": 0,
            "tpsa_rdkit": 0.0,
            "hbd_rdkit": 0,
            "hba_rdkit": 0,
            "fraction_csp3_rdkit": 0.0,
            "only_CH_rdkit": False,
        }

    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    atom_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

    carbon_count = atom_symbols.count("C")

    heteroatom_count = sum(
        1 for symbol in atom_symbols
        if symbol not in ["C", "H"]
    )

    heavy_atom_count = mol.GetNumHeavyAtoms()
    ring_count = mol.GetRingInfo().NumRings()
    aromatic_ring_count = rdMolDescriptors.CalcNumAromaticRings(mol)
    aliphatic_ring_count = int(rdMolDescriptors.CalcNumAliphaticRings(mol))
    ring_atom_sets = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    heteroaromatic_ring_count = int(sum(
        all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)
        and any(mol.GetAtomWithIdx(idx).GetAtomicNum() not in [6, 1] for idx in ring)
        for ring in ring_atom_sets
    ))
    fused_ring_system_count = 0
    remaining_rings = list(ring_atom_sets)
    while remaining_rings:
        seed = remaining_rings.pop()
        stack = [seed]
        fused_system = set(seed)
        while stack:
            current = stack.pop()
            touching = [
                ring for ring in remaining_rings
                if len(current.intersection(ring)) >= 2
            ]
            for ring in touching:
                remaining_rings.remove(ring)
                fused_system.update(ring)
                stack.append(ring)
        fused_ring_system_count += 1

    aromatic_atom_count = sum(
        1 for atom in mol.GetAtoms()
        if atom.GetIsAromatic()
    )
    formal_charge = int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
    positive_atoms = int(sum(atom.GetFormalCharge() > 0 for atom in mol.GetAtoms()))
    negative_atoms = int(sum(atom.GetFormalCharge() < 0 for atom in mol.GetAtoms()))
    if positive_atoms > 0 and negative_atoms > 0:
        charge_class = "zwitterion"
    elif formal_charge > 0:
        charge_class = "cation"
    elif formal_charge < 0:
        charge_class = "anion"
    else:
        charge_class = "neutral"

    only_ch = all(number in [1, 6] for number in atom_numbers)

    return {
        "valid_mol": True,
        "formula_rdkit": structural_filter_formula_from_mol(mol),
        "carbon_count_rdkit": carbon_count,
        "heteroatom_count_rdkit": heteroatom_count,
        "heavy_atom_count_rdkit": heavy_atom_count,
        "ring_count_rdkit": ring_count,
        "aromatic_atom_count_rdkit": aromatic_atom_count,
        "aromatic_ring_count_rdkit": aromatic_ring_count,
        "aliphatic_ring_count_rdkit": aliphatic_ring_count,
        "heteroaromatic_ring_count_rdkit": heteroaromatic_ring_count,
        "fused_ring_system_count_rdkit": fused_ring_system_count,
        "formal_charge_rdkit": formal_charge,
        "positive_atom_count_rdkit": positive_atoms,
        "negative_atom_count_rdkit": negative_atoms,
        "charge_class_rdkit": charge_class,
        "mol_weight_rdkit": float(Descriptors.MolWt(mol)),
        "rotatable_bond_count_rdkit": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "tpsa_rdkit": float(rdMolDescriptors.CalcTPSA(mol)),
        "hbd_rdkit": int(rdMolDescriptors.CalcNumHBD(mol)),
        "hba_rdkit": int(rdMolDescriptors.CalcNumHBA(mol)),
        "fraction_csp3_rdkit": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "only_CH_rdkit": only_ch,
    }


def structural_filter_apply(
    data,
    smiles_col,
    selected_elements=None,
    element_mode="Любой выбранный элемент",
    selected_groups=None,
    group_mode="Любая выбранная группа",
    custom_smarts_text="",
    custom_smarts_mode="Любой SMARTS",
    require_aromatic="Не важно",
    require_only_ch=False,
    carbon_min=None,
    carbon_max=None,
    hetero_min=None,
    hetero_max=None,
    group_count_name=None,
    group_count_min=None,
    group_count_max=None,
    group_count_exact=None,
    element_count_symbol=None,
    element_count_min=None,
    element_count_max=None,
    element_count_exact=None,
    charge_mode="any",
    formal_charge_min=None,
    formal_charge_max=None,
    mol_weight_min=None,
    mol_weight_max=None,
    rotatable_min=None,
    rotatable_max=None,
    tpsa_min=None,
    tpsa_max=None,
    hbd_min=None,
    hbd_max=None,
    hba_min=None,
    hba_max=None,
    ring_min=None,
    ring_max=None,
    aromatic_ring_min=None,
    aromatic_ring_max=None,
    fraction_csp3_min=None,
    fraction_csp3_max=None,
    text_col=None,
    text_query="",
    combine_mode="Все условия одновременно"
):
    """
    Фильтрация датасета по структурным условиям.

    smiles_col — имя колонки со структурами, например 'canonical_smiles'.

    combine_mode:
    - 'Все условия одновременно' = AND
    - 'Хотя бы одно условие' = OR
    """
    if data is None or data.empty:
        raise ValueError("Нет данных для фильтрации.")

    if smiles_col not in data.columns:
        raise ValueError(f"Колонка со структурами не найдена: {smiles_col}")

    selected_elements = selected_elements or []
    selected_groups = selected_groups or []

    work = data.copy()

    mols = []
    analysis_rows = []

    for _, row in work.iterrows():
        smiles = str(row.get(smiles_col, "")).strip()
        mol = structural_filter_mol_from_smiles(smiles)
        mols.append(mol)

        info = structural_filter_analyze_mol(mol)
        analysis_rows.append(info)

    analysis_df = pd.DataFrame(analysis_rows, index=work.index)

    for col in analysis_df.columns:
        work[col] = analysis_df[col]

    conditions = []
    condition_names = []

    valid_condition = work["valid_mol"] == True
    conditions.append(valid_condition)

    if selected_elements:
        element_hits = []

        for mol in mols:
            if mol is None:
                element_hits.append(False)
                continue

            atom_symbols = set(atom.GetSymbol() for atom in mol.GetAtoms())

            if structural_filter_normalize_match_mode(element_mode) == "all":
                element_hits.append(
                    all(el in atom_symbols for el in selected_elements)
                )
            else:
                element_hits.append(
                    any(el in atom_symbols for el in selected_elements)
                )

        conditions.append(pd.Series(element_hits, index=work.index))
        condition_names.append("элементный состав")

    matched_group_labels = [""] * len(work)

    if selected_groups:
        group_patterns = {}

        for group_name in selected_groups:
            smarts = FUNCTIONAL_GROUP_SMARTS.get(group_name)

            if smarts:
                patt = Chem.MolFromSmarts(smarts)

                if patt is not None:
                    group_patterns[group_name] = patt

        if selected_groups and not group_patterns:
            raise ValueError("No selected functional-group SMARTS could be parsed.")

        group_hits = []
        matched_group_labels = []

        for mol in mols:
            if mol is None:
                group_hits.append(False)
                matched_group_labels.append("")
                continue

            current_matches = []

            for group_name, patt in group_patterns.items():
                if mol.HasSubstructMatch(patt):
                    current_matches.append(group_name)

            matched_group_labels.append("; ".join(current_matches))

            if structural_filter_normalize_match_mode(group_mode) == "all":
                group_hits.append(len(current_matches) == len(group_patterns))
            else:
                group_hits.append(len(current_matches) > 0)

        conditions.append(pd.Series(group_hits, index=work.index))
        condition_names.append("функциональные группы")

    work["Найденные группы"] = matched_group_labels

    if group_count_name and group_count_name in FUNCTIONAL_GROUP_SMARTS:
        patt = Chem.MolFromSmarts(FUNCTIONAL_GROUP_SMARTS[group_count_name])
        counts = []
        for mol in mols:
            if mol is None or patt is None:
                counts.append(0)
            else:
                counts.append(len(mol.GetSubstructMatches(patt, uniquify=True)))
        count_col = "functional_group_count_filter"
        work[count_col] = counts
        group_count_condition = pd.Series(True, index=work.index)
        if group_count_exact is not None:
            group_count_condition = group_count_condition & (
                work[count_col] == int(group_count_exact)
            )
        if group_count_min is not None:
            group_count_condition = group_count_condition & (
                work[count_col] >= int(group_count_min)
            )
        if group_count_max is not None:
            group_count_condition = group_count_condition & (
                work[count_col] <= int(group_count_max)
            )
        conditions.append(group_count_condition)
        condition_names.append("functional group count")

    if element_count_symbol:
        element_count_symbol = str(element_count_symbol).strip()
        counts = []
        for mol in mols:
            if mol is None:
                counts.append(0)
            else:
                counts.append(sum(atom.GetSymbol() == element_count_symbol for atom in mol.GetAtoms()))
        count_col = f"element_count_{element_count_symbol}"
        work[count_col] = counts
        element_count_condition = pd.Series(True, index=work.index)
        if element_count_exact is not None:
            element_count_condition = element_count_condition & (
                work[count_col] == int(element_count_exact)
            )
        if element_count_min is not None:
            element_count_condition = element_count_condition & (
                work[count_col] >= int(element_count_min)
            )
        if element_count_max is not None:
            element_count_condition = element_count_condition & (
                work[count_col] <= int(element_count_max)
            )
        conditions.append(element_count_condition)
        condition_names.append(f"element count {element_count_symbol}")

    custom_smarts_list = [
        line.strip()
        for line in str(custom_smarts_text).splitlines()
        if line.strip()
    ]

    if custom_smarts_list:
        smarts_status = structural_filter_validate_custom_smarts(custom_smarts_text)
        if (
            isinstance(smarts_status, pd.DataFrame)
            and not smarts_status.empty
            and smarts_status["status"].astype(str).ne("ok").any()
        ):
            bad = smarts_status[smarts_status["status"].astype(str) != "ok"]
            raise ValueError(
                "Invalid custom SMARTS: "
                + "; ".join(
                    f"line {row['line']}: {row['SMARTS']}"
                    for _, row in bad.iterrows()
                )
            )
        custom_patterns = []
        invalid_smarts = []

        for smarts in custom_smarts_list:
            patt = Chem.MolFromSmarts(smarts)

            if patt is None:
                invalid_smarts.append(smarts)
            else:
                custom_patterns.append((smarts, patt))

        if invalid_smarts:
            raise ValueError(
                "Некорректные SMARTS: " + ", ".join(invalid_smarts)
            )

        custom_hits = []
        custom_hit_labels = []

        for mol in mols:
            if mol is None:
                custom_hits.append(False)
                custom_hit_labels.append("")
                continue

            matched_smarts = []

            for smarts, patt in custom_patterns:
                if mol.HasSubstructMatch(patt):
                    matched_smarts.append(smarts)

            custom_hit_labels.append("; ".join(matched_smarts))

            if structural_filter_normalize_match_mode(custom_smarts_mode) == "all":
                custom_hits.append(len(matched_smarts) == len(custom_patterns))
            else:
                custom_hits.append(len(matched_smarts) > 0)

        conditions.append(pd.Series(custom_hits, index=work.index))
        condition_names.append("пользовательские SMARTS")
        work["Найденные SMARTS"] = custom_hit_labels
    else:
        work["Найденные SMARTS"] = ""

    aromatic_mode = structural_filter_normalize_aromatic_mode(require_aromatic)
    if aromatic_mode == "only_aromatic":
        conditions.append(work["aromatic_atom_count_rdkit"] > 0)
        condition_names.append("ароматичность")

    elif aromatic_mode == "only_non_aromatic":
        conditions.append(work["aromatic_atom_count_rdkit"] == 0)
        condition_names.append("неароматичность")

    if require_only_ch:
        conditions.append(work["only_CH_rdkit"] == True)
        condition_names.append("только C и H")

    if carbon_min is not None:
        conditions.append(work["carbon_count_rdkit"] >= int(carbon_min))
        condition_names.append("минимум C")

    if carbon_max is not None:
        conditions.append(work["carbon_count_rdkit"] <= int(carbon_max))
        condition_names.append("максимум C")

    if hetero_min is not None:
        conditions.append(work["heteroatom_count_rdkit"] >= int(hetero_min))
        condition_names.append("минимум гетероатомов")

    if hetero_max is not None:
        conditions.append(work["heteroatom_count_rdkit"] <= int(hetero_max))
        condition_names.append("максимум гетероатомов")

    charge_mode = str(charge_mode or "any").strip().lower()
    if charge_mode in {"neutral", "cation", "anion", "zwitterion"}:
        conditions.append(work["charge_class_rdkit"] == charge_mode)
        condition_names.append("charge class")

    if formal_charge_min is not None:
        conditions.append(work["formal_charge_rdkit"] >= int(formal_charge_min))
        condition_names.append("formal charge min")

    if formal_charge_max is not None:
        conditions.append(work["formal_charge_rdkit"] <= int(formal_charge_max))
        condition_names.append("formal charge max")

    range_specs = [
        ("mol_weight_rdkit", mol_weight_min, mol_weight_max, float, "molecular weight"),
        ("rotatable_bond_count_rdkit", rotatable_min, rotatable_max, int, "rotatable bonds"),
        ("tpsa_rdkit", tpsa_min, tpsa_max, float, "TPSA"),
        ("hbd_rdkit", hbd_min, hbd_max, int, "HBD"),
        ("hba_rdkit", hba_min, hba_max, int, "HBA"),
        ("ring_count_rdkit", ring_min, ring_max, int, "rings"),
        ("aromatic_ring_count_rdkit", aromatic_ring_min, aromatic_ring_max, int, "aromatic rings"),
        ("fraction_csp3_rdkit", fraction_csp3_min, fraction_csp3_max, float, "fraction Csp3"),
    ]
    for column, min_value, max_value, caster, label in range_specs:
        if min_value is not None:
            conditions.append(work[column] >= caster(min_value))
            condition_names.append(f"{label} min")
        if max_value is not None:
            conditions.append(work[column] <= caster(max_value))
            condition_names.append(f"{label} max")

    if text_col is not None and text_col in work.columns and str(text_query).strip():
        q = str(text_query).strip().lower()

        text_condition = work[text_col].astype(str).str.lower().str.contains(
            q,
            regex=False,
            na=False
        )

        conditions.append(text_condition)
        condition_names.append("текстовый поиск")

    user_conditions = conditions[1:]

    if user_conditions and structural_filter_normalize_combine_mode(combine_mode) == "any":
        final_mask = user_conditions[0].copy()

        for cond in user_conditions[1:]:
            final_mask = final_mask | cond

        final_mask = valid_condition & final_mask
        filtered = work.loc[final_mask].copy()
        report = structural_filter_report(work, filtered, condition_names, combine_mode)
        return filtered, report

    if not user_conditions:
        final_mask = pd.Series(True, index=work.index)
    else:
        if structural_filter_normalize_combine_mode(combine_mode) == "any":
            final_mask = user_conditions[0].copy()

            for cond in user_conditions[1:]:
                final_mask = final_mask | cond

        else:
            final_mask = user_conditions[0].copy()

            for cond in user_conditions[1:]:
                final_mask = final_mask & cond

    final_mask = valid_condition & final_mask

    filtered = work.loc[final_mask].copy()

    report = structural_filter_report(work, filtered, condition_names, combine_mode)

    return filtered, report
