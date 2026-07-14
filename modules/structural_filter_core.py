# -*- coding: utf-8 -*-

"""
Structural filter core for QSPR Forge.

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
    """
    Простая формула по атомам RDKit.
    """
    if mol is None:
        return ""

    counts = {}

    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        counts[symbol] = counts.get(symbol, 0) + 1

    parts = []

    for symbol in ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"]:
        if symbol in counts:
            n = counts.pop(symbol)

            if n == 1:
                parts.append(symbol)
            else:
                parts.append(f"{symbol}{n}")

    for symbol in sorted(counts.keys()):
        n = counts[symbol]

        if n == 1:
            parts.append(symbol)
        else:
            parts.append(f"{symbol}{n}")

    return "".join(parts)


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
            "only_CH_rdkit": False,
        }

    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

    carbon_count = atom_symbols.count("C")

    heteroatom_count = sum(
        1 for symbol in atom_symbols
        if symbol not in ["C", "H"]
    )

    heavy_atom_count = mol.GetNumHeavyAtoms()
    ring_count = mol.GetRingInfo().NumRings()

    aromatic_atom_count = sum(
        1 for atom in mol.GetAtoms()
        if atom.GetIsAromatic()
    )

    only_ch = all(symbol in ["C", "H"] for symbol in atom_symbols)

    return {
        "valid_mol": True,
        "formula_rdkit": structural_filter_formula_from_mol(mol),
        "carbon_count_rdkit": carbon_count,
        "heteroatom_count_rdkit": heteroatom_count,
        "heavy_atom_count_rdkit": heavy_atom_count,
        "ring_count_rdkit": ring_count,
        "aromatic_atom_count_rdkit": aromatic_atom_count,
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
    condition_names.append("корректная структура")

    if selected_elements:
        element_hits = []

        for mol in mols:
            if mol is None:
                element_hits.append(False)
                continue

            atom_symbols = set(atom.GetSymbol() for atom in mol.GetAtoms())

            if element_mode == "Все выбранные элементы":
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

            if group_mode == "Все выбранные группы":
                group_hits.append(len(current_matches) == len(group_patterns))
            else:
                group_hits.append(len(current_matches) > 0)

        conditions.append(pd.Series(group_hits, index=work.index))
        condition_names.append("функциональные группы")

    work["Найденные группы"] = matched_group_labels

    custom_smarts_list = [
        line.strip()
        for line in str(custom_smarts_text).splitlines()
        if line.strip()
    ]

    if custom_smarts_list:
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

            if custom_smarts_mode == "Все SMARTS":
                custom_hits.append(len(matched_smarts) == len(custom_patterns))
            else:
                custom_hits.append(len(matched_smarts) > 0)

        conditions.append(pd.Series(custom_hits, index=work.index))
        condition_names.append("пользовательские SMARTS")
        work["Найденные SMARTS"] = custom_hit_labels
    else:
        work["Найденные SMARTS"] = ""

    if require_aromatic == "Только ароматические":
        conditions.append(work["aromatic_atom_count_rdkit"] > 0)
        condition_names.append("ароматичность")

    elif require_aromatic == "Только неароматические":
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

    if text_col is not None and text_col in work.columns and str(text_query).strip():
        q = str(text_query).strip().lower()

        text_condition = work[text_col].astype(str).str.lower().str.contains(
            q,
            regex=False,
            na=False
        )

        conditions.append(text_condition)
        condition_names.append("текстовый поиск")

    if not conditions:
        final_mask = pd.Series(True, index=work.index)
    else:
        if combine_mode == "Хотя бы одно условие":
            final_mask = conditions[0].copy()

            for cond in conditions[1:]:
                final_mask = final_mask | cond

        else:
            final_mask = conditions[0].copy()

            for cond in conditions[1:]:
                final_mask = final_mask & cond

    filtered = work.loc[final_mask].copy()

    report = {
        "Всего строк": len(work),
        "Корректных структур": int(work["valid_mol"].sum()),
        "После фильтра": len(filtered),
        "Условий применено": len(condition_names),
        "Список условий": ", ".join(condition_names),
    }

    return filtered, report