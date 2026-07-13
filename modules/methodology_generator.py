# -*- coding: utf-8 -*-
"""
methodology_generator.py

Генератор текста методики QSPR-моделирования.
Поддерживает:
- русский и английский языки;
- два стиля: полный (full) и краткий (short);
- интеллектуальное заключение на основе R², RMSE, MAPE, размера выборки, AD и переобучения;
- историю версий (управляется извне).

Шаблоны загружаются из help/methodology_templates.json,
при отсутствии используются встроенные.
"""

import json
import os
from typing import Dict, Any, Optional

TEMPLATES_FILE = os.path.join("help", "methodology_templates.json")

DEFAULT_TEMPLATES = {
    "ru": {
        "intro": "Для построения модели использовали выборку из {n_compounds} соединений с известными значениями свойства «{target_col}».",
        "data_description": "Распределение свойства: среднее = {mean:.2f}, стандартное отклонение = {std:.2f}, диапазон = {min:.2f} – {max:.2f}.",
        "descriptors": {
            "rdkit": "Молекулярные дескрипторы рассчитывали с использованием библиотеки RDKit ({n_rdkit} шт.).",
            "mordred": "Дополнительно рассчитаны дескрипторы Mordred ({n_mordred} шт.).",
            "padel": "Для расширения химического пространства использованы дескрипторы PaDEL ({n_padel} шт.).",
            "selection": "Перед моделированием удаляли константные ({n_const}) и сильно коррелирующие (|r|>{corr_threshold}) признаки. Финальный отбор выполнен методом «{method}» до {n_final} наиболее информативных дескрипторов.",
            "no_selection": "Отбор дескрипторов не применялся; использованы все {n_total} рассчитанных признаков."
        },
        "model": {
            "intro": "Модель регрессии построена методом {model_name}.",
            "params": "Параметры модели: {params}.",
            "optimized": "Гиперпараметры оптимизированы с помощью {search_method} по {cv}-кратной кросс-валидации. Лучшие параметры: {best_params}."
        },
        "validation": {
            "kfold": "Качество модели оценивали с помощью {k}-кратной кросс-валидации (R² = {r2:.3f}, RMSE = {rmse:.2f}, MAE = {mae:.2f}).",
            "holdout": "На тестовой выборке (доля {test_size:.0%}) получены следующие метрики: R² = {r2:.3f}, RMSE = {rmse:.2f}, MAE = {mae:.2f}.",
            "loo": "Методом Leave-One-Out получены: Q² = {q2:.3f}, RMSE = {rmse:.2f}.",
            "bootstrap": "Bootstrap-валидация ({n_iter} итераций) показала средний R² OOB = {r2_mean:.3f} ± {r2_std:.3f}.",
            "external": "Стресс-тест на структурно удалённых объектах (distance-based hold-out, доля {fraction:.0%}) дал R² = {r2_mean:.3f} ± {r2_std:.3f}. Это не строгая внешняя валидация на независимом наборе."
        },
        "ad": "Применимую область модели оценивали по рычаговому расстоянию (leverage). Порог h* = {threshold:.4f}. Выявлено {n_out} веществ ({pct:.1f}%) вне применимой области.",
        "conclusion": {
            "excellent": "Модель демонстрирует высокую предсказательную способность и может быть рекомендована для прогнозирования целевого свойства новых соединений в рамках применимой области.",
            "good": "Модель показывает хорошую предсказательную способность и может использоваться для ориентировочных оценок.",
            "moderate": "Модель обладает умеренной предсказательной способностью. Для повышения точности рекомендуется увеличить выборку или рассмотреть альтернативные методы.",
            "poor": "Модель показывает низкую предсказательную способность. Это может быть связано с недостаточным объёмом данных, нелинейностью зависимости или неинформативностью дескрипторов. Рекомендуется расширить выборку, использовать другие дескрипторы или более сложные алгоритмы.",
            "overfitted": "Модель показывает высокий R² на обучающей выборке, но значительно более низкий на кросс-валидации, что указывает на переобучение. Рекомендуется уменьшить сложность модели или увеличить выборку.",
            "small_sample": "Модель показывает хорошие метрики, однако выборка мала (менее 30 веществ), что снижает статистическую надёжность. Результаты следует интерпретировать с осторожностью.",
            "limited_ad": "Модель имеет хорошие метрики, но значительная часть веществ (более 20%) находится вне применимой области, что ограничивает её использование для новых соединений.",
            "default": "Качество модели следует оценивать по совокупности метрик и области применимости."
        }
    },
    "en": {
        "intro": "A dataset of {n_compounds} compounds with known values of {target_col} was used for modeling.",
        "data_description": "Property distribution: mean = {mean:.2f}, std = {std:.2f}, range = {min:.2f} – {max:.2f}.",
        "descriptors": {
            "rdkit": "Molecular descriptors were calculated using RDKit ({n_rdkit} features).",
            "mordred": "Mordred descriptors were additionally calculated ({n_mordred} features).",
            "padel": "PaDEL descriptors were used to expand chemical space ({n_padel} features).",
            "selection": "Constant ({n_const}) and highly correlated (|r|>{corr_threshold}) descriptors were removed. Final selection was performed by {method} to retain {n_final} most informative features.",
            "no_selection": "No descriptor selection was applied; all {n_total} calculated features were used."
        },
        "model": {
            "intro": "A {model_name} regression model was built.",
            "params": "Model parameters: {params}.",
            "optimized": "Hyperparameters were optimized using {search_method} with {cv}-fold cross-validation. Best parameters: {best_params}."
        },
        "validation": {
            "kfold": "Model performance was assessed using {k}-fold cross-validation (R² = {r2:.3f}, RMSE = {rmse:.2f}, MAE = {mae:.2f}).",
            "holdout": "On the test set ({test_size:.0%} fraction), the following metrics were obtained: R² = {r2:.3f}, RMSE = {rmse:.2f}, MAE = {mae:.2f}.",
            "loo": "Leave-One-Out cross-validation gave Q² = {q2:.3f}, RMSE = {rmse:.2f}.",
            "bootstrap": "Bootstrap validation ({n_iter} iterations) showed mean OOB R² = {r2_mean:.3f} ± {r2_std:.3f}.",
            "external": "Distance-based hold-out stress test (fraction {fraction:.0%}) gave R² = {r2_mean:.3f} ± {r2_std:.3f}. This is not strict external validation on an independent dataset."
        },
        "ad": "Applicability domain was estimated using leverage. The threshold h* = {threshold:.4f}. {n_out} compounds ({pct:.1f}%) were found outside the domain.",
        "conclusion": {
            "excellent": "The model shows high predictive power and can be recommended for predicting the target property of new compounds within the applicability domain.",
            "good": "The model shows good predictive performance and can be used for approximate estimations.",
            "moderate": "The model has moderate predictive ability. To improve accuracy, consider increasing the dataset or trying alternative methods.",
            "poor": "The model exhibits low predictive ability. This may be due to insufficient data, non-linearity, or uninformative descriptors. Consider expanding the dataset, using other descriptors, or more complex algorithms.",
            "overfitted": "The model shows high R² on training but much lower on cross-validation, indicating overfitting. Consider reducing model complexity or increasing the dataset.",
            "small_sample": "The model shows good metrics, but the sample size is small (<30 compounds), which reduces statistical reliability. Results should be interpreted with caution.",
            "limited_ad": "The model has good metrics, but a significant portion (>20%) of compounds are outside the applicability domain, limiting its use for new compounds.",
            "default": "The model quality should be assessed based on the overall metrics and applicability domain."
        }
    }
}


def load_templates(language: str = "ru") -> Dict[str, Any]:
    """Загружает шаблоны из JSON-файла или возвращает встроенные."""
    if os.path.exists(TEMPLATES_FILE):
        try:
            with open(TEMPLATES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if language in data:
                    return data[language]
        except Exception:
            pass
    return DEFAULT_TEMPLATES.get(language, DEFAULT_TEMPLATES["ru"])


def _format_params(params: Dict[str, Any]) -> str:
    """Форматирует словарь параметров в строку 'ключ=значение, ключ=значение'."""
    if not params:
        return ""
    items = [f"{k}={v}" for k, v in params.items()]
    return ", ".join(items)


def _evaluate_quality(
    r2: Optional[float],
    rmse: Optional[float] = None,
    mape: Optional[float] = None,
    n_compounds: Optional[int] = None,
    r2_cv: Optional[float] = None,
    ad_out_fraction: Optional[float] = None
) -> str:
    """
    Оценивает качество модели по нескольким критериям и возвращает ключ для шаблона.
    """
    # Если нет R², возвращаем default
    if r2 is None:
        return "default"

    # Базовый уровень
    if r2 >= 0.85 and (rmse is None or rmse < 5) and (mape is None or mape < 10):
        level = "excellent"
    elif r2 >= 0.70 and (rmse is None or rmse < 10) and (mape is None or mape < 20):
        level = "good"
    elif r2 >= 0.50 and (rmse is None or rmse < 20) and (mape is None or mape < 40):
        level = "moderate"
    else:
        level = "poor"

    # Проверка на переобучение (если есть R² CV)
    if r2_cv is not None and (r2 - r2_cv) > 0.2:
        return "overfitted"

    # Проверка на малый размер выборки
    if n_compounds is not None and n_compounds < 30 and level in ["excellent", "good"]:
        return "small_sample"

    # Проверка на долю вне AD
    if ad_out_fraction is not None and ad_out_fraction > 0.2:
        return "limited_ad"

    return level


def generate_methodology_text(data: Dict[str, Any], language: str = "ru", style: str = "full") -> str:
    """
    Генерирует текст методики на основе переданных данных.

    Аргументы:
        data (dict): словарь с данными, обычно извлекаемый из session_state.
            Ожидаемые ключи:
            - n_compounds: int
            - target_col: str
            - target_stats: dict {'mean', 'std', 'min', 'max'}
            - descriptors: dict {'rdkit': int, 'mordred': int, 'padel': int, 'total': int}
            - descriptor_selection: dict или None
                {'n_const': int, 'n_corr': int, 'corr_threshold': float, 'method': str, 'n_final': int}
            - model: dict {'name': str, 'params': dict, 'optimized': bool, 'search_method': str, 'cv': int, 'best_params': dict}
            - validation: dict
                {'kfold': {'k': int, 'r2': float, 'rmse': float, 'mae': float, 'q2': float} или None,
                 'holdout': {'test_size': float, 'r2': float, 'rmse': float, 'mae': float} или None,
                 'loo': {'q2': float, 'rmse': float} или None,
                 'bootstrap': {'n_iter': int, 'r2_mean': float, 'r2_std': float} или None,
                 'external': {'fraction': float, 'r2_mean': float, 'r2_std': float} или None}
            - ad: dict или None {'threshold': float, 'n_out': int, 'pct': float}
            - conclusion_r2: float (используется для выбора тона заключения)
            - conclusion_rmse: float (опционально)
            - conclusion_mape: float (опционально)
            - conclusion_r2_cv: float (опционально)
            - conclusion_ad_out_fraction: float (опционально)
    """
    templates = load_templates(language)
    sections = []

    # 1. Введение
    intro = templates.get("intro", "").format(
        n_compounds=data.get("n_compounds", "?"),
        target_col=data.get("target_col", "свойства")
    )
    sections.append(intro)

    # 2. Описание данных (статистика)
    stats = data.get("target_stats", {})
    if stats:
        desc = templates.get("data_description", "").format(
            mean=stats.get("mean", 0),
            std=stats.get("std", 0),
            min=stats.get("min", 0),
            max=stats.get("max", 0)
        )
        sections.append(desc)

    # 3. Дескрипторы
    desc_dict = templates.get("descriptors", {})
    desc_parts = []
    d = data.get("descriptors", {})
    if d.get("rdkit", 0) > 0:
        desc_parts.append(desc_dict.get("rdkit", "").format(n_rdkit=d["rdkit"]))
    if d.get("mordred", 0) > 0:
        desc_parts.append(desc_dict.get("mordred", "").format(n_mordred=d["mordred"]))
    if d.get("padel", 0) > 0:
        desc_parts.append(desc_dict.get("padel", "").format(n_padel=d["padel"]))

    sel = data.get("descriptor_selection")
    if sel:
        sel_text = desc_dict.get("selection", "").format(
            n_const=sel.get("n_const", 0),
            n_corr=sel.get("n_corr", 0),
            corr_threshold=sel.get("corr_threshold", 0.95),
            method=sel.get("method", "быстрый"),
            n_final=sel.get("n_final", 0)
        )
        desc_parts.append(sel_text)
    else:
        total = d.get("total", 0)
        if total > 0:
            desc_parts.append(desc_dict.get("no_selection", "").format(n_total=total))

    if desc_parts:
        sections.append(" ".join(desc_parts))

    # 4. Модель
    model_data = data.get("model", {})
    model_parts = []
    model_intro = templates.get("model", {}).get("intro", "").format(
        model_name=model_data.get("name", "модель")
    )
    model_parts.append(model_intro)

    # Параметры модели (только для полного стиля)
    params = model_data.get("params", {})
    if params and style != "short":
        params_str = _format_params(params)
        if params_str:
            model_parts.append(templates.get("model", {}).get("params", "").format(params=params_str))

    # Оптимизация (только для полного стиля)
    if model_data.get("optimized", False) and style != "short":
        opt_text = templates.get("model", {}).get("optimized", "").format(
            search_method=model_data.get("search_method", "GridSearch"),
            cv=model_data.get("cv", 5),
            best_params=_format_params(model_data.get("best_params", {}))
        )
        model_parts.append(opt_text)

    if model_parts:
        sections.append(" ".join(model_parts))

    # 5. Валидация
    val = data.get("validation", {})
    val_parts = []
    val_templates = templates.get("validation", {})

    if val.get("kfold"):
        kf = val["kfold"]
        val_parts.append(val_templates.get("kfold", "").format(
            k=kf.get("k", 5),
            r2=kf.get("r2", 0),
            rmse=kf.get("rmse", 0),
            mae=kf.get("mae", 0)
        ))

    if val.get("holdout"):
        ho = val["holdout"]
        val_parts.append(val_templates.get("holdout", "").format(
            test_size=ho.get("test_size", 0.2),
            r2=ho.get("r2", 0),
            rmse=ho.get("rmse", 0),
            mae=ho.get("mae", 0)
        ))

    if val.get("loo"):
        lo = val["loo"]
        val_parts.append(val_templates.get("loo", "").format(
            q2=lo.get("q2", 0),
            rmse=lo.get("rmse", 0)
        ))

    if val.get("bootstrap"):
        bs = val["bootstrap"]
        val_parts.append(val_templates.get("bootstrap", "").format(
            n_iter=bs.get("n_iter", 0),
            r2_mean=bs.get("r2_mean", 0),
            r2_std=bs.get("r2_std", 0)
        ))

    if val.get("external"):
        ex = val["external"]
        val_parts.append(val_templates.get("external", "").format(
            fraction=ex.get("fraction", 0.2),
            r2_mean=ex.get("r2_mean", 0),
            r2_std=ex.get("r2_std", 0)
        ))

    if val_parts:
        sections.append(" ".join(val_parts))

    # 6. Applicability Domain
    ad = data.get("ad")
    if ad:
        ad_text = templates.get("ad", "").format(
            threshold=ad.get("threshold", 0),
            n_out=ad.get("n_out", 0),
            pct=ad.get("pct", 0)
        )
        sections.append(ad_text)

    # 7. Заключение (с расширенной логикой)
    r2_for_conclusion = data.get("conclusion_r2")
    rmse_for_conclusion = data.get("conclusion_rmse")
    mape_for_conclusion = data.get("conclusion_mape")
    r2_cv_for_conclusion = data.get("conclusion_r2_cv")
    ad_out_fraction = data.get("conclusion_ad_out_fraction")
    n_compounds = data.get("n_compounds")

    level = _evaluate_quality(
        r2=r2_for_conclusion,
        rmse=rmse_for_conclusion,
        mape=mape_for_conclusion,
        n_compounds=n_compounds,
        r2_cv=r2_cv_for_conclusion,
        ad_out_fraction=ad_out_fraction
    )

    conclusion_block = templates.get("conclusion", {}).get(level, templates.get("conclusion", {}).get("default", ""))
    if conclusion_block:
        sections.append(conclusion_block)

    # Объединение с абзацами
    return "\n\n".join(sections)
