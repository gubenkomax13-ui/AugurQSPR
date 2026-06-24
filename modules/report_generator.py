    # -*- coding: utf-8 -*-
"""
report_generator.py

Генератор полного научного отчёта по QSPR-моделированию,
соответствующего требованиям OECD и стандартам научных публикаций.

Включает все обязательные разделы:
1. Цель и конечная точка
2. Данные (химическое пространство, предобработка, статистика, визуализация)
3. Дескрипторы (типы, фильтрация, итоговое число)
4. Метод моделирования (алгоритм, гиперпараметры, валидация)
5. Результаты и метрики (таблицы, графики, Y-рандомизация)
6. Применимая область (AD) с графиком Вильямса
7. Интерпретация модели (коэффициенты, важность)
8. Выводы и рекомендации

Поддерживает русский и английский языки, вставку графиков в base64.
"""

import os
import json
import base64
import io
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Шаблоны текстов (можно вынести в JSON, но пока встроим)
TEMPLATES = {
    "ru": {
        "title": "Отчёт по QSPR-моделированию свойства «{target}»",
        "purpose": {
            "intro": "Целью работы являлось построение прогностической модели для свойства «{target}».",
            "source": "Экспериментальные данные взяты из источника: {source}.",
            "units": "Единицы измерения: {units}.",
            "accuracy": "Точность экспериментального метода: {accuracy}."
        },
        "data": {
            "intro": "Исходная выборка содержала {n_initial} соединений.",
            "classes": "Химическое пространство представлено следующими классами: {classes}.",
            "preprocessing": "Предобработка включала: {steps}.",
            "final_size": "Итоговый размер набора после очистки: {n_final} веществ.",
            "stats": "Описательная статистика свойства: среднее = {mean:.2f}, медиана = {median:.2f}, стандартное отклонение = {std:.2f}, минимум = {min:.2f}, максимум = {max:.2f}, асимметрия = {skew:.2f}.",
            "split": "Разделение на обучающую и тестовую выборки проводилось методом {split_method} с долей теста {test_size:.0%}."
        },
        "descriptors": {
            "intro": "Молекулярные дескрипторы рассчитывали с использованием {software}.",
            "types": "Рассчитаны следующие типы дескрипторов: {types}.",
            "initial_count": "Исходное количество дескрипторов: {n_initial}.",
            "filtering": "Предварительная фильтрация: удалены дескрипторы с постоянным значением ({n_const}), удалены сильно коррелированные пары (|r|>{corr_threshold}) ({n_corr}).",
            "selection": "Дополнительный отбор выполнен методом {method} до {n_final} наиболее информативных признаков.",
            "final_count": "Итоговое число дескрипторов, подаваемых в модель: {n_final}."
        },
        "model": {
            "intro": "Модель построена методом {algorithm}.",
            "params": "Гиперпараметры: {params}.",
            "tuning": "Подбор гиперпараметров выполнен с помощью {tuning_method} по {cv}-кратной кросс-валидации.",
            "validation": "Внутренняя валидация проводилась с использованием {validation_method} ({k}-кратной кросс-валидации). Метрики: Q²_CV = {q2_cv:.3f}, RMSE_CV = {rmse_cv:.2f}, MAE_CV = {mae_cv:.2f}."
        },
        "results": {
            "intro": "В таблице приведены метрики для обучающей, кросс-валидационной и тестовой выборок.",
            "metrics_table": "Таблица метрик",
            "r2": "R²",
            "rmse": "RMSE",
            "mae": "MAE",
            "r0": "R₀² (Голбрайх–Тропше)",
            "k": "k (наклон через начало координат)",
            "y_rand": "Y-рандомизация: среднее R²_rand = {r2_rand_mean:.3f}, среднее Q²_rand = {q2_rand_mean:.3f} (p-value = {p_value:.4f}), что подтверждает неслучайность модели."
        },
        "ad": {
            "intro": "Применимая область (AD) оценена методом {method}.",
            "threshold": "Порог рычагового расстояния h* = {threshold:.4f} (рассчитан как 3×(p+1)/n).",
            "counts": "Вне AD находятся {n_out} веществ ({pct:.1f}%) в тестовой выборке и {n_out_train} ({pct_train:.1f}%) в обучающей.",
            "williams": "На рисунке представлен график Вильямса (стандартизованные остатки vs рычаговое расстояние)."
        },
        "interpretation": {
            "intro": "Интерпретация модели выполнена на основе анализа {method}.",
            "top_features": "Наиболее важные дескрипторы: {top_list}.",
            "meaning": "Физико-химический смысл: {meaning}."
        },
        "conclusion": {
            "summary": "Построенная модель демонстрирует {quality} предсказательную способность.",
            "limitations": "Ограничения: {limitations}.",
            "recommendations": "Рекомендации по улучшению: {recommendations}."
        }
    },
    "en": {
        "title": "QSPR Modeling Report for {target}",
        "purpose": {
            "intro": "The goal was to build a predictive model for the property '{target}'.",
            "source": "Experimental data were taken from: {source}.",
            "units": "Units: {units}.",
            "accuracy": "Experimental method accuracy: {accuracy}."
        },
        "data": {
            "intro": "The initial dataset contained {n_initial} compounds.",
            "classes": "Chemical space includes the following classes: {classes}.",
            "preprocessing": "Preprocessing included: {steps}.",
            "final_size": "Final dataset size after cleaning: {n_final} compounds.",
            "stats": "Descriptive statistics: mean = {mean:.2f}, median = {median:.2f}, std = {std:.2f}, min = {min:.2f}, max = {max:.2f}, skewness = {skew:.2f}.",
            "split": "The split into training and test sets was performed using {split_method} with test fraction {test_size:.0%}."
        },
        "descriptors": {
            "intro": "Molecular descriptors were calculated using {software}.",
            "types": "Descriptor types: {types}.",
            "initial_count": "Initial number of descriptors: {n_initial}.",
            "filtering": "Pre-filtering: constant descriptors removed ({n_const}), highly correlated pairs removed (|r|>{corr_threshold}) ({n_corr}).",
            "selection": "Additional selection by {method} retained {n_final} most informative features.",
            "final_count": "Final number of descriptors used in the model: {n_final}."
        },
        "model": {
            "intro": "The {algorithm} model was built.",
            "params": "Hyperparameters: {params}.",
            "tuning": "Hyperparameters were tuned using {tuning_method} with {cv}-fold cross-validation.",
            "validation": "Internal validation used {validation_method} ({k}-fold cross-validation). Metrics: Q²_CV = {q2_cv:.3f}, RMSE_CV = {rmse_cv:.2f}, MAE_CV = {mae_cv:.2f}."
        },
        "results": {
            "intro": "The table below shows metrics for training, cross-validation, and test sets.",
            "metrics_table": "Metrics table",
            "r2": "R²",
            "rmse": "RMSE",
            "mae": "MAE",
            "r0": "R₀² (Golbraikh–Tropsha)",
            "k": "k (slope through origin)",
            "y_rand": "Y-randomization: mean R²_rand = {r2_rand_mean:.3f}, mean Q²_rand = {q2_rand_mean:.3f} (p-value = {p_value:.4f}), confirming model non-randomness."
        },
        "ad": {
            "intro": "Applicability domain (AD) was assessed using {method}.",
            "threshold": "Leverage threshold h* = {threshold:.4f} (calculated as 3×(p+1)/n).",
            "counts": "Outside AD: {n_out} compounds ({pct:.1f}%) in test set, {n_out_train} ({pct_train:.1f}%) in training set.",
            "williams": "The Williams plot (standardized residuals vs leverage) is shown below."
        },
        "interpretation": {
            "intro": "Model interpretation was based on {method}.",
            "top_features": "The most important descriptors: {top_list}.",
            "meaning": "Physicochemical meaning: {meaning}."
        },
        "conclusion": {
            "summary": "The model shows {quality} predictive ability.",
            "limitations": "Limitations: {limitations}.",
            "recommendations": "Recommendations for improvement: {recommendations}."
        }
    }
}


def _fig_to_base64(fig):
    """Преобразует matplotlib figure в base64 строку."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _format_list(items, max_items=5):
    """Форматирует список для вставки в текст."""
    if not items:
        return "—"
    if len(items) <= max_items:
        return ", ".join(str(x) for x in items)
    return ", ".join(str(x) for x in items[:max_items]) + f" и ещё {len(items)-max_items}"


def _get_quality_label(r2):
    """Возвращает текстовую оценку качества по R²."""
    if r2 is None:
        return "неизвестной"
    if r2 >= 0.85:
        return "высокой"
    elif r2 >= 0.70:
        return "хорошей"
    elif r2 >= 0.50:
        return "умеренной"
    else:
        return "низкой"


def generate_full_report(data: Dict[str, Any], language: str = "ru") -> Dict[str, Any]:
    """
    Генерирует полный научный отчёт по QSPR.

    Аргументы:
        data (dict): словарь с данными (из session_state).
            Ожидаемые ключи:
            - target_col: str
            - source: str (источник данных)
            - units: str (единицы измерения)
            - accuracy: str (точность метода)
            - n_initial: int (исходное число веществ)
            - n_final: int (после очистки)
            - chemical_classes: list (структурные классы)
            - preprocessing_steps: list (шаги предобработки)
            - target_stats: dict {'mean', 'median', 'std', 'min', 'max', 'skew'}
            - split_method: str (метод разделения)
            - test_size: float (доля теста)
            - descriptors: dict
                {'software': str, 'types': list, 'n_initial': int,
                 'n_const': int, 'n_corr': int, 'corr_threshold': float,
                 'selection_method': str, 'n_final': int}
            - model: dict
                {'algorithm': str, 'params': dict, 'tuning_method': str,
                 'cv': int, 'validation_method': str, 'k': int,
                 'q2_cv': float, 'rmse_cv': float, 'mae_cv': float}
            - metrics: dict
                {'train': {'r2': float, 'rmse': float, 'mae': float},
                 'cv': {'r2': float, 'rmse': float, 'mae': float},
                 'test': {'r2': float, 'rmse': float, 'mae': float}}
            - y_randomization: dict or None
                {'r2_rand_mean': float, 'q2_rand_mean': float, 'p_value': float}
            - ad: dict or None
                {'method': str, 'threshold': float,
                 'n_out_test': int, 'pct_test': float,
                 'n_out_train': int, 'pct_train': float}
            - interpretation: dict or None
                {'method': str, 'top_features': list, 'meaning': str}
            - plots: dict (опционально) с matplotlib figures:
                {'histogram': fig, 'scatter': fig, 'residuals': fig,
                 'williams': fig, 'y_rand_hist': fig}
    """
    tpl = TEMPLATES.get(language, TEMPLATES["ru"])
    sections = {}

    # ---- 1. Цель ----
    sections['title'] = tpl['title'].format(target=data.get('target_col', 'свойства'))
    purpose_parts = [
        tpl['purpose']['intro'].format(target=data.get('target_col', 'свойства'))
    ]
    if data.get('source'):
        purpose_parts.append(tpl['purpose']['source'].format(source=data['source']))
    if data.get('units'):
        purpose_parts.append(tpl['purpose']['units'].format(units=data['units']))
    if data.get('accuracy'):
        purpose_parts.append(tpl['purpose']['accuracy'].format(accuracy=data['accuracy']))
    sections['purpose'] = " ".join(purpose_parts)

    # ---- 2. Данные ----
    data_parts = [
        tpl['data']['intro'].format(n_initial=data.get('n_initial', '?'))
    ]
    if data.get('chemical_classes'):
        data_parts.append(tpl['data']['classes'].format(classes=_format_list(data['chemical_classes'])))
    if data.get('preprocessing_steps'):
        data_parts.append(tpl['data']['preprocessing'].format(steps="; ".join(data['preprocessing_steps'])))
    data_parts.append(tpl['data']['final_size'].format(n_final=data.get('n_final', '?')))
    stats = data.get('target_stats', {})
    if stats:
        data_parts.append(tpl['data']['stats'].format(
            mean=stats.get('mean', 0),
            median=stats.get('median', 0),
            std=stats.get('std', 0),
            min=stats.get('min', 0),
            max=stats.get('max', 0),
            skew=stats.get('skew', 0)
        ))
    if data.get('split_method') and data.get('test_size') is not None:
        data_parts.append(tpl['data']['split'].format(
            split_method=data['split_method'],
            test_size=data['test_size']
        ))
    sections['data'] = " ".join(data_parts)

    # ---- 3. Дескрипторы ----
    desc = data.get('descriptors', {})
    desc_parts = [
        tpl['descriptors']['intro'].format(software=desc.get('software', 'RDKit'))
    ]
    if desc.get('types'):
        desc_parts.append(tpl['descriptors']['types'].format(types=_format_list(desc['types'])))
    desc_parts.append(tpl['descriptors']['initial_count'].format(n_initial=desc.get('n_initial', 0)))
    if desc.get('n_const', 0) > 0 or desc.get('n_corr', 0) > 0:
        desc_parts.append(tpl['descriptors']['filtering'].format(
            n_const=desc.get('n_const', 0),
            corr_threshold=desc.get('corr_threshold', 0.95),
            n_corr=desc.get('n_corr', 0)
        ))
    if desc.get('selection_method'):
        desc_parts.append(tpl['descriptors']['selection'].format(
            method=desc['selection_method'],
            n_final=desc.get('n_final', 0)
        ))
    desc_parts.append(tpl['descriptors']['final_count'].format(n_final=desc.get('n_final', 0)))
    sections['descriptors'] = " ".join(desc_parts)

    # ---- 4. Метод моделирования ----
    model = data.get('model', {})
    model_parts = [
        tpl['model']['intro'].format(algorithm=model.get('algorithm', 'модель'))
    ]
    if model.get('params'):
        params_str = ", ".join(f"{k}={v}" for k, v in model['params'].items())
        model_parts.append(tpl['model']['params'].format(params=params_str))
    if model.get('tuning_method'):
        model_parts.append(tpl['model']['tuning'].format(
            tuning_method=model['tuning_method'],
            cv=model.get('cv', 5)
        ))
    if model.get('validation_method') and model.get('k'):
        model_parts.append(tpl['model']['validation'].format(
            validation_method=model['validation_method'],
            k=model['k'],
            q2_cv=model.get('q2_cv', 0),
            rmse_cv=model.get('rmse_cv', 0),
            mae_cv=model.get('mae_cv', 0)
        ))
    sections['model'] = " ".join(model_parts)

    # ---- 5. Результаты и метрики ----
    metrics = data.get('metrics', {})
    results_parts = [tpl['results']['intro']]
    # Таблица метрик (в HTML)
    table_html = "<table><tr><th>Выборка</th><th>R²</th><th>RMSE</th><th>MAE</th></tr>"
    for name, m in [('Обучающая', metrics.get('train', {})),
                    ('Кросс-валидация', metrics.get('cv', {})),
                    ('Тестовая', metrics.get('test', {}))]:
        if m:
            table_html += f"<tr><td>{name}</td><td>{m.get('r2', '—'):.3f}</td><td>{m.get('rmse', '—'):.2f}</td><td>{m.get('mae', '—'):.2f}</td></tr>"
    table_html += "</table>"
    results_parts.append(table_html)

    # Y-рандомизация
    yrand = data.get('y_randomization')
    if yrand:
        results_parts.append(tpl['results']['y_rand'].format(
            r2_rand_mean=yrand.get('r2_rand_mean', 0),
            q2_rand_mean=yrand.get('q2_rand_mean', 0),
            p_value=yrand.get('p_value', 1)
        ))
    sections['results'] = " ".join(results_parts)

    # ---- 6. Применимая область ----
    ad = data.get('ad')
    if ad:
        ad_parts = [
            tpl['ad']['intro'].format(method=ad.get('method', 'leverage')),
            tpl['ad']['threshold'].format(threshold=ad.get('threshold', 0))
        ]
        if ad.get('n_out_test') is not None:
            ad_parts.append(tpl['ad']['counts'].format(
                n_out=ad.get('n_out_test', 0),
                pct=ad.get('pct_test', 0),
                n_out_train=ad.get('n_out_train', 0),
                pct_train=ad.get('pct_train', 0)
            ))
        sections['ad'] = " ".join(ad_parts)
    else:
        sections['ad'] = "Применимая область не оценивалась."

    # ---- 7. Интерпретация ----
    interp = data.get('interpretation')
    if interp:
        interp_parts = [
            tpl['interpretation']['intro'].format(method=interp.get('method', 'анализа важности'))
        ]
        if interp.get('top_features'):
            interp_parts.append(tpl['interpretation']['top_features'].format(
                top_list=_format_list(interp['top_features'], max_items=10)
            ))
        if interp.get('meaning'):
            interp_parts.append(tpl['interpretation']['meaning'].format(meaning=interp['meaning']))
        sections['interpretation'] = " ".join(interp_parts)
    else:
        sections['interpretation'] = "Интерпретация модели не проводилась."

    # ---- 8. Выводы ----
    r2_for_quality = metrics.get('test', {}).get('r2') or metrics.get('cv', {}).get('r2')
    quality = _get_quality_label(r2_for_quality)
    conclusion_parts = [
        tpl['conclusion']['summary'].format(quality=quality)
    ]
    if data.get('limitations'):
        conclusion_parts.append(tpl['conclusion']['limitations'].format(limitations=data['limitations']))
    if data.get('recommendations'):
        conclusion_parts.append(tpl['conclusion']['recommendations'].format(recommendations=data['recommendations']))
    sections['conclusion'] = " ".join(conclusion_parts)

    # ---- Сборка HTML ----
    html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><title>{sections['title']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 40px auto; line-height: 1.6; }}
        h1, h2 {{ color: #1a3a5c; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ccc; padding: 8px; text-align: center; }}
        th {{ background: #e6edf5; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        .section {{ margin-bottom: 30px; }}
    </style>
    </head>
    <body>
    <h1>{sections['title']}</h1>
    <div class="section"><h2>1. Цель и конечная точка</h2><p>{sections['purpose']}</p></div>
    <div class="section"><h2>2. Данные</h2><p>{sections['data']}</p></div>
    """

    # Вставка графиков, если есть
    plots = data.get('plots', {})
    if plots.get('histogram'):
        html += f'<img src="data:image/png;base64,{_fig_to_base64(plots["histogram"])}" alt="Гистограмма распределения свойства">'

    html += f"""
    <div class="section"><h2>3. Дескрипторы</h2><p>{sections['descriptors']}</p></div>
    <div class="section"><h2>4. Метод моделирования</h2><p>{sections['model']}</p></div>
    <div class="section"><h2>5. Результаты и метрики</h2><p>{sections['results']}</p>
    """

    if plots.get('scatter'):
        html += f'<img src="data:image/png;base64,{_fig_to_base64(plots["scatter"])}" alt="График эксперимент-прогноз">'
    if plots.get('residuals'):
        html += f'<img src="data:image/png;base64,{_fig_to_base64(plots["residuals"])}" alt="График остатков">'
    if plots.get('y_rand_hist'):
        html += f'<img src="data:image/png;base64,{_fig_to_base64(plots["y_rand_hist"])}" alt="Y-рандомизация">'

    html += f"""
    </div>
    <div class="section"><h2>6. Применимая область</h2><p>{sections['ad']}</p>
    """
    if plots.get('williams'):
        html += f'<img src="data:image/png;base64,{_fig_to_base64(plots["williams"])}" alt="График Вильямса">'
    html += f"""
    </div>
    <div class="section"><h2>7. Интерпретация модели</h2><p>{sections['interpretation']}</p></div>
    <div class="section"><h2>8. Выводы</h2><p>{sections['conclusion']}</p></div>
    <p><em>Отчёт сгенерирован {datetime.now().strftime("%Y-%m-%d %H:%M")}</em></p>
    </body>
    </html>
    """

    return {
        'html': html,
        'sections': sections,
        'metadata': {
            'target': data.get('target_col'),
            'n_final': data.get('n_final'),
            'r2_test': metrics.get('test', {}).get('r2'),
            'r2_cv': metrics.get('cv', {}).get('r2'),
            'quality': quality
        }
    }