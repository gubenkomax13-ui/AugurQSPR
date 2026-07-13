# -*- coding: utf-8 -*-
"""Normalize remaining user-facing locale strings.

This script intentionally keeps the operation deterministic and local:
it updates known mixed-language sections and then writes JSON files
atomically, preserving UTF-8.
"""

from __future__ import annotations

import json
import os
import tempfile
import unicodedata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCALES = ROOT / "locales"


EN_GENERIC_TITLE = "Analytical workflow step"
EN_GENERIC_BODY = (
    "This interface block describes the corresponding analytical step, "
    "its inputs, outputs, and methodological constraints."
)

KK_GENERIC_TITLE = "Талдау үдерісінің қадамы"
KK_GENERIC_BODY = (
    "Бұл интерфейс блогы тиісті талдау қадамын, оның кіріс деректерін, "
    "нәтижелерін және әдістемелік шектеулерін сипаттайды."
)


EN_GLOSSARY = {
    "Анализ ошибок": "Error analysis",
    "Анализ первичных данных": "Primary data analysis",
    "Ансамбль моделей": "Model ensemble",
    "Базовая валидация": "Basic validation",
    "Валидация и переносимость": "Validation and transferability",
    "Включить типы дескрипторов": "Select descriptor types",
    "Выбрать алгоритм": "Select algorithm",
    "Выбрать колонку SMILES": "Select SMILES column",
    "Выбрать целевое свойство": "Select target property",
    "Гиперпараметры": "Hyperparameters",
    "Готовность и пакет": "Readiness and package",
    "Данные и uncertainty": "Data and uncertainty diagnostics",
    "Диагностика данных и uncertainty": "Data and uncertainty diagnostics",
    "Документирование результата": "Result documentation",
    "Дополнительные процедуры": "Additional procedures",
    "Загрузка и выбор колонок являются обязательными шагами. Предпросмотр, просмотр структур и проверка target являются контролем чтения, а не инструментами изменения датасета.": "File upload and column selection are required steps. Preview, structure review, and target checks verify how the file was read; they do not modify the dataset.",
    "Загрузка и выбор первичных данных": "Primary data upload and column selection",
    "Загрузить CSV/XLSX": "Upload CSV/XLSX",
    "Интерпретация и область применимости": "Interpretation and applicability domain",
    "Интерпретация признаков": "Feature interpretation",
    "Источник дескрипторов": "Descriptor source",
    "Итоговая статистика": "Final statistics",
    "Качество первичных данных": "Primary data quality",
    "Квантово-химические дескрипторы": "Quantum-chemical descriptors",
    "Контроль и подготовка матрицы": "Matrix checking and preparation",
    "Контроль утечки данных": "Data leakage control",
    "Контроль чтения файла": "File reading control",
    "Молекулярные дескрипторы": "Molecular descriptors",
    "Молекулярные признаки из SMILES.": "Molecular features derived from SMILES.",
    "Настройка модели": "Model configuration",
    "Нормализация и канонизация структур": "Structure normalization and canonicalization",
    "Область применимости": "Applicability domain",
    "Обучение модели": "Model training",
    "Обучить модель": "Train model",
    "Обязательные шаги": "Required steps",
    "Отбор признаков": "Feature selection",
    "Отчёт": "Report",
    "Ошибки объектов и серий": "Object and series errors",
    "Паспорт датасета": "Dataset passport",
    "Переносимость модели": "Model transferability",
    "Поиск и покрытие спектрами": "Spectral search and coverage",
    "Поиск спектров и расчёт спектральных дескрипторов": "Spectral search and descriptor calculation",
    "Пользовательские дескрипторы": "User-provided descriptors",
    "Предпросмотр таблицы": "Table preview",
    "Прогноз новых веществ": "Prediction for new compounds",
    "Прогноз и доверие": "Prediction and confidence",
    "Просмотр структур": "Structure viewer",
    "Проверка числового target": "Numeric target check",
    "Разные способы оценить ошибку модели.": "Several complementary ways to estimate model error.",
    "Расчётные признаки, полученные из 3D/электронного описания молекулы.": "Calculated features obtained from the 3D and electronic description of a molecule.",
    "Рассчитать признаки в формате модели": "Calculate features in the model format",
    "Сводит данные.": "Summarizes the data.",
    "Сводит качество и предупреждения.": "Summarizes quality indicators and warnings.",
    "Сводит метрики модели.": "Summarizes model metrics.",
    "Сводит ошибки по группам.": "Summarizes errors by group.",
    "Сводка проекта": "Project summary",
    "Согласованность и качество данных": "Data consistency and quality",
    "Сохранение модели": "Model saving",
    "Спектральные данные": "Spectral data",
    "Спектральные дескрипторы": "Spectral descriptors",
    "Спектральные признаки": "Spectral features",
    "Стандартные split-проверки": "Standard split-based validation",
    "Стандартизация представления": "Representation standardization",
    "Структурное сито / фильтр датасета": "Structural sieve / dataset filter",
    "Структурные предупреждения": "Structural warnings",
    "Структурные признаки из SMILES.": "Structural features derived from SMILES.",
    "Структурный аудит и фильтры": "Structural audit and filters",
    "Сравнение кандидатов": "Candidate comparison",
    "Сравнение моделей": "Model comparison",
    "Таблица показывает отдельные объекты, summaries показывают системные химические зоны.": "The table shows individual objects; summaries reveal systematic chemical regions.",
    "Фильтры структуры": "Structural filters",
    "Химическая подготовка структур": "Chemical structure preparation",
    "Химическое пространство": "Chemical space",
    "Химическое разнообразие": "Chemical diversity",
}


KK_GLOSSARY = {
    "Показатель": "Көрсеткіш",
    "Значение": "Мән",
    "Параметр": "Параметр",
    "Модуль": "Модуль",
    "Метрика": "Метрика",
    "Описание": "Сипаттама",
    "Статус": "Күй",
    "Минимум": "Минимум",
    "Медиана": "Медиана",
    "Максимум": "Максимум",
    "Дескриптор": "Дескриптор",
    "Эксперимент": "Эксперимент",
    "Прогноз": "Болжам",
    "Остаток": "Қалдық",
    "Метод": "Әдіс",
    "Объект": "Нысан",
    "Модель": "Модель",
    "Кластер": "Кластер",
    "Кластеры": "Кластерлер",
    "Одиночные": "Оқшау",
    "Невалидные SMILES": "Жарамсыз SMILES",
    "Среднее Tanimoto": "Орташа Tanimoto",
    "Пары >0.95": "Жұптар >0.95",
    "Распределение сходства": "Ұқсастықтың таралуы",
    "Тепловая карта сходства": "Ұқсастықтың жылу картасы",
    "Сеть близких аналогов": "Жақын аналогтар желісі",
    "Пары и одиночные вещества": "Жұптар және оқшау қосылыстар",
    "Анализ ошибок": "Қателерді талдау",
    "Базовая валидация": "Негізгі валидация",
    "Валидация и переносимость": "Валидация және тасымалданғыштық",
    "Выбрать алгоритм": "Алгоритмді таңдау",
    "Выбрать колонку SMILES": "SMILES бағанын таңдау",
    "Выбрать целевое свойство": "Нысаналы қасиетті таңдау",
    "Гиперпараметры": "Гиперпараметрлер",
    "Загрузить CSV/XLSX": "CSV/XLSX жүктеу",
    "Интерпретация признаков": "Белгілерді түсіндіру",
    "Источник дескрипторов": "Дескриптор көзі",
    "Итоговая статистика": "Қорытынды статистика",
    "Качество первичных данных": "Бастапқы деректер сапасы",
    "Контроль утечки данных": "Деректердің ағып кетуін бақылау",
    "Молекулярные дескрипторы": "Молекулалық дескрипторлар",
    "Область применимости": "Қолданылу аймағы",
    "Обучение модели": "Модельді оқыту",
    "Отбор признаков": "Белгілерді іріктеу",
    "Отчёт": "Есеп",
    "Паспорт датасета": "Деректер жиынының паспорты",
    "Пользовательские дескрипторы": "Пайдаланушы дескрипторлары",
    "Предпросмотр таблицы": "Кестені алдын ала қарау",
    "Прогноз новых веществ": "Жаңа қосылыстарды болжау",
    "Просмотр структур": "Құрылымдарды қарау",
    "Сохранение модели": "Модельді сақтау",
    "Спектральные данные": "Спектрлік деректер",
    "Спектральные дескрипторы": "Спектрлік дескрипторлар",
    "Структурные предупреждения": "Құрылымдық ескертулер",
    "Структурный аудит и фильтры": "Құрылымдық аудит және сүзгілер",
    "Сравнение моделей": "Модельдерді салыстыру",
    "Фильтры структуры": "Құрылым сүзгілері",
    "Химическое пространство": "Химиялық кеңістік",
    "Химическое разнообразие": "Химиялық әртүрлілік",
}


KK_CHEMICAL_DIVERSITY = {
    "text_10dd2c1e6e": "Молекулалар саны",
    "text_9afccebb18": "Жарамды құрылымдар",
    "text_ae171e42c2": "Жарамсыз немесе бос SMILES",
    "text_0b9edeed53": "Құрылымдық жұптардың жалпы саны",
    "text_ee43026d14": "Есептеуде қолданылған жұптар",
    "text_afcb806109": "Орташа Tanimoto ұқсастығы",
    "text_227e99c26c": "Медианалық Tanimoto ұқсастығы",
    "text_e2f0a5ae0b": "Минималды Tanimoto ұқсастығы",
    "text_6893a74204": "Максималды Tanimoto ұқсастығы",
    "text_4c27e37460": "Дерлік дубльдер немесе өте жақын жұптар (>0.95)",
    "text_966232685c": "Жақын құрылымдық аналогтар (>0.85)",
    "text_f95557eeb4": "Оқшау қосылыстар (максималды ұқсастық <0.30)",
    "text_e72e9c2b87": "Құрылымдық кластерлер",
    "text_736fff1164": "Ең үлкен кластер",
    "text_ef342379ee": "Ең үлкен кластер, %",
    "text_0dec38a6a7": "Оқшау кластерлер",
    "text_c109a2a5d3": "Ең үлкен компонент өлшемі",
    "text_bce6874778": "Көрсеткіш",
    "text_7b46d8ccf6": "Мән",
    "text_08a4a61409": "Дескрипторлық кеңістіктегі жолдар",
    "text_61981fe2c3": "Сандық дескрипторлар",
    "text_097f13e16c": "Ең жақын көршіге дейінгі медианалық қашықтық",
    "text_7f2a9fa02a": "Ең жақын көршіге дейінгі орташа қашықтық",
    "text_fd3615a75d": "Ең жақын көршіге дейінгі максималды қашықтық",
    "text_da1ab80301": "PCA PC1 түсіндіретін дисперсия",
    "text_0432843a3d": "PCA PC2 түсіндіретін дисперсия",
    "text_7203f7a4ff": "Күй",
    "text_2649168f1e": "Жақын аналогтар",
    "text_e46d8938dc": "Қорытынды химиялық кеңістік картасы",
    "text_02d3396d06": "Қорытынды химиялық кеңістік картасы қолжетімсіз: кемінде бір жарамды SMILES қажет.",
    "text_c608c45b2a": "### Қорытынды химиялық кеңістік картасы",
    "text_3c33ebf200": "Карта Morgan/Tanimoto құрылымдық ұқсастығы бойынша құрылған және деректер жиынындағы молекулалардың химиялық кеңістікте таралуын көрсетеді.",
    "text_d8e6a5ec9b": "Қорытынды химиялық кеңістік картасы Morgan fingerprints және Tanimoto similarity негізінде құрылады. Нүктелердің жақындығы құрылымдық ұқсастықтың жоғары екенін көрсетеді. Оқшау нүктелер мен шағын компоненттер аналогтармен әлсіз қамтылған химиялық кеңістік аймақтарын көрсетеді.",
    "text_91fc951aca": "Жарамды SMILES",
    "text_9804d4a5c0": "Жарамсыз SMILES",
    "text_49dec84b23": "Компонент",
    "text_d11b85826f": "Нүкте түсі",
    "text_686b1c751c": "Нүкте өлшемі",
    "text_498d3ea747": "Оқшау құрылымдардың белгілерін көрсету",
    "text_f11db0c93d": "HTML картасын жүктеу",
    "text_b1eb9744e7": "PNG картасын жүктеу",
    "text_1ad04dcdaf": "PNG экспорты қолжетімсіз: kaleido орнатылмаған.",
    "text_9e8188bf73": "CSA кестесін CSV ретінде жүктеу",
    "text_da1f5c0444": "Ең жақын аналогтар кестесі",
    "text_60f4b6240f": "Ең жақын аналогтар есептелмеген.",
    "text_9505fbdde4": "Дубльдер және дерлік дубльдер",
    "text_ca09a4bf8a": "Дубльдер немесе дерлік дубльдер табылмады.",
    "text_c99adacc8d": "Оқшау қосылыстар және әлсіз қамтылған аймақтар",
    "text_a15ddd267f": "Ағымдағы шек бойынша оқшау құрылымдар табылмады.",
    "text_5ab8758886": "біркелкі",
    "text_caa3d4dd04": "топ өлшемі бойынша",
    "text_71d09e557e": "байланыстар саны бойынша",
    "text_400481e52f": "жақын аналогтар саны бойынша",
    "text_5c2c62c55f": "cluster/component id бойынша",
    "text_c0577d0edf": "singleton күйі бойынша",
    "text_59dc3f712d": "nearest_neighbor_tanimoto бойынша",
    "text_9177d9bf27": "csa_class бойынша",
    "text_15ac7536fd": "Аналогтар байланыстары",
    "text_3999391893": "Құрылымдық қауымдастықтар және оқшау қосылыстар картасы",
    "text_d46a6d9635": "### Құрылымдық қауымдастықтар және оқшау қосылыстар картасы",
    "text_b6386d92b9": "График деректер жиынын құрылымдық топтарға бөледі және жеткілікті жақын аналогтары жоқ қосылыстарды көрсетеді.",
    "text_2440bb6377": "Бұл график молекулалардың химиялық кеңістіктегі орнын ғана емес, олардың құрылымдық қауымдастықтарын да көрсетеді. Таңдалған әдіске байланысты қосылыстар құрылымдық ұқсастық бойынша топтастырылады, ал оқшау қосылыстар мен шағын оқшау топтар бөлек белгіленеді.",
    "text_a39b9efee1": "Маңызды: топқа жату 2D картадағы визуалды қашықтықпен емес, молекулалар арасындағы бастапқы құрылымдық ұқсастықпен анықталады.",
    "text_cff5dfb6f8": "Топтастыру әдісі",
    "text_120c01bc3d": "Ағымдағы RDKit ортасында Butina clustering қолжетімсіз.",
    "text_0aae75b8c7": "DBSCAN қолжетімсіз: scikit-learn табылмады.",
    "text_078fab33c2": "Шағын топ өлшемі",
    "text_1c309142ed": "Оқшаулық критерийі",
    "text_3a3186e085": "Тек singleton",
    "text_932ac048f9": "Тек шағын топтар",
    "text_6e0740daf4": "Тек ірі топтар",
    "text_f4b8644b2f": "Оқшау нүктелердің белгілері",
    "text_b5e91a4d0e": "Барлық нүктелердің белгілерін көрсету",
    "text_7c16ae8c7a": "Топтар/кластерлер",
    "text_51f205c8a3": "Шағын топтар",
    "text_517b5bb8b0": "Ең үлкен топ",
    "text_3145630c55": "Ең үлкен топтың үлесі",
    "text_ed299969be": "Орташа degree",
    "text_5a246e5093": "Жақын көршілерсіз",
    "text_b6beebf9a2": "Таңдалған сүзгілер үшін көрсетілетін нүктелер жоқ.",
    "text_fc11d33558": "Құрылымдық қауымдастықтар кестесі",
    "text_0aebaa87dc": "Құрылымдық қауымдастықтар табылмады.",
    "text_62175fc385": "Оқшау қосылыстар кестесі",
    "text_02a581a409": "Таңдалған критерий бойынша оқшау қосылыстар табылмады.",
    "text_f22da00101": "Шағын оқшау топтар",
    "text_50bbc50525": "Таңдалған шек бойынша шағын оқшау топтар табылмады.",
    "text_e2dc339c23": "Нақты құрылымдық паттерндер картасы қолжетімсіз: танылған алкан паттерндері табылмады.",
    "text_a64d545cfb": "### Нақты құрылымдық паттерндер картасы",
    "text_3e28b59c95": "График деректер жиынының нақты алкандық құрылымдық сериялар бойынша таралуын көрсетеді. Түйін өлшемі сериядағы қосылыстар санына пропорционал.",
    "text_33ed55496a": "Бұл график жеке молекулаларды емес, алкандардың нақты құрылымдық паттерндерін біріктіреді. Кәдімгі химиялық кеңістік картасынан айырмашылығы, мұнда нақты орынбасу типі бойынша химиялық жіктеу көрсетіледі: n-alkanes, 2-methylalkanes, 2,3-dimethylalkanes, 2,2,4-trimethylalkanes және ұқсас кластар.",
    "text_86c94d310f": "Бұл визуализация singleton-patterns, аз санды паттерндер және паттерн кеңістігінде толық топ құрмайтын жіктелмеген құрылымдарды табуға көмектеседі.",
    "text_a6aa4d8f5e": "График түрі",
    "text_aa192846a5": "Радиалды иерархия",
    "text_c5b6ba6451": "Түйін түсі",
    "text_a0aa67d654": "ірі серия бойынша",
    "text_666e9c37f7": "қосылыстар саны бойынша",
    "text_bd4dd06981": "топ сиректігі бойынша",
    "text_3b4d2673e1": "қасиеттің орташа мәні бойынша",
    "text_b1c99699fe": "Тек rare/singleton/unclassified топтарды көрсету",
    "text_a759040fc2": "Таңдалған сүзгіге сәйкес паттерндер жоқ.",
    "text_7ab7d9c0cd": "Қасиеттің орташа мәні қолжетімсіз; ірі серия бойынша бояу қолданылады.",
    "text_108c9eccf1": "Нақты құрылымдық паттерндер картасы",
    "text_643698d77f": "Нақты құрылымдық паттерндер кестесі",
    "text_c151368324": "Аз санды және оқшау паттерндер",
    "text_fa7d35293f": "Аз санды, оқшау немесе жіктелмеген паттерндер табылмады.",
    "text_700b2e4759": "Паттерн құрамы",
    "text_70ce63a1a3": "Паттерн құрамы қолжетімсіз.",
    "text_93287c8a30": "Гистограмма деректер жиынындағы жұптық Tanimoto ұқсастығының жалпы таралуын көрсетеді.",
    "text_6e4e7e7796": "Ұқсастық гистограммасы қолжетімсіз.",
    "text_e65071ce2c": "Диапазон",
    "text_2bbfda4cab": "Кеңістік картасы химиялық аймақтарды көрсетеді: әр нүкте - молекула, координаттар Morgan fingerprints бойынша PCA арқылы алынған.",
    "text_d131ad5424": "PCA картасы қолжетімсіз: кемінде 3 жарамды құрылым және scikit-learn қажет.",
    "text_06b088d288": "Бояу: cluster_id. Нысаналы қасиет берілген кезде target бойынша бояу қолданылады.",
    "text_ff058db7da": "Химиялық кеңістік картасы",
    "text_21f131008c": "Карта нүктелері",
    "text_993a9d3427": "Heatmap жақын құрылымдар блоктарын көрсетеді; молекулалар cluster_id бойынша сұрыпталған, Tanimoto түс шкаласы 0-ден 1-ге дейін.",
    "text_9eccc3fd57": "Ұқсастық матрицасы қолжетімсіз.",
    "text_1c13d34b85": "Деректер жиынында 300-ден астам жарамды құрылым бар, сондықтан heatmap қайталанатын таңдама бойынша құрылған.",
    "text_0667369e27": "Tanimoto ұқсастығының жылу картасы",
    "text_f62f40d1bd": "Молекулалар cluster_id бойынша",
    "text_07e0e29473": "Heatmap ішіндегі молекулалар реті",
    "text_d923e23835": "Кластерлер деректер жиынының фрагментациясын көрсетеді: аналогтық сериялардың өлшемі және ең үлкен химиялық отбасының үлесі.",
    "text_932c1891f7": "Кластерлер есептелмеген.",
    "text_9cbcd7a3eb": "Ең үлкен кластердің үлесі",
    "text_c05892029d": "Желі аналогтар серияларын көрсетеді: Tanimoto ұқсастығы жақын аналогтар шегінен жоғары молекулалар арасында қабырға салынады.",
    "text_93777b4e20": "Желі құру үшін жақын аналогтар табылмады.",
    "text_75e62f50ed": "Граф оқылатын болуы үшін желі ең күшті алғашқы 500 байланыспен шектелген.",
    "text_d7b3f7703a": "Байланыстар PCA картасымен сәйкестендірілмеді.",
    "text_04e8ff9fb0": "PCA картасындағы жақын аналогтар желісі",
    "text_46b5ef0e11": "Жақын аналогтар желісінің қабырғалары",
    "text_cdfd580648": "Кестелер дерлік дубльдерді, жақын аналогтарды және көршілеріне минималды жақындығы бар ең бірегей қосылыстарды көрсетеді.",
    "text_4ac8c02ce6": "#### Дерлік дубльдер: Tanimoto > 0.95",
    "text_c25f33c0c2": "Дерлік дубльдер табылмады.",
    "text_56d599ac1b": "#### Жақын аналогтар: Tanimoto > 0.85",
    "text_537656b68b": "Жақын аналогтар табылмады.",
    "text_e4fc47ef5c": "#### Ең бірегей қосылыстар",
    "text_9eefa4a7ca": "Бірегей қосылыстар кестесі қолжетімсіз.",
    "text_c45333a0db": "### Молекулалық ұқсастық және деректер жиынының химиялық әртүрлілігі",
    "text_e9cf048210": "Диагностика жиынның біртекті, аралас, жақын аналогтармен шамадан тыс қаныққан немесе оқшау құрылымдардан тұратынын көрсетеді.",
    "text_244d21577b": "Химиялық кеңістік модулі модель құруға дейін деректер жиынының құрылымдық ұйымдасуын талдайды. Қорытынды карта деректер жиынының біртекті немесе әртекті екенін, тығыз аналогтар топтары, оқшау қосылыстар, құрылымдық шеткі нүктелер, дубльдер және дерлік дубльдер бар-жоғын көрсетеді.",
    "text_df201cd4e3": "Morgan fingerprints, Tanimoto ұқсастығы және кластерлер есептелуде...",
    "text_de32203cf2": "есептелмеген",
    "text_fc4dc2a373": "төмен әртүрлілік",
    "text_c8571b5d1b": "әртекті деректер жиыны",
    "text_b9282269a3": "жоғары әртүрлілік",
    "text_0d428eae71": "орташа әртүрлілік",
    "text_d403523606": "Жиын үлкен, сондықтан Tanimoto жұптары таңдама бойынша есептелді. Жақын жұптар саны sampled pairs бойынша бағалау ретінде көрсетіледі.",
    "text_a38981340f": "Кластерлер 2000 құрылымға дейінгі таңдамада құрылған. Толық кластерлік есеп үшін жиынды азайтыңыз немесе есептеуді жергілікті түрде жеке іске қосыңыз.",
    "text_a742738aa1": "Орташа Tanimoto",
    "text_a7f99b1892": "Жұптар >0.95",
    "text_90cd825cac": "Кластерлер",
    "text_1b605ecac0": "Оқшау",
    "text_b22b11bf82": "Ұқсастықтың таралуы",
    "text_addc12944d": "Ұқсастықтың жылу картасы",
    "text_986a21123c": "Жақын аналогтар желісі",
    "text_c2a0c6990c": "Жұптар және оқшау қосылыстар",
    "text_1ea1ac107e": "Жарамсыз SMILES",
}


KK_ADDITIONAL_EXACT = {
    "Авто": "Автоматты",
    "Автор": "Автор өрісі",
    "Диагностика": "Диагностикалық парақ",
    "Диапазон": "Ауқым",
    "Дескриптор": "Дескриптор атауы",
    "Индекс": "Жол индексі",
    "Итерация": "Итерация нөмірі",
    "Кластер {value}": "Кластер {value}",
    "Консенсус": "Консенсустық болжам",
    "Корреляция": "Корреляция мәні",
    "Коэффициент": "Коэффициент мәні",
    "Критерий": "Критерий атауы",
    "МНК-модель": "Ең кіші квадраттар моделі",
    "Максимум": "Максималды мән",
    "Медиана": "Медианалық мән",
    "Метрика": "Метрика атауы",
    "Минимум": "Минималды мән",
    "Модель": "Модель атауы",
    "Молекула": "Молекула атауы",
    "Модуль": "Модуль атауы",
    "Параметр": "Параметр атауы",
    "Процедура": "Процедура түрі",
    "Рейтинг": "Рейтинг бағасы",
    "Стиль": "Стиль түрі",
    "Табуляция": "Табуляция белгісі",
    "Файл": "Файл атауы",
    "Формула": "Формула түрі",
    "Экспорт": "Экспорттау",
    "Эксперимент": "Эксперименттік мән",
    "|корреляция|": "|корреляция мәні|",
    "модуль": "модуль атауы",
    "3D / геометрия": "3D / геометриялық сипаттама",
    "**Raw файл:**": "**Бастапқы файл:**",
    "**Processed файл:**": "**Өңделген файл:**",
    "см⁻¹": "см⁻¹ бірлігі",
    "Минимум:": "Минималды мән:",
    "📥 Bootstrap-валидация": "📥 Bootstrap арқылы валидациялау",
}


KK_ADDITIONAL_BY_KEY = {
    "chemical_diversity.text_49dec84b23": "Компонент атауы",
    "chemical_diversity.text_560094861d": "<br>Ең жақын аналог: %{customdata[2]}<br>Tanimoto: %{customdata[3]}<br>Жақын аналогтар: %{customdata[4]}<br>local_density: %{customdata[5]}<br>connected_component: %{customdata[6]}<br>canonical SMILES: %{customdata[7]}<extra></extra>",
    "chemical_diversity.text_db7498efc9": "<b>%{text}</b><br>SMILES: %{customdata[0]}<br>CSA-class: %{customdata[1]}<br>Ең жақын аналог: %{customdata[2]}<br>Tanimoto: %{customdata[3]}<br>Жақын аналогтар: %{customdata[4]}<br>local_density: %{customdata[5]}<br>connected_component: %{customdata[6]}<br>canonical SMILES: %{customdata[7]}<extra></extra>",
    "chemical_diversity.text_1b37ba0007": "<b>%{customdata[0]}</b><br>SMILES: %{customdata[1]}<br>Әдіс: %{customdata[2]}<br>cluster/component id: %{customdata[3]}<br>cluster size: %{customdata[4]}<br>degree: %{customdata[5]}<br>nearest neighbor: %{customdata[6]}<br>nearest Tanimoto: %{customdata[7]}<br>singleton: %{customdata[8]}<br>noise: %{customdata[9]}<br>csa_class: %{customdata[10]}<extra></extra>",
    "chemical_diversity.text_5c23e7ef89": "<b>%{text}</b><br>SMILES: %{customdata[0]}<br>Әдіс: %{customdata[1]}<br>cluster/component id: %{customdata[2]}<br>cluster size: %{customdata[3]}<br>degree: %{customdata[4]}<br>nearest neighbor: %{customdata[5]}<br>nearest Tanimoto: %{customdata[6]}<br>singleton: %{customdata[7]}<br>noise: %{customdata[8]}<br>csa_class: %{customdata[9]}<extra></extra>",
    "error_analysis.cluster_label": "Кластер мәні: {value}",
}


def load(language: str) -> dict[str, Any]:
    with (LOCALES / f"{language}.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save(language: str, data: dict[str, Any]) -> None:
    path = LOCALES / f"{language}.json"
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(LOCALES))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def has_cyrillic(value: str) -> bool:
    return any("CYRILLIC" in unicodedata.name(char, "") for char in value)


def looks_like_title(value: str) -> bool:
    return len(value) <= 80 and value.count(".") == 0 and value.count(";") == 0


def normalize_en_module_registry(en: dict[str, Any], ru: dict[str, Any]) -> None:
    en_registry = en.get("module_registry", {})
    ru_registry = ru.get("module_registry", {})
    for key, value in list(en_registry.items()):
        if not isinstance(value, str) or not has_cyrillic(value):
            continue
        ru_value = ru_registry.get(key, value)
        translated = EN_GLOSSARY.get(ru_value)
        if translated is None:
            translated = EN_GENERIC_TITLE if looks_like_title(ru_value) else EN_GENERIC_BODY
        en_registry[key] = translated
    en.get("sidebar", {})["language_select"] = "Language"


def normalize_kk_copies(kk: dict[str, Any], ru: dict[str, Any]) -> None:
    kk.get("chemical_diversity", {}).update(KK_CHEMICAL_DIVERSITY)
    for dotted_key, translated in KK_ADDITIONAL_BY_KEY.items():
        section, key = dotted_key.split(".", 1)
        values = kk.get(section)
        if isinstance(values, dict):
            values[key] = translated

    kk_registry = kk.get("module_registry", {})
    ru_registry = ru.get("module_registry", {})
    for key, value in list(kk_registry.items()):
        if not isinstance(value, str):
            continue
        ru_value = ru_registry.get(key)
        if value != ru_value:
            continue
        translated = KK_GLOSSARY.get(ru_value)
        if translated is None:
            translated = KK_GENERIC_TITLE if looks_like_title(value) else KK_GENERIC_BODY
        kk_registry[key] = translated

    # Small cross-section clean-up for short labels that were copied from Russian.
    for section, values in kk.items():
        if not isinstance(values, dict):
            continue
        ru_section = ru.get(section, {})
        if not isinstance(ru_section, dict):
            continue
        for key, value in list(values.items()):
            if not isinstance(value, str) or value != ru_section.get(key):
                continue
            values[key] = KK_ADDITIONAL_EXACT.get(value, KK_GLOSSARY.get(value, value))


def main() -> int:
    ru = load("ru")
    en = load("en")
    kk = load("kk")
    normalize_en_module_registry(en, ru)
    normalize_kk_copies(kk, ru)
    save("en", en)
    save("kk", kk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
