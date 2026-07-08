# -*- coding: utf-8 -*-

"""Human-readable registry of Augur QSPR user-facing modules."""

MODULES = {
    "data_preparation": {
        "title": "Загрузка и выбор первичных данных",
        "goal": "Завести таблицу в проект и выбрать минимально необходимые колонки для дальнейшего анализа.",
        "blocks": [
            {
                "name": "Обязательные шаги",
                "kind": "steps",
                "items": [
                    {
                        "name": "Загрузить CSV/XLSX",
                        "purpose": "Создаёт исходный рабочий датасет проекта.",
                        "updates": "st.session_state.data",
                        "kind": "step",
                    },
                    {
                        "name": "Выбрать колонку SMILES",
                        "purpose": "Фиксирует, где находится структурное представление молекул.",
                        "updates": "рабочий smiles_col",
                        "kind": "step",
                    },
                    {
                        "name": "Выбрать целевое свойство",
                        "purpose": "Фиксирует endpoint для моделирования.",
                        "updates": "target_col и числовой target",
                        "kind": "step",
                    },
                ],
            },
            {
                "name": "Контроль чтения файла",
                "purpose": "Помогает убедиться, что файл прочитан так, как ожидалось.",
                "kind": "control",
                "items": [
                    {"name": "Предпросмотр таблицы", "purpose": "Показывает первые строки и формат колонок.", "kind": "control"},
                    {"name": "Просмотр структур", "purpose": "Показывает молекулы из выбранной SMILES-колонки.", "kind": "control"},
                    {"name": "Проверка числового target", "purpose": "Показывает, можно ли использовать выбранную колонку как числовой endpoint.", "kind": "control"},
                ],
            },
        ],
        "difference": (
            "Загрузка и выбор колонок являются обязательными шагами. "
            "Предпросмотр, просмотр структур и проверка target являются контролем чтения, а не инструментами изменения датасета."
        ),
    },
    "primary_data_analysis": {
        "title": "Анализ первичных данных",
        "goal": "Понять, что за датасет загружен, до любых химических преобразований и фильтраций.",
        "blocks": [
            {
                "name": "Паспорт датасета",
                "purpose": "Даёт общую сводку по загруженной таблице.",
                "items": [
                    {
                        "name": "Паспорт датасета",
                        "purpose": "Показывает размер, полноту, диапазон свойства, валидные структуры и общий статус датасета.",
                        "kind": "analysis",
                    },
                ],
            },
            {
                "name": "Качество первичных данных",
                "purpose": "Диагностирует проблемы исходной таблицы и целевого свойства.",
                "items": [
                    {
                        "name": "Согласованность и качество данных",
                        "purpose": "Показывает конфликтующие дубли, подозрительные значения и первичные выбросы свойства. IQR, z-score и графики являются частью отчёта, а не отдельными инструментами.",
                        "kind": "analysis",
                    },
                ],
            },
        ],
        "difference": (
            "Паспорт описывает датасет целиком, а диагностика качества показывает проблемы и подозрительные случаи для проверки."
        ),
    },
    "chemical_correctness": {
        "title": "Химическая подготовка структур",
        "goal": (
            "При необходимости привести структуры к более сопоставимому химическому представлению. "
            "Инструмент не очищает датасет и не удаляет строки. Он приводит SMILES к единому виду: "
            "проверяет структуру, выбирает основной фрагмент, нормализует запись, нейтрализует заряды, "
            "строит canonical SMILES и InChIKey. Дубликаты и конфликтные записи только диагностируются; "
            "удаление выполняется отдельными инструментами качества данных."
        ),
        "blocks": [
            {
                "name": "Стандартизация представления",
                "purpose": "Приводит структурные записи к сопоставимому виду перед расчётом дескрипторов.",
                "items": [
                    {
                        "name": "Нормализация и канонизация структур",
                        "purpose": "Проверяет структуры, выбирает основной фрагмент, нормализует запись, строит canonical SMILES и диагностирует дубли. Таблицы преобразований и предупреждений являются результатом инструмента.",
                        "updates": "рабочий датасет только после нажатия кнопки применения",
                        "kind": "tool",
                    },
                ],
            },
        ],
        "difference": (
            "Модуль содержит один рабочий инструмент стандартизации; детали отчёта не являются отдельными инструментами."
        ),
    },
    "chemical_space": {
        "title": "Химическое пространство",
        "goal": "Понять, насколько разнообразен набор и какие химические области он покрывает.",
        "blocks": [
            {
                "name": "Химическое разнообразие",
                "purpose": "Оценивает разнообразие и покрытие химического пространства датасета.",
                "items": [
                    {
                        "name": "Близость молекул и химическое разнообразие датасета",
                        "purpose": "Показывает близость молекул, кластеры, близкие аналоги и одиночные структуры как части одного анализа.",
                        "kind": "analysis",
                    },
                ],
            },
        ],
        "difference": "Показатели близости и кластеризации являются результатами одного анализа химического разнообразия.",
    },
    "structural_audit": {
        "title": "Структурный аудит и фильтры",
        "goal": "Проверить, соответствует ли набор заданным структурным правилам и ограничениям области применения.",
        "blocks": [
            {
                "name": "Фильтры структуры",
                "purpose": "Задаёт правила отбора веществ по структурным признакам.",
                "items": [
                    {
                        "name": "Структурное сито / фильтр датасета",
                        "purpose": "Фильтрует датасет по элементам, SMARTS и функциональным группам; эти режимы являются настройками инструмента.",
                        "updates": "рабочий датасет после подтверждения",
                        "kind": "tool",
                    },
                ],
            },
            {
                "name": "Структурные предупреждения",
                "purpose": "Ищет химически спорные случаи для ручной проверки.",
                "items": [
                    {
                        "name": "Структурная проверка и контроль согласованности данных",
                        "purpose": "Показывает структурные предупреждения и таблицы ручной проверки как единый инструмент аудита.",
                        "kind": "tool",
                    },
                ],
            },
        ],
        "difference": "Фильтр меняет набор только после подтверждения; структурная проверка диагностирует спорные случаи.",
    },
    "spectral_data": {
        "title": "Спектральные данные",
        "goal": "Проверить и подключить спектры как дополнительный источник информации о молекулах.",
        "blocks": [
            {
                "name": "Поиск и покрытие спектрами",
                "purpose": "Показывает, насколько датасет обеспечен спектральными данными.",
                "items": [
                    {
                        "name": "Поиск спектров и расчёт спектральных дескрипторов",
                        "purpose": "Ищет IR/Mass спектры, показывает покрытие и рассчитывает спектральные признаки.",
                        "updates": "спектральная матрица признаков; QSPR-матрица после подключения",
                        "kind": "tool",
                    },
                ],
            },
            {
                "name": "Спектральные признаки",
                "purpose": "Преобразует спектры в числовое описание.",
                "items": [
                    {
                        "name": "Спектральные дескрипторы",
                        "purpose": "Подключает рассчитанные спектральные признаки как источник X.",
                        "kind": "tool",
                    },
                ],
            },
        ],
        "difference": "Поиск отвечает за наличие спектров, coverage за полноту, descriptors за превращение спектров в признаки.",
    },
    "feature_sources": {
        "title": "Источник дескрипторов",
        "goal": "Собрать дескрипторную матрицу X/y из выбранных источников признаков.",
        "required_steps": [
            {
                "name": "Выбрать способ получения дескрипторов",
                "purpose": "Определяет, дескрипторы рассчитываются программой или берутся из файла.",
                "updates": "режим расчёта или загрузки дескрипторов",
            },
            {
                "name": "Включить типы дескрипторов",
                "purpose": "Чекбоксы плюсуют источники: molecular + quantum + spectral + custom.",
                "updates": "выбранные источники итоговой матрицы",
            },
            {
                "name": "Настроить источники во вкладках",
                "purpose": "Каждая вкладка отвечает за настройки и расчёт своего типа признаков.",
                "updates": "параметры расчёта источников",
            },
            {
                "name": "Собрать итоговую матрицу X/y",
                "purpose": "Создаёт матрицу признаков и вектор целевого свойства для обучения модели.",
                "updates": "X_all, y_all, desc_names, df_desc",
            },
        ],
        "blocks": [
            {
                "name": "Обязательные шаги",
                "kind": "steps",
                "items": [
                    {
                        "name": "Выбрать способ получения дескрипторов",
                        "purpose": "Определяет, дескрипторы рассчитываются программой или берутся из файла.",
                        "updates": "режим расчёта или загрузки дескрипторов",
                        "kind": "step",
                    },
                    {
                        "name": "Включить типы дескрипторов",
                        "purpose": "Чекбоксы плюсуют источники: molecular + quantum + spectral + custom.",
                        "updates": "выбранные источники итоговой матрицы",
                        "kind": "step",
                    },
                    {
                        "name": "Собрать итоговую матрицу X/y",
                        "purpose": "Создаёт матрицу признаков и вектор целевого свойства для обучения модели.",
                        "updates": "X_all, y_all, desc_names, df_desc",
                        "kind": "step",
                    },
                ],
            },
            {
                "name": "Молекулярные дескрипторы",
                "purpose": "Структурные признаки из SMILES.",
                "items": [
                    {
                        "name": "Молекулярные дескрипторы",
                        "purpose": "Рассчитывает структурные признаки из SMILES; RDKit, Mordred, PaDEL и fingerprints являются наборами внутри инструмента.",
                        "kind": "tool",
                    },
                ],
            },
            {
                "name": "Квантово-химические дескрипторы",
                "purpose": "Расчётные признаки, полученные из 3D/электронного описания молекулы.",
                "items": [
                    {
                        "name": "Квантово-химические дескрипторы",
                        "purpose": "Рассчитывает 3D/электронные признаки; xTB, morfeus и DScribe являются источниками внутри инструмента.",
                        "kind": "tool",
                    },
                ],
            },
            {
                "name": "Спектральные дескрипторы",
                "purpose": "Признаки из IR/Mass спектров, подключаемые как источник X.",
                "items": [
                    {
                        "name": "Поиск спектров и расчёт спектральных дескрипторов",
                        "purpose": "Находит спектры, показывает покрытие и рассчитывает признаки из IR/Mass данных.",
                        "kind": "tool",
                    },
                    {
                        "name": "Спектральные дескрипторы",
                        "purpose": "Подключает уже рассчитанные спектральные признаки к итоговой матрице X/y.",
                        "kind": "tool",
                    },
                ],
            },
            {
                "name": "Пользовательские дескрипторы",
                "purpose": "Готовые числовые признаки из таблицы пользователя.",
                "items": [
                    {
                        "name": "Пользовательские дескрипторы и МНК-регрессия",
                        "purpose": "Берёт выбранные числовые колонки пользователя и позволяет проверить их через МНК.",
                        "kind": "tool",
                    },
                ],
            },
            {
                "name": "Контроль и подготовка матрицы",
                "purpose": "Проверяет матрицу признаков перед обучением.",
                "items": [
                    {
                        "name": "Контроль утечки данных",
                        "purpose": "Ищет признаки, которые могут напрямую содержать целевое свойство.",
                        "kind": "tool",
                    },
                    {
                        "name": "Очистить матрицу признаков",
                        "purpose": "Удаляет непригодные признаки и строки при подготовке к обучению.",
                        "updates": "рабочая матрица признаков",
                        "kind": "procedure",
                    },
                ],
            },
        ],
        "difference": (
            "Молекулярные признаки описывают структуру из SMILES, квантово-химические добавляют расчётную 3D/электронную информацию, "
            "спектральные используют IR/Mass-данные, пользовательские берутся из таблицы."
        ),
    },
    "training": {
        "title": "Обучение модели",
        "goal": "Построить QSPR-модель на выбранной матрице признаков.",
        "required_steps": [
            {"name": "Выбрать алгоритм", "purpose": "Без алгоритма модель не может быть обучена.", "updates": "model_type"},
            {"name": "Обучить модель", "purpose": "Создаёт обученный регрессор.", "updates": "trained_models"},
        ],
        "groups": [
            {
                "name": "Настройка модели",
                "purpose": "Необязательные способы улучшить обучение.",
                "tools": [
                    {"name": "Гиперпараметры", "purpose": "Управляют сложностью и регуляризацией."},
                    {"name": "Отбор признаков", "purpose": "Снижает шум и риск переобучения."},
                    {"name": "Train metrics", "purpose": "Показывают качество на обучающих данных."},
                ],
            },
        ],
        "difference": "Алгоритм и обучение обязательны; гиперпараметры и отбор признаков являются настраиваемыми инструментами.",
    },
    "validation": {
        "title": "Базовая валидация",
        "goal": "Получить первичную оценку качества модели на стандартных разбиениях.",
        "groups": [
            {
                "name": "Стандартные split-проверки",
                "purpose": "Разные способы оценить ошибку модели.",
                "tools": [
                    {"name": "Hold-out", "purpose": "Одна отложенная проверка."},
                    {"name": "K-Fold", "purpose": "Усреднение по фолдам."},
                    {"name": "LOO", "purpose": "Проверка для малых датасетов."},
                ],
            },
        ],
        "difference": "Hold-out быстрее, K-Fold устойчивее, LOO полезен для малых наборов.",
    },
    "advanced_validation": {
        "title": "Валидация и переносимость",
        "goal": "Проверить устойчивость модели и её переносимость на новую химию.",
        "groups": [
            {
                "name": "Устойчивость модели",
                "purpose": "Проверяет зависимость результата от случайности.",
                "tools": [
                    {"name": "Repeated K-Fold", "purpose": "Повторяет K-Fold с разными разбиениями."},
                    {"name": "Bootstrap", "purpose": "Оценивает разброс качества."},
                    {"name": "Y-randomization", "purpose": "Проверяет случайную связь."},
                ],
            },
            {
                "name": "Переносимость модели",
                "purpose": "Проверяет работу на новых химических областях.",
                "tools": [
                    {"name": "Scaffold split", "purpose": "Откладывает химические каркасы."},
                    {"name": "Group split", "purpose": "Откладывает заданные группы."},
                    {"name": "Cluster split", "purpose": "Откладывает структурные кластеры."},
                    {"name": "External-distance split", "purpose": "Проверяет экстраполяцию."},
                ],
            },
            {
                "name": "Диагностика данных и uncertainty",
                "purpose": "Проверяет, хватает ли данных и честны ли интервалы.",
                "tools": [
                    {"name": "Learning curves", "purpose": "Отделяют дефицит данных от переобучения."},
                    {"name": "Prediction interval coverage", "purpose": "Проверяет фактическое покрытие интервалов."},
                    {"name": "Split comparison", "purpose": "Сравнивает random/scaffold/cluster/group/external-distance split."},
                ],
            },
        ],
        "difference": "Robustness проверяет случайность, transferability проверяет новую химию, learning/coverage проверяют данные и uncertainty.",
    },
    "diagnostics": {
        "title": "Интерпретация и область применимости",
        "goal": "Понять, где модель работает, где ошибается и какие признаки влияют на прогноз.",
        "groups": [
            {
                "name": "Область применимости",
                "purpose": "Оценивает доверие к прогнозу.",
                "tools": [
                    {"name": "Applicability Domain", "purpose": "Проверяет близость к обучающей области."},
                    {"name": "Williams plot", "purpose": "Связывает leverage и остатки."},
                ],
            },
            {
                "name": "Интерпретация признаков",
                "purpose": "Показывает вклад дескрипторов.",
                "tools": [
                    {"name": "Scaled coefficients", "purpose": "Для линейных моделей."},
                    {"name": "Tree importance", "purpose": "Для деревьев и бустинга."},
                    {"name": "Permutation importance", "purpose": "Для любой модели."},
                    {"name": "SHAP", "purpose": "Для локального объяснения."},
                ],
            },
        ],
        "difference": "AD говорит о применимости, residuals об ошибках, importance/SHAP о поведении признаков.",
    },
    "error_analysis": {
        "title": "Анализ ошибок",
        "goal": "Найти вещества и химические серии, где модель ошибается системно.",
        "groups": [
            {
                "name": "Ошибки объектов и серий",
                "purpose": "Ищет локальные и групповые провалы модели.",
                "tools": [
                    {"name": "Error table", "purpose": "Сортирует вещества по ошибке."},
                    {"name": "Structural annotations", "purpose": "Размечает серии и scaffolds."},
                    {"name": "Series summaries", "purpose": "Сводит ошибки по группам."},
                ],
            },
        ],
        "difference": "Таблица показывает отдельные объекты, summaries показывают системные химические зоны.",
    },
    "model_comparison": {
        "title": "Сравнение моделей",
        "goal": "Выбрать модель по качеству, устойчивости и ограничениям.",
        "groups": [
            {
                "name": "Сравнение кандидатов",
                "purpose": "Сопоставляет обученные модели.",
                "tools": [
                    {"name": "Metrics table", "purpose": "Сравнивает численные метрики."},
                    {"name": "Rating", "purpose": "Сводит качество и предупреждения."},
                ],
            },
        ],
        "difference": "Метрики дают числа, rating помогает выбрать с учётом рисков.",
    },
    "consensus": {
        "title": "Consensus-прогноз",
        "goal": "Объединить несколько валидированных моделей в более устойчивый прогноз.",
        "groups": [
            {
                "name": "Ансамбль моделей",
                "purpose": "Определяет состав consensus.",
                "tools": [
                    {"name": "Top-N", "purpose": "Выбирает лучшие модели."},
                    {"name": "Quality threshold", "purpose": "Отсекает слабые модели."},
                    {"name": "Model spread", "purpose": "Показывает согласие моделей."},
                ],
            },
        ],
        "difference": "Top-N задаёт состав, threshold контролирует качество, spread показывает неопределённость ансамбля.",
    },
    "prediction": {
        "title": "Прогноз новых веществ",
        "goal": "Применить готовую модель к новым структурам с оценкой применимости.",
        "required_steps": [
            {"name": "Ввести или загрузить новые SMILES", "purpose": "Создаёт набор объектов для прогноза.", "updates": "prediction input"},
            {"name": "Рассчитать признаки в формате модели", "purpose": "Готовит X для выбранной модели.", "updates": "prediction descriptors"},
        ],
        "groups": [
            {
                "name": "Прогноз и доверие",
                "purpose": "Даёт значение свойства и предупреждения.",
                "tools": [
                    {"name": "Prediction table", "purpose": "Показывает прогноз."},
                    {"name": "Uncertainty", "purpose": "Показывает разброс или интервал."},
                    {"name": "AD for new molecules", "purpose": "Проверяет применимость к новым веществам."},
                ],
            },
        ],
        "difference": "Прогноз даёт число, uncertainty даёт диапазон, AD объясняет применимость.",
    },
    "report": {
        "title": "Отчёт",
        "goal": "Зафиксировать данные, модель, валидацию, ограничения и выводы.",
        "groups": [
            {
                "name": "Документирование результата",
                "purpose": "Собирает доказательства качества модели.",
                "tools": [
                    {"name": "Methodology", "purpose": "Описывает процесс построения."},
                    {"name": "Tables and plots", "purpose": "Фиксируют результаты."},
                    {"name": "Conclusions", "purpose": "Формулируют ограничения."},
                ],
            },
        ],
        "difference": "Методология описывает процесс, таблицы доказывают результат, выводы фиксируют применимость.",
    },
    "final_statistics": {
        "title": "Итоговая статистика",
        "goal": "Собрать ключевые результаты проекта в одну сводку.",
        "groups": [
            {
                "name": "Сводка проекта",
                "purpose": "Показывает состояние данных и модели.",
                "tools": [
                    {"name": "Dataset metrics", "purpose": "Сводит данные."},
                    {"name": "Validation summary", "purpose": "Сводит метрики модели."},
                    {"name": "Reliability signals", "purpose": "Сводит предупреждения."},
                ],
            },
        ],
        "difference": "Dataset metrics описывает данные, validation summary модель, reliability signals ограничения.",
    },
    "save_model": {
        "title": "Сохранение модели",
        "goal": "Сохранить воспроизводимый пакет модели только после ключевых проверок.",
        "groups": [
            {
                "name": "Готовность и пакет",
                "purpose": "Проверяет и сохраняет модель.",
                "tools": [
                    {"name": "Validation checklist", "purpose": "Показывает выполненные проверки."},
                    {"name": "Model package", "purpose": "Сохраняет модель, scaler, дескрипторы и метаданные."},
                ],
            },
        ],
        "difference": "Checklist отвечает за готовность, package за переносимость прогноза.",
    },
}

MODULE_ORDER = [
    "data_preparation",
    "primary_data_analysis",
    "chemical_correctness",
    "chemical_space",
    "structural_audit",
    "feature_sources",
    "training",
    "validation",
    "advanced_validation",
    "diagnostics",
    "error_analysis",
    "model_comparison",
    "consensus",
    "prediction",
    "report",
    "final_statistics",
    "save_model",
]


def get_module_description(module_key):
    return MODULES.get(module_key)


def iter_module_descriptions():
    for key in MODULE_ORDER:
        item = MODULES.get(key)
        if item:
            yield key, item


def module_anchor_id(module_key):
    return f"module-{module_key}"


def tool_anchor_id(module_key, tool_number):
    return f"tool-{module_key}-{tool_number}"


def iter_module_blocks(item):
    blocks = item.get("blocks")
    if blocks is not None:
        return blocks

    converted = []
    steps = item.get("required_steps") or []
    if steps:
        converted.append({
            "name": "Обязательные шаги",
            "kind": "steps",
            "items": [
                {**step, "kind": "step"}
                for step in steps
            ],
        })

    procedures = item.get("applicable_procedures") or []
    if procedures:
        converted.append({
            "name": "Дополнительные процедуры",
            "kind": "procedures",
            "items": [
                {**procedure, "kind": "procedure"}
                for procedure in procedures
            ],
        })

    for group in item.get("groups", []):
        converted.append({
            "name": group["name"],
            "purpose": group.get("purpose", ""),
            "kind": group.get("kind", "tools"),
            "items": [
                {**tool, "kind": tool.get("kind", "tool")}
                for tool in group.get("tools", [])
            ],
        })

    return converted


def block_display_name(block):
    return block.get("title") or block.get("name") or "Блок"


def iter_module_tool_names(item):
    for block in iter_module_blocks(item):
        yield block_display_name(block)
        for item_row in block.get("items", []):
            yield item_row["name"]


def module_overview_markdown():
    """Build a compact clickable module map."""
    lines = ["### Карта модулей", ""]
    for module_key, item in iter_module_descriptions():
        lines.append(f"- [{item['title']}](#{module_anchor_id(module_key)})")
        anchor_number = 1
        for block in iter_module_blocks(item):
            block_anchor = tool_anchor_id(module_key, anchor_number)
            lines.append(f"  - [{block_display_name(block)}](#{block_anchor})")
            anchor_number += 1
            for item_row in block.get("items", []):
                item_anchor = tool_anchor_id(module_key, anchor_number)
                if item_row.get("kind") == "tool":
                    lines.append(f"    - [{item_row['name']}](#{item_anchor})")
                anchor_number += 1
    return "\n".join(lines).strip() + "\n"
