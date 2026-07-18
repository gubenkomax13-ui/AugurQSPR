# Augur QSPR

Augur QSPR is a Streamlit application for QSPR/QSAR modelling, molecular descriptor calculation, model validation, applicability domain analysis, prediction for new compounds, uncertainty diagnostics, spectral descriptors, and multilingual UI support.

## Online Deployment

The app is prepared for deployment through GitHub and Streamlit Community Cloud.

Recommended Streamlit settings:

```text
Repository: USERNAME/augur-qspr
Branch: main
Main file path: qspr_app.py
```

## Полная версия и локальный запуск

Онлайн-версия Augur QSPR на Streamlit Cloud предназначена для ознакомления с основными возможностями программы. Полная функциональность, включая работу с локальными базами спектров, расширенные расчёты дескрипторов, длительные процедуры валидации и служебные инструменты, доступна при локальном запуске приложения на собственном компьютере.

Репозиторий полной версии: [GitHub](https://github.com/gubenkomax13-ui/AugurQSPR).

### 1. Скачать проект

```bash
git clone https://github.com/gubenkomax13-ui/AugurQSPR.git
cd AugurQSPR
```

Также можно нажать `Code` -> `Download ZIP` на странице репозитория, распаковать архив и открыть папку проекта.

### 2. Создать виртуальное окружение

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Установить зависимости

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Запустить приложение

```bash
streamlit run qspr_app.py
```

После запуска откройте:

```text
http://localhost:8501
```

## Системные требования

- Рекомендуется Python 3.11.
- Минимум 8 GB RAM; для больших наборов дескрипторов желательно 16 GB RAM.
- Java Runtime Environment требуется для PaDEL-дескрипторов. В Streamlit Cloud он задаётся через `packages.txt` как `default-jre`.
- Интернет требуется только для загрузки внешних ресурсов, Google Drive баз или онлайн-поиска спектров.

## Возможности локальной версии

Локальный запуск позволяет использовать:

- QSPR/QSAR-моделирование;
- расчёт RDKit, Mordred и PaDEL-дескрипторов;
- обучение и сравнение моделей;
- Hold-out, K-Fold, LOO, Repeated Hold-out, Bootstrap и Y-randomization;
- Applicability Domain;
- диагностику ошибок модели;
- анализ важности дескрипторов;
- прогноз свойств новых веществ;
- работу с локальными спектральными базами;
- расчёт спектральных дескрипторов;
- расширенные отчёты и экспорт результатов.

## Ограничения Streamlit Cloud версии

Облачная версия может иметь ограничения:

- нет прямого доступа к диску пользователя;
- большие спектральные базы должны подключаться отдельно;
- длительные расчёты могут быть ограничены ресурсами Streamlit Cloud;
- часть служебных инструментов доступна только администратору;
- локальные модели и базы данных не входят в демонстрационную поставку.

## Зависимости

Основные зависимости устанавливаются из `requirements.txt`: Streamlit, NumPy, pandas, SciPy, scikit-learn, matplotlib, seaborn, RDKit, joblib, openpyxl, xlsxwriter, SHAP, streamlit-ketcher и пакеты для работы со спектральными ресурсами.

Mordred и PaDEL используются для расширенных наборов молекулярных дескрипторов. PaDEL требует Java Runtime Environment; для базового запуска достаточно RDKit-дескрипторов. Если установка Mordred/PaDEL нестабильна в облачной среде, используйте локальный запуск или режим RDKit-дескрипторов.

Опциональные возможности используют дополнительные пакеты:

- `gdown` - загрузка ресурсов из Google Drive;
- `python-docx` и `reportlab` - подготовка расширенных отчётов;
- `plotly` - интерактивные графики, если они включены в соответствующих блоках;
- `morfeus-ml` - 3D/morfeus-дескрипторы;
- `dscribe` - атомистические дескрипторы;
- `xgboost`, `lightgbm`, `catboost` - дополнительные алгоритмы машинного обучения;
- `jcamp` - разбор спектральных файлов JCAMP-DX.

## Run Locally

Recommended full Windows setup:

```bat
setup_full.bat
run_augur.bat
```

PowerShell alternative:

```powershell
.\setup_full.ps1
.\run_augur.ps1
```

The full setup creates `.venv`, installs `requirements-full.txt`, prepares
PySR/Juliacall on first import, and prints the number of model candidates seen
by Augur. A complete local installation should report 25 available model
candidates.

Manual minimal setup:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run qspr_app.py
```

On Linux/macOS, activate the environment with:

```bash
source .venv/bin/activate
```

## Project Files

Core application files:

```text
qspr_app.py
modules/
locales/
help/
model_encyclopedia.json
descriptor_lists.json
descriptor_meanings.json
padel_unique_descriptors.txt
requirements.txt
packages.txt
```

Important modules:

```text
modules/prognostic_model_core.py
modules/prediction_uncertainty.py
modules/saod2_core.py
modules/model_catalog.py
modules/error_analysis_core.py
modules/error_analysis_ui.py
modules/descriptor_importance_core.py
```

Language files:

```text
locales/ru.json
locales/en.json
locales/kk.json
```

## Streamlit Cloud Notes

Python dependencies are installed from `requirements.txt`. System packages are installed from `packages.txt`; Java is included for PaDEL support through `default-jre`.

The application requires NumPy 1.x (`numpy<2`). If Streamlit Cloud previously installed NumPy 2.x, push the updated `requirements.txt` and reboot or redeploy the app from Streamlit Cloud.

The prepared Augur spectral bank is published as an online Google Drive directory:
[Augur spectral bank](https://drive.google.com/drive/folders/1OsxFY_Rs2K55tPVqoo1QhyB0hxwZPoKd?usp=drive_link).

For Streamlit Cloud the spectral bank can be used lazily: the app downloads only
`spectra_index.csv` first, then downloads only the matched processed spectra.
The Google Drive folder should keep the same structure as the local bank:

```text
spectra_bank/
  spectra_index.csv
  spectra_manifest.csv
  IR/processed/...
  Mass/processed/...
```

`spectra_manifest.csv` maps the same relative paths to Google Drive file IDs:

```text
path,file_id
IR/processed/IR_NIST_gas_XXXX_001_processed.csv,GOOGLE_DRIVE_FILE_ID
Mass/processed/Mass_MASSBANK_gas_XXXX_001_processed.csv,GOOGLE_DRIVE_FILE_ID
```

In Streamlit secrets add either direct URLs or file IDs for the small service files:

```toml
AUGUR_SPECTRA_INDEX_FILE_ID = "..."
AUGUR_SPECTRA_MANIFEST_FILE_ID = "..."
AUGUR_SPECTRA_SEARCH_CACHE_FILE_ID = "..." # optional
```

Alternatively, the app can build `spectra_manifest.csv` automatically by walking
the public Google Drive folder metadata. This does not download all spectra; it
only reads file names and file IDs:

```toml
AUGUR_SPECTRA_INDEX_FILE_ID = "..."
AUGUR_SPECTRA_BANK_FOLDER_ID = "1OsxFY_Rs2K55tPVqoo1QhyB0hxwZPoKd"
AUGUR_GOOGLE_DRIVE_API_KEY = "..."
```

The individual spectra are not listed in secrets; they are resolved through
`spectra_manifest.csv` and downloaded on demand. `spectra_search_cache.csv`
is optional and stores service search history/statuses; it does not replace
the manifest.

Admin mode is enabled only through Streamlit secrets. In Streamlit Community Cloud, open the app settings and add:

```toml
ADMIN_PASSWORD = "your-private-password"
```

The online demo notice is shown automatically on Streamlit Cloud hosts. It opens on page load, collapses once after 10 seconds, and then works as a manual expand/collapse block until the page is reloaded. To force this behavior, add:

```toml
AUGUR_SHOW_ONLINE_DEMO_NOTICE = "true"
```

For a local run, leave this secret unset or set it to `"false"`.

Do not commit `.streamlit/secrets.toml` to GitHub.

The public repository should not include private datasets, generated reports, local caches, or private model packages. These are excluded by `.gitignore`.

## Security Note

Only upload `.pkl` or `.joblib` models from trusted sources. Pickle/joblib files can execute Python code when loaded.

## Безопасность

Загружайте `.pkl` и `.joblib` модели только из доверенных источников. Эти форматы используют pickle-механизм Python и потенциально могут выполнять код при загрузке.
