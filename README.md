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

## Run Locally

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

Admin mode is enabled only through Streamlit secrets. In Streamlit Community Cloud, open the app settings and add:

```toml
ADMIN_PASSWORD = "your-private-password"
```

Do not commit `.streamlit/secrets.toml` to GitHub.

The public repository should not include private datasets, generated reports, local caches, or private model packages. These are excluded by `.gitignore`.

## Security Note

Only upload `.pkl` or `.joblib` models from trusted sources. Pickle/joblib files can execute Python code when loaded.
