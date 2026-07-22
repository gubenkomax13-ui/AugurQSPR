# Current audit findings

Dataset: `2026 Alkany.xlsx`, target `Bp exp`, 268 rows after QC.

- Chemical space: passed; calculation completed and status/plots appeared.
- Canonicalization: passed; normalization completed without the previous Streamlit state error.
- Structural filter: search returned 268/268. The result-page `ValueError` in `show_dataset_change_report` was fixed by clearing Series metadata before `pd.concat`.
- Structural filter application: passed; 268 compounds became the current working dataset without an error.
- SAOD: passed after resetting the filter; run completed and reported 12 suspicious property values, with no runtime error.
- SAOD after applying the structural filter: the module opens and its run button is available.
- SAOD run on the clean route: passed; 268 loaded, 188 analyzed, 80 marked non-checkable, 0 recommended for automatic exclusion.
- SAOD manual-review apply: passed; the working dataset remained 268 rows and no error appeared.
- Model comparison settings: the requested 8 default models are selected correctly; Hold-out and K-Fold are enabled, while LOO base is disabled. However, `LOO only for best models` is checked by default, contradicting the caption that LOO should be enabled after the primary selection and potentially triggering a long run immediately.
- Monte-Carlo, Bootstrap, and Y-randomization are disabled by default and their repeat counts are 10.
- Descriptor source screen: molecular descriptors are selected by default, the expanded mode is selected, and 1775 descriptors are shown; descriptor calculation was previously verified, so it was skipped in this pass.
