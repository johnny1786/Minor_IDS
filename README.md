# Minor_IDS

Ensemble Gated Image–Tabular Fusion for IoT Intrusion Detection

This repository contains an implementation and experimental outputs for a unified IDS pipeline that transforms tabular network telemetry into image-like representations, trains a gated image–tabular deep branch, trains classical tabular learners, and combines their outputs with a weighted late-fusion ensemble.

Key datasets used: UNSW-NB15, BoT-IoT, TON-IoT

Authors: Ganteda Sai Sagar, Kuna Remanth Kumar, Madas Johnson
Advisor: Dr. Ditipriya Sinha
Report date: May 13, 2025

## Highlights
- Implements a disciplined train-holdout-before-oversampling workflow (train-only WCGAN-GP).
- Tabular-to-image conversion (PCA + LDA + padding to square) for CNN learning.
- Gated fusion branch that adaptively combines tabular and image representations.
- Weighted late-fusion ensemble: 40% XGBoost, 40% LightGBM, 20% Gated Fusion.
- Reproducible outputs saved per dataset (trained models, configs, metrics, plots).

## Key Results (from saved outputs)
- BoT-IoT: 99.03% accuracy (Random Forest and late fusion ensemble)
- TON-IoT: 95.96% accuracy (late fusion ensemble)
- UNSW-NB15: 60.04% accuracy (late fusion ensemble)

These numbers are the best-performing results produced by the project's saved runs and are included in the `outputs_*` folders.

## Repository Structure
- `bot_iot_gated_fusion.py` — experiment pipeline for BoT-IoT
- `ton_iot_gated_fusion.py` — experiment pipeline for TON-IoT
- `unsw_gated_fusion.py` — experiment pipeline for UNSW-NB15
- `outputs_bot_iot_gated_fusion/`, `outputs_ton_gated_fusion/`, `outputs_unsw_gated_fusion/` — run artifacts (models, `pipeline_config.json`, `results_summary.csv`, `results.json`, plots)

Typical artifacts inside each `outputs_*` folder:
- `gated_fusion_best.keras` — best Keras model (gated fusion branch)
- `gated_image_tabular_fusion.keras` — saved gated fusion model snapshot
- `pipeline_config.json` — configuration used for the run
- `results_summary.csv` and `results.json` — evaluation metrics and summaries
- Figures: PCA/LDA plots, confusion matrices, performance visualizations

## Method Overview
1. Load dataset CSV and apply label/feature selection.
2. Cap numeric outliers (IQR-based percentile clipping).
3. Quantile normalization (QuantileTransformer -> normal output).
4. PCA for variance-preserving representation and deduplication.
5. LDA to increase class separability.
6. Convert combined tabular + PCA + LDA vector into a square grayscale image (pad to nearest square).
7. Train-only WCGAN-GP to oversample minority classes on the training split.
8. Train classical tabular models (e.g., Random Forest, XGBoost, LightGBM) and the gated fusion deep branch.
9. Combine predictions via weighted late fusion (0.40 XGBoost, 0.40 LightGBM, 0.20 Gated Fusion).

## Requirements
- Python 3.8+ recommended
- Typical packages: `tensorflow`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`, `lightgbm`.

Install a minimal environment (example):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If a `requirements.txt` is not present, install the likely packages:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn xgboost lightgbm
```

GPU builds of `tensorflow` are recommended for faster training on large datasets.

## Quick Start
1. Prepare dataset CSVs and update dataset paths inside the corresponding script (`bot_iot_gated_fusion.py`, `ton_iot_gated_fusion.py`, or `unsw_gated_fusion.py`).
2. Run a dataset pipeline (example):

```bash
python3 bot_iot_gated_fusion.py
```

3. After the run, inspect the generated folder `outputs_bot_iot_gated_fusion/` for `pipeline_config.json`, trained models, and `results_summary.csv`.

## Reproducing Results
- Ensure the same `pipeline_config.json` is used for the experiment. The `outputs_*` folders contain the config used for each saved run.
- Confirm that train/test holdout is created before any oversampling step (this is critical to avoid leakage).
- Use the saved models in `*.keras` for evaluation or further analysis.

## Notes on Evaluations
- The project reports multiple metrics (accuracy, precision, recall, F1, false alarm rate). Check `results_summary.csv` and `results.json` in each `outputs_*` folder for the full breakdown.
- Visual diagnostics (PCA variance, LDA projections, confusion matrices) are saved as figures alongside the results.

## Extending the Project
- To add a new dataset: copy an existing script (for example `bot_iot_gated_fusion.py`), update dataset-specific loading and feature selection, and run the pipeline.
- Consider adding a `requirements.txt`, a lightweight example dataset, and a small CI workflow to validate the scripts automatically.

## License
Specify your preferred license (e.g., MIT). Update this section with the chosen license.

## Contact
Add author/maintainer contact information here.

---

If you want, I can now:
- generate a `requirements.txt` from the environment,
- add a short `run_example.sh` script that runs one dataset with a sample config, or
- extract and incorporate specific figures/tables from `/Users/johnson/Downloads/report.pdf` into a `docs/` folder.
Tell me which option you prefer.