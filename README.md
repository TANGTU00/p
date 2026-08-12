# Doctoral Building Element Models

博士期间建筑构件语义与性能推定模型的整理版项目。

This repository collects the training pipelines used for building element semantic inference and performance prediction. The original one-off scripts have been reorganized into reusable command-line modules. Plotting, LIME demos, desktop-only paths, and ad hoc validation blocks were removed from the main training flow.

## Models

| Model | Target(s) | Algorithm | Config |
| --- | --- | --- | --- |
| EF inference | `Ef_3` | Random Forest classifier | `configs/ef.yaml` |
| SS inference | `Ss` | Random Forest classifier | `configs/ss.yaml` |
| Wall/slab fire inference | `FireRating`, `Combustible`, `Compartmentation` | XGBoost classifier | `configs/fire_rating.yaml`, `configs/combustible.yaml`, `configs/compartmentation.yaml` |
| Independent element PR prediction | configurable PR target | LightGBM classifier | `configs/independent_pr.yaml` |
| Interior PR prediction | ceiling/floor/skirting/wall-covering PR labels | two-stage Random Forest multilabel pipeline | `configs/interior_pr.yaml` |

## Project Layout

```text
doctoral-building-element-models/
  configs/                 # model-specific training configuration
  src/doctoral_models/      # reusable package
  tests/                    # lightweight regression tests
  pyproject.toml
  requirements.txt
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Place datasets under `data/` or pass an absolute/relative CSV path through `--data`.

The project intentionally does not commit raw thesis datasets. The expected columns are documented in each config file.

## Usage

Run EF or SS inference:

```bash
python -m doctoral_models.cli train --config configs/ef.yaml --data data/KNN_S.csv
python -m doctoral_models.cli train --config configs/ss.yaml --data data/KNN_S.csv
```

Run wall/slab fire-related inference:

```bash
python -m doctoral_models.cli train --config configs/fire_rating.yaml --data data/Dataset.csv
python -m doctoral_models.cli train --config configs/combustible.yaml --data data/Dataset.csv
python -m doctoral_models.cli train --config configs/compartmentation.yaml --data data/Dataset.csv
```

Run independent PR prediction:

```bash
python -m doctoral_models.cli train --config configs/independent_pr.yaml --data data/independent_pr.csv
```

Run interior PR prediction:

```bash
python -m doctoral_models.cli train-interior-pr --config configs/interior_pr.yaml --data data/sl-pr.csv
```

All commands write metrics and model artifacts to `outputs/<model-name>/` unless overridden.

## Notes

- Categorical feature columns are label-encoded with unknown-value handling.
- Numeric columns are parsed robustly, including values with commas or units.
- Train/test splitting uses stratification when class counts allow it.
- The independent PR model uses the requested LightGBM parameters:
  `n_estimators=200`, `learning_rate=0.05`, `num_leaves=31`, `subsample=0.9`, `colsample_bytree=0.9`, `reg_lambda=1.0`.
