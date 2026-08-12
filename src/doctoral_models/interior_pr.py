from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

from doctoral_models.preprocessing import clean_text, parse_numeric_series, require_columns


def make_rf(params: dict[str, Any]) -> RandomForestClassifier:
    return RandomForestClassifier(**params)


def choose_project_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def task_labels_from_row(row: pd.Series, slots: list[str], no_label: str) -> list[str]:
    labels = [clean_text(row.get(slot, "")) for slot in slots]
    labels = [label for label in labels if label]
    return list(dict.fromkeys(labels)) or [no_label]


def fit_onehot(frame: pd.DataFrame) -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(frame.astype(str))
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False).fit(frame.astype(str))


def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_samples": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "jaccard_samples": float(jaccard_score(y_true, y_pred, average="samples", zero_division=0)),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }


def prepare_interior_frame(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    feature_columns = list(config["feature_columns"])
    stage1_targets = list(config["stage1_targets"])
    task_config = dict(config["stage2_tasks"])
    slot_columns = [slot for task in task_config.values() for slot in task["slots"]]
    require_columns(df, [*feature_columns, *stage1_targets, *slot_columns])

    work = df.copy()
    project_column = choose_project_column(work, list(config.get("project_column_candidates", [])))
    if project_column and project_column != "Project":
        work = work.rename(columns={project_column: "Project"})
    elif not project_column:
        work["Project"] = "Project_1"

    for column in feature_columns:
        work[column] = parse_numeric_series(work[column])
    for target in stage1_targets:
        work[target] = work[target].map(clean_text)
        work[target] = work[target].replace("", f"No_{target}")
    for task_name, task in task_config.items():
        work[task_name] = work.apply(
            lambda row, slots=task["slots"], no_label=task["no_label"]: task_labels_from_row(row, slots, no_label),
            axis=1,
        )
    return work


def split_frame(df: pd.DataFrame, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(df))
    stratify = df["SL"] if "SL" in df.columns and df["SL"].value_counts().min() >= 2 else None
    train_idx, holdout_idx = train_test_split(
        idx,
        test_size=float(config.get("test_size", 0.15)) + float(config.get("validation_size", 0.15)),
        random_state=int(config.get("random_state", 42)),
        stratify=stratify,
    )
    holdout = df.iloc[holdout_idx]
    holdout_stratify = holdout["SL"] if "SL" in holdout.columns and holdout["SL"].value_counts().min() >= 2 else None
    validation_relative, test_relative = train_test_split(
        np.arange(len(holdout_idx)),
        test_size=0.5,
        random_state=int(config.get("random_state", 42)),
        stratify=holdout_stratify,
    )
    return train_idx, holdout_idx[validation_relative], holdout_idx[test_relative]


def add_stage1_features(
    base_x: np.ndarray,
    stage1_frame: pd.DataFrame,
    encoder: OneHotEncoder,
) -> np.ndarray:
    semantic_x = encoder.transform(stage1_frame.astype(str))
    return np.hstack([base_x, semantic_x])


def threshold_multilabel(probabilities: list[np.ndarray], threshold: float) -> np.ndarray:
    columns = []
    for proba in probabilities:
        if proba.shape[1] == 1:
            columns.append(np.ones(proba.shape[0], dtype=int))
        else:
            columns.append((proba[:, 1] >= threshold).astype(int))
    pred = np.vstack(columns).T
    empty_rows = pred.sum(axis=1) == 0
    if empty_rows.any():
        stacked = np.vstack([p[:, -1] if p.shape[1] > 1 else np.ones(p.shape[0]) for p in probabilities]).T
        pred[empty_rows, np.argmax(stacked[empty_rows], axis=1)] = 1
    return pred


def train_interior_pr(config: dict[str, Any], data_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    raw = pd.read_csv(data_path, keep_default_na=False)
    df = prepare_interior_frame(raw, config)
    output = Path(output_dir or Path("outputs") / str(config["name"]))
    output.mkdir(parents=True, exist_ok=True)

    train_idx, validation_idx, test_idx = split_frame(df, config)
    feature_columns = list(config["feature_columns"])
    raw_x = df[feature_columns]
    imputer = KNNImputer(n_neighbors=5)
    x_train = imputer.fit_transform(raw_x.iloc[train_idx])
    x_validation = imputer.transform(raw_x.iloc[validation_idx])
    x_test = imputer.transform(raw_x.iloc[test_idx])

    stage1_models: dict[str, RandomForestClassifier] = {}
    stage1_predictions: dict[str, dict[str, np.ndarray]] = {}
    stage1_rows: list[dict[str, Any]] = []
    for target in config["stage1_targets"]:
        model = make_rf(dict(config["stage1_model_params"]))
        y_train = df.iloc[train_idx][target].astype(str)
        model.fit(x_train, y_train)
        stage1_models[target] = model
        stage1_predictions[target] = {
            "train": model.predict(x_train),
            "validation": model.predict(x_validation),
            "test": model.predict(x_test),
        }
        for split_name, split_idx, x_split in [
            ("train", train_idx, x_train),
            ("validation", validation_idx, x_validation),
            ("test", test_idx, x_test),
        ]:
            y_true = df.iloc[split_idx][target].astype(str)
            y_pred = model.predict(x_split)
            stage1_rows.append(
                {
                    "split": split_name,
                    "target": target,
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                }
            )

    train_stage1_frame = pd.DataFrame({target: stage1_predictions[target]["train"] for target in config["stage1_targets"]})
    validation_stage1_frame = pd.DataFrame({target: stage1_predictions[target]["validation"] for target in config["stage1_targets"]})
    test_stage1_frame = pd.DataFrame({target: stage1_predictions[target]["test"] for target in config["stage1_targets"]})
    semantic_encoder = fit_onehot(train_stage1_frame)

    x2_train = add_stage1_features(x_train, train_stage1_frame, semantic_encoder)
    x2_validation = add_stage1_features(x_validation, validation_stage1_frame, semantic_encoder)
    x2_test = add_stage1_features(x_test, test_stage1_frame, semantic_encoder)

    stage2_models: dict[str, Any] = {}
    multilabel_binarizers: dict[str, MultiLabelBinarizer] = {}
    stage2_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    threshold = float(config.get("threshold", 0.25))

    for task_name in config["stage2_tasks"]:
        mlb = MultiLabelBinarizer()
        y_train = mlb.fit_transform(df.iloc[train_idx][task_name])
        model = MultiOutputClassifier(make_rf(dict(config["stage2_model_params"])))
        model.fit(x2_train, y_train)
        stage2_models[task_name] = model
        multilabel_binarizers[task_name] = mlb

        for split_name, split_idx, x_split in [
            ("validation", validation_idx, x2_validation),
            ("test", test_idx, x2_test),
        ]:
            y_true = mlb.transform(df.iloc[split_idx][task_name])
            y_pred = threshold_multilabel(model.predict_proba(x_split), threshold)
            stage2_rows.append({"split": split_name, "task": task_name, **multilabel_metrics(y_true, y_pred)})
            true_labels = ["|".join(labels) for labels in df.iloc[split_idx][task_name]]
            pred_labels = ["|".join(labels) for labels in mlb.inverse_transform(y_pred)]
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "split": split_name,
                        "task": task_name,
                        "row_id": df.iloc[split_idx].index,
                        "y_true": true_labels,
                        "y_pred": pred_labels,
                    }
                )
            )

    pd.DataFrame(stage1_rows).to_csv(output / "stage1_metrics.csv", index=False)
    pd.DataFrame(stage2_rows).to_csv(output / "stage2_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output / "stage2_predictions.csv", index=False)

    artifact = {
        "imputer": imputer,
        "semantic_encoder": semantic_encoder,
        "stage1_models": stage1_models,
        "stage2_models": stage2_models,
        "multilabel_binarizers": multilabel_binarizers,
        "feature_columns": feature_columns,
        "config": config,
    }
    joblib.dump(artifact, output / "model.joblib")

    run_config = {
        "config": config,
        "data_path": str(data_path),
        "n_rows": int(len(df)),
        "split_counts": {
            "train": int(len(train_idx)),
            "validation": int(len(validation_idx)),
            "test": int(len(test_idx)),
        },
    }
    (output / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved outputs to {output}")
    return run_config
