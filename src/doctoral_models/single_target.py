from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split

from doctoral_models.metrics import save_classification_outputs
from doctoral_models.model_factory import make_classifier
from doctoral_models.preprocessing import prepare_single_target_frame


def can_stratify(y: np.ndarray) -> bool:
    counts = Counter(y)
    return len(counts) > 1 and all(count >= 2 for count in counts.values())


def safe_n_splits(y: np.ndarray, requested: int = 5) -> int:
    counts = np.bincount(y)
    if len(counts) < 2:
        return 0
    return max(2, min(requested, int(counts.min())))


def weighted_class_prediction(probabilities: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    counts = Counter(y_train)
    total = sum(counts.values())
    class_weights = np.ones(probabilities.shape[1], dtype=float)
    for class_id, count in counts.items():
        class_weights[int(class_id)] = total / (len(counts) * count)
    weighted = probabilities * class_weights.reshape(1, -1)
    row_sum = weighted.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return np.argmax(weighted / row_sum, axis=1)


def train_single_target(config: dict[str, Any], data_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    data = pd.read_csv(data_path, keep_default_na=False)
    name = config["name"]
    output = Path(output_dir or Path("outputs") / name)
    output.mkdir(parents=True, exist_ok=True)

    prepared = prepare_single_target_frame(
        data,
        features=list(config["features"]),
        target=str(config["target"]),
        categorical_features=list(config.get("categorical_features", [])),
    )

    stratify = prepared.y_encoded if can_stratify(prepared.y_encoded) else None
    x_train, x_test, y_train, y_test = train_test_split(
        prepared.x,
        prepared.y_encoded,
        test_size=float(config.get("test_size", 0.2)),
        random_state=int(config.get("random_state", 42)),
        stratify=stratify,
    )

    model = make_classifier(
        str(config["algorithm"]),
        dict(config.get("model_params", {})),
        n_classes=len(prepared.target_encoder.classes_),
    )
    model.fit(x_train, y_train)

    if config.get("class_weighted_prediction") and hasattr(model, "predict_proba"):
        y_pred_encoded = weighted_class_prediction(model.predict_proba(x_test), y_train)
    else:
        y_pred_encoded = model.predict(x_test)

    y_test_text = prepared.target_encoder.inverse_transform(y_test)
    y_pred_text = prepared.target_encoder.inverse_transform(y_pred_encoded.astype(int))
    labels = list(prepared.target_encoder.classes_)
    save_classification_outputs(output, y_test_text, y_pred_text, labels)

    cv_summary = cross_validate_single_target(config, prepared)
    if cv_summary:
        pd.DataFrame(cv_summary).to_csv(output / "cross_validation.csv", index=False)

    artifact = {
        "model": model,
        "target_encoder": prepared.target_encoder,
        "feature_encoders": prepared.feature_encoders,
        "numeric_medians": prepared.numeric_medians,
        "features": list(config["features"]),
        "categorical_features": list(config.get("categorical_features", [])),
        "target": str(config["target"]),
    }
    joblib.dump(artifact, output / "model.joblib")

    run_config = {
        "config": config,
        "data_path": str(data_path),
        "n_rows": int(len(prepared.x)),
        "classes": labels,
        "accuracy": float(accuracy_score(y_test_text, y_pred_text)),
    }
    (output / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(classification_report(y_test_text, y_pred_text, labels=labels, zero_division=0))
    print(f"Saved outputs to {output}")
    return run_config


def cross_validate_single_target(config: dict[str, Any], prepared) -> list[dict[str, Any]]:
    n_splits = safe_n_splits(prepared.y_encoded, requested=int(config.get("cv_folds", 5)))
    if n_splits == 0:
        return []

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(config.get("random_state", 42)),
    )
    rows: list[dict[str, Any]] = []
    for fold, (train_idx, validation_idx) in enumerate(splitter.split(prepared.x, prepared.y_encoded), start=1):
        model = make_classifier(
            str(config["algorithm"]),
            dict(config.get("model_params", {})),
            n_classes=len(prepared.target_encoder.classes_),
        )
        x_train = prepared.x.iloc[train_idx]
        x_validation = prepared.x.iloc[validation_idx]
        y_train = prepared.y_encoded[train_idx]
        y_validation = prepared.y_encoded[validation_idx]
        model.fit(x_train, y_train)
        y_pred = model.predict(x_validation)
        y_true_text = prepared.target_encoder.inverse_transform(y_validation)
        y_pred_text = prepared.target_encoder.inverse_transform(y_pred.astype(int))
        rows.append(
            {
                "fold": fold,
                "accuracy": float(accuracy_score(y_true_text, y_pred_text)),
            }
        )
    return rows
