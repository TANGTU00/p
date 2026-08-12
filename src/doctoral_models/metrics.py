from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def classification_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def save_classification_outputs(
    output_dir: Path,
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    labels: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([classification_metrics(y_true, y_pred)]).to_csv(output_dir / "metrics.csv", index=False)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(output_dir / "confusion_matrix.csv")
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(output_dir / "predictions.csv", index=False)
