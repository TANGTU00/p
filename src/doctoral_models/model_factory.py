from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier


def make_classifier(algorithm: str, params: dict[str, Any], n_classes: int | None = None):
    algorithm = algorithm.lower()
    if algorithm == "random_forest":
        return RandomForestClassifier(**params)
    if algorithm == "xgboost":
        from xgboost import XGBClassifier

        xgb_params = dict(params)
        if n_classes and n_classes > 2:
            xgb_params.setdefault("objective", "multi:softprob")
            xgb_params.setdefault("num_class", n_classes)
        return XGBClassifier(**xgb_params)
    if algorithm == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**params)
    raise ValueError(f"Unsupported algorithm: {algorithm}")
