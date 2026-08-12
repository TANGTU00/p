from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

FLOAT_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
UNKNOWN_LABEL = "__UNKNOWN__"


@dataclass
class PreparedData:
    x: pd.DataFrame
    y_text: pd.Series
    y_encoded: np.ndarray
    target_encoder: LabelEncoder
    feature_encoders: dict[str, LabelEncoder]
    numeric_medians: dict[str, float]


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def parse_numeric_series(series: pd.Series) -> pd.Series:
    values = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .map(lambda value: FLOAT_PATTERN.search(value).group(0) if FLOAT_PATTERN.search(value) else np.nan)
    )
    return pd.to_numeric(values, errors="coerce")


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def fit_label_encoder(values: pd.Series) -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.fit(values.astype(str).map(clean_text))
    return encoder


def transform_labels_with_unknown(encoder: LabelEncoder, values: pd.Series) -> np.ndarray:
    known = set(str(x) for x in encoder.classes_)
    encoded_values = []
    for value in values.astype(str).map(clean_text):
        encoded_values.append(value if value in known else UNKNOWN_LABEL)
    if UNKNOWN_LABEL in encoded_values and UNKNOWN_LABEL not in known:
        encoder.classes_ = np.append(encoder.classes_, UNKNOWN_LABEL)
    return encoder.transform(encoded_values)


def prepare_single_target_frame(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    categorical_features: list[str] | None = None,
) -> PreparedData:
    categorical = set(categorical_features or [])
    require_columns(df, [*features, target])

    work = df.copy()
    work[target] = work[target].map(clean_text)
    work = work.loc[work[target] != ""].copy()
    if work.empty:
        raise ValueError(f"No labeled rows found for target {target!r}")

    feature_encoders: dict[str, LabelEncoder] = {}
    numeric_medians: dict[str, float] = {}

    for column in features:
        if column in categorical:
            encoder = fit_label_encoder(work[column])
            work[column] = encoder.transform(work[column].astype(str).map(clean_text))
            feature_encoders[column] = encoder
        else:
            numeric = parse_numeric_series(work[column])
            median = float(numeric.median()) if not numeric.dropna().empty else 0.0
            work[column] = numeric.fillna(median)
            numeric_medians[column] = median

    target_encoder = fit_label_encoder(work[target])
    y_text = work[target].astype(str)
    y_encoded = target_encoder.transform(y_text)
    return PreparedData(
        x=work[features],
        y_text=y_text,
        y_encoded=y_encoded,
        target_encoder=target_encoder,
        feature_encoders=feature_encoders,
        numeric_medians=numeric_medians,
    )
