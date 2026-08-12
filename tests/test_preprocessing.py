import pandas as pd

from doctoral_models.preprocessing import parse_numeric_series, prepare_single_target_frame


def test_parse_numeric_series_handles_units_and_commas():
    values = pd.Series(["1,234.5 mm", "abc", "", "7"])
    parsed = parse_numeric_series(values)
    assert parsed.iloc[0] == 1234.5
    assert pd.isna(parsed.iloc[1])
    assert pd.isna(parsed.iloc[2])
    assert parsed.iloc[3] == 7


def test_prepare_single_target_frame_encodes_categorical_and_drops_empty_targets():
    df = pd.DataFrame(
        {
            "IfcEntity": ["Wall", "Slab", "Wall"],
            "Area": ["10 m2", "20", "30"],
            "Target": ["A", "", "B"],
        }
    )
    prepared = prepare_single_target_frame(
        df,
        features=["IfcEntity", "Area"],
        target="Target",
        categorical_features=["IfcEntity"],
    )
    assert len(prepared.x) == 2
    assert set(prepared.target_encoder.classes_) == {"A", "B"}
    assert "IfcEntity" in prepared.feature_encoders
