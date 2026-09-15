import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.model_artifacts import validate_sklearn_feature_schema


class LegacyModel:
    pass


class NamedModel:
    feature_names_in_ = np.array(["Open", "RSI_14", "Sentiment"], dtype=object)


def test_validate_feature_schema_accepts_exact_name_and_order_match():
    validate_sklearn_feature_schema(
        NamedModel(), ["Open", "RSI_14", "Sentiment"]
    )


def test_validate_feature_schema_rejects_reordered_columns():
    with pytest.raises(ValueError, match="does not match"):
        validate_sklearn_feature_schema(
            NamedModel(), ["RSI_14", "Open", "Sentiment"]
        )


def test_validate_feature_schema_rejects_legacy_anonymous_artifact():
    with pytest.raises(ValueError, match="does not contain a feature schema"):
        validate_sklearn_feature_schema(LegacyModel(), ["Open", "RSI_14"])


def test_validate_feature_schema_rejects_empty_current_schema():
    with pytest.raises(ValueError, match="must not be empty"):
        validate_sklearn_feature_schema(NamedModel(), [])


def test_random_forest_fitted_on_dataframe_persists_feature_names():
    X = pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0, 13.0],
            "RSI_14": [40.0, 45.0, 50.0, 55.0],
        }
    )
    y = np.array([10.5, 11.5, 12.5, 13.5])

    model = RandomForestRegressor(n_estimators=2, random_state=0).fit(X, y)

    assert model.feature_names_in_.tolist() == ["Open", "RSI_14"]
    validate_sklearn_feature_schema(model, X.columns)
