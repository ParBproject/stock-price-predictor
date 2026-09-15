"""Helpers for validating serialized model artifacts before inference."""

from collections.abc import Iterable


def validate_sklearn_feature_schema(model, feature_names: Iterable[str]) -> None:
    """Require an exact feature-name/order match for a fitted sklearn model.

    Models trained on pandas DataFrames expose ``feature_names_in_``. Legacy
    artifacts trained on anonymous NumPy arrays do not, so they cannot be
    safely reused after a feature-schema change and must be retrained.
    """
    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        raise ValueError(
            "Loaded model does not contain a feature schema; retrain it with the "
            "current pipeline before backtesting"
        )

    expected_names = list(expected)
    actual_names = list(feature_names)

    if not actual_names:
        raise ValueError("Current feature schema must not be empty")
    if expected_names != actual_names:
        raise ValueError(
            "Loaded model feature schema does not match the current pipeline. "
            f"Expected {expected_names}, got {actual_names}"
        )
