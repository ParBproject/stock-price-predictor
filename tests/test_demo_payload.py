import numpy as np
import pytest

from scripts.build_demo import dumps_payload, mae_ratio, persistence_forecast


def test_persistence_forecast_is_the_close_already_known():
    current_close = np.array([101.25, 99.5, 110.0])

    forecast = persistence_forecast(current_close)

    assert forecast is not current_close
    assert np.array_equal(forecast, current_close)


@pytest.mark.parametrize(
    "values, match",
    [
        (np.array([[1.0, 2.0]]), "one-dimensional"),
        (np.array([]), "must not be empty"),
        (np.array([1.0, np.nan]), "finite"),
        (np.array([1.0, np.inf]), "finite"),
    ],
)
def test_persistence_forecast_rejects_invalid_closes(values, match):
    with pytest.raises(ValueError, match=match):
        persistence_forecast(values)


def test_mae_ratio_uses_the_two_errors():
    assert mae_ratio(16.0, 2.0) == pytest.approx(8.0)


def test_mae_ratio_rejects_a_non_positive_baseline():
    with pytest.raises(ValueError, match="positive"):
        mae_ratio(1.0, 0.0)


def test_dumps_payload_rejects_an_oversized_document():
    payload = {"blob": "x" * 500_001}

    with pytest.raises(RuntimeError, match="500000"):
        dumps_payload(payload)


def test_dumps_payload_is_compact_and_finite():
    text = dumps_payload({"mae": 1.5, "ok": True})

    assert "\n" not in text
    assert text == '{"mae":1.5,"ok":true}'
