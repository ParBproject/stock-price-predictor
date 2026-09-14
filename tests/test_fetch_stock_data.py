import numpy as np
import pandas as pd

import src.data_loader as data_loader


def _raw_stock_data() -> pd.DataFrame:
    index = pd.date_range("2025-01-02", periods=80, freq="B")
    close = np.linspace(100.0, 120.0, len(index))
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.arange(1_000, 1_000 + len(index), dtype=float),
        },
        index=index,
    )


def _stub_external_data(monkeypatch):
    monkeypatch.setattr(
        data_loader.yf,
        "download",
        lambda *args, **kwargs: _raw_stock_data(),
    )
    monkeypatch.setattr(data_loader, "adf_test", lambda *args, **kwargs: {})


def test_fetch_stock_data_requests_adjusted_prices_explicitly(monkeypatch):
    captured = {}

    def fake_download(*args, **kwargs):
        captured.update(kwargs)
        return _raw_stock_data()

    monkeypatch.setattr(data_loader.yf, "download", fake_download)
    monkeypatch.setattr(data_loader, "adf_test", lambda *args, **kwargs: {})

    data_loader.fetch_stock_data()

    assert captured["auto_adjust"] is True


def test_fetch_stock_data_saves_basename_in_current_directory(monkeypatch, tmp_path):
    _stub_external_data(monkeypatch)
    monkeypatch.chdir(tmp_path)

    result = data_loader.fetch_stock_data(save_path="features.csv")

    output = tmp_path / "features.csv"
    assert output.is_file()
    saved = pd.read_csv(output)
    assert len(saved) == len(result)


def test_fetch_stock_data_creates_nested_parent_directories(monkeypatch, tmp_path):
    _stub_external_data(monkeypatch)
    output = tmp_path / "nested" / "deeper" / "features.csv"

    result = data_loader.fetch_stock_data(save_path=str(output))

    assert output.is_file()
    saved = pd.read_csv(output)
    assert len(saved) == len(result)
