import pandas as pd

from src.feature_store import (
    load_dataset_meta,
    load_feature_frame,
    monthly_close_summary,
    save_feature_frame,
)


def _frame():
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-02-01"])
    return pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0],
            "Close": [10.5, 11.5, 12.5],
            "Volume": [100.0, 110.0, 120.0],
        },
        index=index,
    )


def test_feature_store_roundtrip_filters_dates_and_blocks_injection(tmp_path):
    database = tmp_path / "market.sqlite"
    save_feature_frame(
        _frame(),
        database,
        ticker="AAPL",
        request_start="2024-01-01",
        request_end="2024-02-28",
        sentiment_mode="neutral",
    )

    loaded = load_feature_frame(database, "AAPL", "2024-01-03", "2024-02-28")
    assert loaded["Close"].tolist() == [11.5, 12.5]
    assert load_feature_frame(database, "AAPL' OR '1'='1", "2024-01-01", "2024-12-31").empty

    meta = load_dataset_meta(database, "AAPL")
    assert meta["n_rows"] == 3
    assert meta["sentiment_mode"] == "neutral"
    assert load_dataset_meta(database, "MSFT") is None


def test_monthly_close_summary_groups_in_sql(tmp_path):
    database = tmp_path / "market.sqlite"
    save_feature_frame(
        _frame(),
        database,
        ticker="AAPL",
        request_start="2024-01-01",
        request_end="2024-02-28",
        sentiment_mode="neutral",
    )

    summary = monthly_close_summary(database, "AAPL")

    assert summary["month"].tolist() == ["2024-01", "2024-02"]
    assert summary.loc[0, "n_days"] == 2
    assert summary.loc[0, "avg_close"] == 11.0
    assert summary.loc[1, "min_close"] == 12.5
