import pandas as pd

import src.sentiment_analyzer as sentiment_analyzer


def test_sentiment_alignment_never_backfills_future_news(monkeypatch):
    headlines = [
        {"date": "2026-01-07", "headline": "positive"},
        {"date": "2026-01-09", "headline": "negative"},
    ]
    scores = {"positive": 0.8, "negative": -0.4}

    monkeypatch.setattr(
        sentiment_analyzer,
        "fetch_news_headlines",
        lambda *args, **kwargs: headlines,
    )
    monkeypatch.setattr(
        sentiment_analyzer,
        "score_headline",
        lambda headline: scores[headline],
    )

    trading_days = pd.date_range("2026-01-05", "2026-01-09", freq="B")
    result = sentiment_analyzer.build_daily_sentiment(
        "TEST", "2026-01-05", "2026-01-09", date_index=trading_days
    )

    expected = pd.Series(
        [0.0, 0.0, 0.8, 0.8, -0.4],
        index=trading_days,
        name="Sentiment",
    )
    pd.testing.assert_series_equal(result, expected)


def test_weekend_news_propagates_to_next_trading_day(monkeypatch):
    headlines = [
        {"date": "2026-01-09", "headline": "friday"},
        {"date": "2026-01-11", "headline": "sunday"},
    ]
    scores = {"friday": -0.3, "sunday": 0.9}

    monkeypatch.setattr(
        sentiment_analyzer,
        "fetch_news_headlines",
        lambda *args, **kwargs: headlines,
    )
    monkeypatch.setattr(
        sentiment_analyzer,
        "score_headline",
        lambda headline: scores[headline],
    )

    trading_days = pd.DatetimeIndex(
        ["2026-01-09", "2026-01-12", "2026-01-13"]
    )
    result = sentiment_analyzer.build_daily_sentiment(
        "TEST", "2026-01-09", "2026-01-13", date_index=trading_days
    )

    expected = pd.Series(
        [-0.3, 0.9, 0.9],
        index=trading_days,
        name="Sentiment",
    )
    pd.testing.assert_series_equal(result, expected)


def test_missing_news_uses_neutral_sentiment(monkeypatch):
    monkeypatch.setattr(
        sentiment_analyzer,
        "fetch_news_headlines",
        lambda *args, **kwargs: [],
    )

    trading_days = pd.date_range("2026-01-05", "2026-01-09", freq="B")
    result = sentiment_analyzer.build_daily_sentiment(
        "TEST", "2026-01-05", "2026-01-09", date_index=trading_days
    )

    expected = pd.Series(0.0, index=trading_days, name="Sentiment")
    pd.testing.assert_series_equal(result, expected)
