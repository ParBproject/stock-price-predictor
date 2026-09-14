import pandas as pd
import pytest

import src.sentiment_analyzer as sentiment_analyzer


@pytest.mark.parametrize(
    "published_at,expected_date",
    [
        # Winter: Eastern Standard Time is UTC-5.
        ("2026-01-05T20:59:59Z", "2026-01-05"),
        ("2026-01-05T21:00:00Z", "2026-01-06"),
        # Summer: Eastern Daylight Time is UTC-4.
        ("2026-07-06T19:59:59Z", "2026-07-06"),
        ("2026-07-06T20:00:00Z", "2026-07-07"),
    ],
)
def test_headline_effective_market_date_respects_eastern_close(
    published_at, expected_date
):
    assert sentiment_analyzer.headline_effective_market_date(published_at) == expected_date


def test_headline_effective_market_date_rejects_invalid_timestamp():
    assert sentiment_analyzer.headline_effective_market_date("not-a-timestamp") == ""


def test_fetch_news_headlines_paginates_until_total_results(monkeypatch):
    payloads = {
        1: {
            "totalResults": 3,
            "articles": [
                {"publishedAt": "2026-01-03T12:00:00Z", "title": "newest"},
                {"publishedAt": "2026-01-02T12:00:00Z", "title": "middle"},
            ],
        },
        2: {
            "totalResults": 3,
            "articles": [
                {"publishedAt": "2026-01-01T12:00:00Z", "title": "oldest"},
            ],
        },
    }
    requested_pages = []

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, params, timeout):
        requested_pages.append(params["page"])
        return FakeResponse(payloads[params["page"]])

    monkeypatch.setattr(sentiment_analyzer.requests, "get", fake_get)

    result = sentiment_analyzer.fetch_news_headlines(
        "TEST",
        "2026-01-01",
        "2026-01-03",
        api_key="test-key",
        page_size=2,
    )

    assert requested_pages == [1, 2]
    assert result == [
        {"date": "2026-01-03", "headline": "newest"},
        {"date": "2026-01-02", "headline": "middle"},
        {"date": "2026-01-01", "headline": "oldest"},
    ]


def test_fetch_news_headlines_preserves_prior_pages_after_later_failure(monkeypatch):
    requested_pages = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "totalResults": 4,
                "articles": [
                    {"publishedAt": "2026-01-04T12:00:00Z", "title": "newest"},
                    {"publishedAt": "2026-01-03T12:00:00Z", "title": "older"},
                ],
            }

    def fake_get(url, params, timeout):
        page = params["page"]
        requested_pages.append(page)
        if page == 1:
            return FakeResponse()
        raise sentiment_analyzer.requests.RequestException("temporary failure")

    monkeypatch.setattr(sentiment_analyzer.requests, "get", fake_get)

    result = sentiment_analyzer.fetch_news_headlines(
        "TEST",
        "2026-01-01",
        "2026-01-04",
        api_key="test-key",
        page_size=2,
    )

    assert requested_pages == [1, 2]
    assert result == [
        {"date": "2026-01-04", "headline": "newest"},
        {"date": "2026-01-03", "headline": "older"},
    ]


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
