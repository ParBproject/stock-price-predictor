"""
sentiment_analyzer.py
---------------------
Generates daily sentiment scores from news headlines using VADER.
Falls back to neutral scores (0.0) if no API key / data is available,
so the rest of the pipeline always has a 'Sentiment' column to work with.
"""

import os
import datetime
import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Optional: set NEWS_API_KEY environment variable to fetch live headlines
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

_analyzer = SentimentIntensityAnalyzer()


def score_headline(text: str) -> float:
    """Returns VADER compound score in [-1, 1]."""
    return _analyzer.polarity_scores(str(text))["compound"]


def fetch_news_headlines(query: str,
                         from_date: str,
                         to_date:   str,
                         api_key:   str = NEWS_API_KEY,
                         page_size: int = 100) -> list[dict]:
    """
    Fetches headlines from NewsAPI.org.
    Returns a list of {date, headline} dicts.
    Requires a free API key: https://newsapi.org
    """
    if not api_key:
        print("[Sentiment] NEWS_API_KEY not set – returning empty list.")
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q":        query,
        "from":     from_date,
        "to":       to_date,
        "language": "en",
        "sortBy":   "publishedAt",
        "pageSize": page_size,
        "apiKey":   api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        results = []
        for art in articles:
            pub = art.get("publishedAt", "")[:10]   # YYYY-MM-DD
            title = art.get("title") or art.get("description") or ""
            results.append({"date": pub, "headline": title})
        return results
    except Exception as e:
        print(f"[Sentiment] NewsAPI request failed: {e}")
        return []


def build_daily_sentiment(ticker:    str,
                           start:    str,
                           end:      str,
                           date_index: pd.DatetimeIndex | None = None) -> pd.Series:
    """
    Returns a daily Series of mean compound VADER scores for `ticker` news.

    If NEWS_API_KEY is not available, falls back to synthetic random-walk
    scores (seeded for reproducibility) so the pipeline still runs.

    Parameters
    ----------
    ticker      : stock symbol used as the search query
    start, end  : date strings 'YYYY-MM-DD'
    date_index  : optional trading-day index to reindex / forward-fill onto
    """
    headlines = fetch_news_headlines(ticker, start, end)

    if headlines:
        records = [{"date": h["date"],
                    "score": score_headline(h["headline"])}
                   for h in headlines]
        sent_df = pd.DataFrame(records)
        sent_df["date"] = pd.to_datetime(sent_df["date"])
        daily  = sent_df.groupby("date")["score"].mean()
        print(f"[Sentiment] Computed scores for {len(daily)} trading days "
              f"from {len(headlines)} headlines.")
    else:
        # Synthetic fallback – random walk clamped to [-1, 1]
        import numpy as np
        np.random.seed(42)
        idx = pd.date_range(start, end, freq="B")   # business days
        scores = np.random.randn(len(idx)).cumsum()
        scores = scores / (np.abs(scores).max() + 1e-9)  # normalise
        daily  = pd.Series(scores, index=idx)
        print("[Sentiment] Using synthetic (random-walk) sentiment fallback.")

    daily.name = "Sentiment"

    if date_index is not None:
        daily = daily.reindex(date_index).ffill().bfill()

    return daily


def add_sentiment_to_df(df: pd.DataFrame,
                        ticker: str,
                        start:  str | None = None,
                        end:    str | None = None) -> pd.DataFrame:
    """
    Convenience wrapper: merges the sentiment series into an existing
    OHLCV / feature dataframe (indexed by date).
    """
    start = start or str(df.index.min())[:10]
    end   = end   or str(df.index.max())[:10]

    sentiment = build_daily_sentiment(ticker, start, end,
                                      date_index=df.index)
    df["Sentiment"] = sentiment.values
    return df


# ── CLI demo ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    s = build_daily_sentiment("AAPL", "2024-01-01", "2024-06-30")
    print(s.head(10))
