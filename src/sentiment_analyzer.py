"""
sentiment_analyzer.py
---------------------
Generates daily sentiment scores from news headlines using VADER.
Falls back to neutral scores (0.0) if no API key / data is available,
so the rest of the pipeline always has a 'Sentiment' column to work with.
"""

from functools import lru_cache
import os
import datetime
import requests
import pandas as pd
import pandas_market_calendars as mcal
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Optional: set NEWS_API_KEY environment variable to fetch live headlines
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
MARKET_TIMEZONE = "America/New_York"

_analyzer = SentimentIntensityAnalyzer()
_NYSE_CALENDAR = mcal.get_calendar("NYSE")


def score_headline(text: str) -> float:
    """Returns VADER compound score in [-1, 1]."""
    return _analyzer.polarity_scores(str(text))["compound"]


@lru_cache(maxsize=512)
def _nyse_market_close(local_date: str) -> pd.Timestamp | None:
    """Return the actual NYSE close for a local calendar date, if it is a session."""
    schedule = _NYSE_CALENDAR.schedule(start_date=local_date, end_date=local_date)
    if schedule.empty:
        return None

    close = pd.Timestamp(schedule.iloc[0]["market_close"])
    if close.tzinfo is None:
        close = close.tz_localize("UTC")
    return close.tz_convert(MARKET_TIMEZONE)


def headline_effective_market_date(published_at: str) -> str:
    """Map a publication timestamp to the date its information was tradable.

    NewsAPI timestamps are normalized to U.S. Eastern time. On NYSE sessions,
    headlines published at or after that session's actual market close become
    effective on the next calendar day. This handles scheduled early closes as
    well as normal 4:00 PM closes. On non-trading days the local calendar date
    is preserved so downstream trading-day alignment can carry the observation
    to the next market row.
    """
    timestamp = pd.to_datetime(published_at, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return ""

    market_time = timestamp.tz_convert(MARKET_TIMEZONE)
    effective_date = market_time.normalize()
    local_date = market_time.strftime("%Y-%m-%d")
    market_close = _nyse_market_close(local_date)

    if market_close is not None and market_time >= market_close:
        effective_date += pd.Timedelta(days=1)

    return effective_date.strftime("%Y-%m-%d")


def fetch_news_headlines(query: str,
                         from_date: str,
                         to_date:   str,
                         api_key:   str = NEWS_API_KEY,
                         page_size: int = 100) -> list[dict]:
    """
    Fetches all available headline pages from NewsAPI.org.
    Returns a list of {date, headline} dicts.
    Requires a NewsAPI key: https://newsapi.org
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
    results = []
    page = 1

    while True:
        params["page"] = page
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            print(f"[Sentiment] NewsAPI page {page} request failed: {e}")
            break

        articles = payload.get("articles", [])

        for art in articles:
            pub = headline_effective_market_date(art.get("publishedAt", ""))
            title = art.get("title") or art.get("description") or ""
            results.append({"date": pub, "headline": title})

        total_results = payload.get("totalResults")
        if not articles:
            break
        if isinstance(total_results, int) and len(results) >= total_results:
            break
        if len(articles) < page_size:
            break
        page += 1

    return results


def build_daily_sentiment(ticker:    str,
                           start:    str,
                           end:      str,
                           date_index: pd.DatetimeIndex | None = None) -> pd.Series:
    """
    Returns a daily Series of mean compound VADER scores for `ticker` news.

    If NEWS_API_KEY or headline data is not available, returns neutral scores
    (0.0) so missing external data does not become a fabricated model signal.

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
        idx = pd.date_range(start, end, freq="B")   # business days
        daily = pd.Series(0.0, index=idx)
        print("[Sentiment] No headline data – using neutral sentiment fallback.")

    daily.name = "Sentiment"

    if date_index is not None:
        # Preserve non-trading observations (for example weekend news) while
        # propagating only information that was already known forward in time.
        alignment_index = daily.index.union(date_index).sort_values()
        daily = (
            daily.reindex(alignment_index)
            .ffill()
            .reindex(date_index)
            .fillna(0.0)
        )

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
