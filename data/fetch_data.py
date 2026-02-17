"""
fetch_data.py
-------------
Standalone script to download fresh stock data and save to data/.
Run from repo root:  python data/fetch_data.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import fetch_stock_data
from src.sentiment_analyzer import add_sentiment_to_df

TICKER = "AAPL"
START  = "2015-01-01"
END    = "2024-12-31"
OUT    = os.path.join(os.path.dirname(__file__), f"{TICKER}_features.csv")

if __name__ == "__main__":
    df = fetch_stock_data(TICKER, START, END)
    df = add_sentiment_to_df(df, TICKER, START, END)
    df.to_csv(OUT)
    print(f"Saved {len(df)} rows → {OUT}")
