"""SQLite cache for engineered daily feature rows."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FEATURE_TABLE = "daily_features"
_META_TABLE = "dataset_meta"


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection, creating parent directories as needed."""
    path = Path(db_path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def _validate_column_names(columns: list[str]) -> None:
    invalid = [column for column in columns if not _IDENTIFIER.fullmatch(column)]
    if invalid:
        raise ValueError(f"Feature names must be SQL identifiers: {invalid}")


def _prepare_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    frame = df.copy()
    if isinstance(frame.index, pd.DatetimeIndex) or frame.index.name is not None:
        frame = frame.reset_index()
    date_column = frame.columns[0]
    frame = frame.rename(columns={date_column: "trade_date"})
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.strftime("%Y-%m-%d")
    if frame["trade_date"].isna().any():
        raise ValueError("Feature rows contain dates that could not be parsed")
    frame.insert(0, "ticker", ticker)
    feature_columns = [column for column in frame.columns if column not in {"ticker", "trade_date"}]
    _validate_column_names(feature_columns)
    numeric = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Feature rows must be numeric before they are stored")
    frame[feature_columns] = numeric
    return frame


def save_feature_frame(
    df: pd.DataFrame,
    db_path: str | Path,
    ticker: str,
    request_start: str,
    request_end: str,
    sentiment_mode: str,
) -> int:
    """Replace stored rows for ``ticker`` and record how they were requested.

    The table is rebuilt from the frame's columns so a later feature-schema
    change cannot leave a stale SQLite table in place.
    """
    frame = _prepare_frame(df, ticker)
    feature_columns = [column for column in frame.columns if column not in {"ticker", "trade_date"}]
    column_sql = ", ".join(f'"{column}" REAL' for column in feature_columns)
    insert_columns = ", ".join(["ticker", "trade_date", *[f'"{column}"' for column in feature_columns]])
    placeholders = ", ".join(["?"] * (2 + len(feature_columns)))
    rows = list(frame.itertuples(index=False, name=None))

    with connect(db_path) as connection:
        connection.execute(f"DROP TABLE IF EXISTS {_FEATURE_TABLE}")
        connection.execute(
            f"""
            CREATE TABLE {_FEATURE_TABLE} (
                ticker TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                {column_sql},
                PRIMARY KEY (ticker, trade_date)
            )
            """
        )
        connection.executemany(
            f"INSERT INTO {_FEATURE_TABLE} ({insert_columns}) VALUES ({placeholders})",
            rows,
        )
        connection.execute(f"DROP TABLE IF EXISTS {_META_TABLE}")
        connection.execute(
            f"""
            CREATE TABLE {_META_TABLE} (
                ticker TEXT PRIMARY KEY,
                request_start TEXT NOT NULL,
                request_end TEXT NOT NULL,
                sentiment_mode TEXT NOT NULL,
                n_rows INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            f"""
            INSERT INTO {_META_TABLE} (
                ticker, request_start, request_end, sentiment_mode, n_rows
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (ticker, request_start, request_end, sentiment_mode, len(rows)),
        )
    return len(rows)


def load_feature_frame(
    db_path: str | Path,
    ticker: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Load inclusive ``[start, end]`` rows with a parameterized SQL query."""
    query = f"""
        SELECT *
        FROM {_FEATURE_TABLE}
        WHERE ticker = ?
          AND trade_date >= ?
          AND trade_date <= ?
        ORDER BY trade_date
    """
    with connect(db_path) as connection:
        frame = pd.read_sql_query(query, connection, params=(ticker, start, end))
    if frame.empty:
        return frame
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame.set_index("trade_date").drop(columns=["ticker"])


def load_dataset_meta(db_path: str | Path, ticker: str) -> dict | None:
    """Return the stored request metadata for ``ticker``, if the cache exists."""
    path = Path(db_path)
    if not path.exists():
        return None
    query = f"""
        SELECT ticker, request_start, request_end, sentiment_mode, n_rows
        FROM {_META_TABLE}
        WHERE ticker = ?
    """
    with connect(path) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (_META_TABLE,),
        ).fetchall()
        if not tables:
            return None
        row = connection.execute(query, (ticker,)).fetchone()
    if row is None:
        return None
    return dict(row)


def monthly_close_summary(db_path: str | Path, ticker: str) -> pd.DataFrame:
    """Aggregate stored closes by calendar month."""
    query = f"""
        SELECT substr(trade_date, 1, 7) AS month,
               AVG(Close) AS avg_close,
               MIN(Close) AS min_close,
               MAX(Close) AS max_close,
               COUNT(*) AS n_days
        FROM {_FEATURE_TABLE}
        WHERE ticker = ?
        GROUP BY month
        ORDER BY month
    """
    with connect(db_path) as connection:
        return pd.read_sql_query(query, connection, params=(ticker,))
