import json
from pathlib import Path


def _code_source(path: str) -> str:
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def test_backtest_sharpe_uses_annual_rate_with_daily_annualization():
    source = _code_source("notebooks/backtesting.ipynb")

    assert "ANNUAL_RISK_FREE_RATE = 0.04" in source
    assert "riskfreerate=ANNUAL_RISK_FREE_RATE" in source
    assert "timeframe=bt.TimeFrame.Days" in source
    assert "annualize=True" in source
    assert "riskfreerate=0.04/252" not in source
    assert "riskfreerate=0.04 / 252" not in source
