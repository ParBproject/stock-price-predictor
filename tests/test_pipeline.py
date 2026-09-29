import numpy as np
import pandas as pd

from src.backtesting import assemble_signal_frame
from src.model_trainer import train_random_forest_regressor
from src.pipeline import RF_FEATURE_COLUMNS, PipelineConfig, command_line, run_pipeline


def test_random_forest_trainer_honors_estimator_count():
    features = pd.DataFrame(
        {"a": np.arange(12, dtype=float), "b": np.arange(12, dtype=float) + 1}
    )
    target = np.arange(12, dtype=float) + 0.5

    model = train_random_forest_regressor(
        features, target, tune=False, n_estimators=7, n_jobs=1
    )

    assert model.n_estimators == 7


def test_signal_frame_leaves_the_terminal_bar_without_a_new_signal():
    index = pd.bdate_range("2024-01-01", periods=4)
    frame = pd.DataFrame(
        {
            "Open": [1, 2, 3, 4],
            "High": [2, 3, 4, 5],
            "Low": [0.5, 1.5, 2.5, 3.5],
            "Close": [1.5, 2.5, 3.5, 4.5],
            "Volume": [10, 10, 10, 10],
        },
        index=index,
    )

    assembled = assemble_signal_frame(
        frame,
        feature_index=index[:2],
        target_index=index[1:3],
        predictions=np.array([2.0, 3.0]),
        signals=np.array([1, 0]),
    )

    assert assembled["Signal"].tolist() == [1, 0, -1]
    assert np.isnan(assembled["Pred_Close"].iloc[-1])


def test_notebook_uses_the_pipeline_feature_list():
    source = open("notebooks/random_forest_model.ipynb", encoding="utf-8").read()
    for column in RF_FEATURE_COLUMNS:
        assert f"'{column}'" in source


def test_demo_pipeline_writes_real_computed_metrics(tmp_path):
    config = PipelineConfig(
        demo=True,
        n_estimators=15,
        n_splits=2,
        n_jobs=1,
        output_dir=tmp_path / "results",
        database_path=tmp_path / "market.sqlite",
    )

    metrics = run_pipeline(config)

    assert metrics["demo"] is True
    assert metrics["ticker"] == "DEMO"
    assert metrics["walk_forward"]["mean_persistence_mae"] > 0
    assert "Sentiment" in metrics["dropped_constant_features"]
    assert (tmp_path / "results" / "metrics.json").exists()
    assert (tmp_path / "results" / "baseline_comparison.png").exists()
    assert (tmp_path / "results" / "equity_curve.png").exists()
    assert metrics["final_holdout"]["strategy"]["engine"] == "backtrader"
    assert metrics["final_holdout"]["strategy"]["final_value"] > 0
    assert "--demo" in command_line(config)
