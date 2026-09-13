import matplotlib
import numpy as np

matplotlib.use("Agg")

from src.evaluator import classification_metrics, plot_confusion_matrix


def test_classification_metrics_handles_all_up_holdout(capsys):
    y_true = np.array([1, 1, 1])
    y_pred = np.array([1, 1, 1])

    metrics = classification_metrics(y_true, y_pred, label="All Up")
    output = capsys.readouterr().out

    assert metrics == {
        "Accuracy": 1.0,
        "Precision": 1.0,
        "Recall": 1.0,
        "F1": 1.0,
    }
    assert "Down" in output
    assert "Up" in output


def test_classification_metrics_handles_all_down_holdout(capsys):
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])

    metrics = classification_metrics(y_true, y_pred, label="All Down")
    output = capsys.readouterr().out

    assert metrics == {
        "Accuracy": 1.0,
        "Precision": 0.0,
        "Recall": 0.0,
        "F1": 0.0,
    }
    assert "Down" in output
    assert "Up" in output


def test_plot_confusion_matrix_keeps_two_by_two_binary_shape(monkeypatch):
    captured = {}

    def fake_heatmap(data, *args, **kwargs):
        captured["shape"] = data.shape
        captured["xticklabels"] = kwargs["xticklabels"]
        captured["yticklabels"] = kwargs["yticklabels"]
        return kwargs["ax"]

    monkeypatch.setattr("src.evaluator.sns.heatmap", fake_heatmap)
    monkeypatch.setattr("src.evaluator.plt.show", lambda: None)

    plot_confusion_matrix(
        np.array([1, 1, 1]),
        np.array([1, 1, 1]),
        save=False,
    )

    assert captured["shape"] == (2, 2)
    assert captured["xticklabels"] == ["Down", "Up"]
    assert captured["yticklabels"] == ["Down", "Up"]
