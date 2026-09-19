import unittest

import numpy as np

from src.research_evaluation import (
    compare_with_random_walk,
    evaluate_one_step_forecasts,
    random_walk_forecast,
    trailing_mean_return_forecast,
    walk_forward_windows,
)


class ResearchEvaluationTests(unittest.TestCase):
    def test_random_walk_uses_only_current_close(self) -> None:
        current = np.array([100.0, 101.0, 99.0])
        self.assertTrue(np.array_equal(random_walk_forecast(current), current))

    def test_perfect_forecast_has_zero_return_error(self) -> None:
        current = np.array([100.0, 102.0, 101.0, 104.0])
        actual = np.array([102.0, 101.0, 104.0, 106.0])
        result = evaluate_one_step_forecasts(
            actual,
            current,
            actual,
            risk_free_rate=0.0,
        )
        self.assertAlmostEqual(result.price_mae, 0.0)
        self.assertAlmostEqual(result.return_mae, 0.0)
        self.assertAlmostEqual(result.directional_accuracy, 1.0)

    def test_random_walk_comparison_uses_same_sample(self) -> None:
        current = np.array([100.0, 101.0, 102.0, 103.0])
        actual = np.array([101.0, 100.0, 104.0, 102.0])
        predicted = np.array([101.0, 100.5, 103.0, 102.5])
        comparison = compare_with_random_walk(
            predicted,
            current,
            actual,
            risk_free_rate=0.0,
        )
        self.assertEqual(list(comparison.columns), ["Model", "Random Walk"])
        self.assertIn("directional_accuracy", comparison.index)

    def test_trailing_mean_forecast_never_uses_future_returns(self) -> None:
        prices = np.array([100.0, 101.0, 103.0, 106.0])
        forecast = trailing_mean_return_forecast(prices, window=2)
        expected_return_at_index_2 = np.mean([0.01, 103.0 / 101.0 - 1.0])
        self.assertAlmostEqual(
            forecast[2],
            prices[2] * (1.0 + expected_return_at_index_2),
        )

    def test_walk_forward_windows_are_chronological(self) -> None:
        windows = walk_forward_windows(
            100,
            min_train_size=40,
            test_size=10,
            step_size=10,
            expanding=True,
        )
        self.assertGreater(len(windows), 0)
        for window in windows:
            self.assertLessEqual(window.train_end, window.test_start)
            self.assertEqual(window.train_start, 0)

    def test_rolling_windows_keep_fixed_train_length(self) -> None:
        windows = walk_forward_windows(
            100,
            min_train_size=40,
            test_size=10,
            step_size=10,
            expanding=False,
        )
        for window in windows:
            self.assertEqual(window.train_end - window.train_start, 40)


if __name__ == "__main__":
    unittest.main()
