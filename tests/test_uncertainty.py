"""Smoke tests for uncertainty quantification: conformal + ensemble intervals."""
import numpy as np
import pytest
from src.uncertainty import SplitConformal, ensemble_intervals


class TestSplitConformal:

    def test_calibrate_returns_quantile(self):
        rng = np.random.default_rng(42)
        y_true = rng.normal(3.0, 0.5, 100)
        y_pred = y_true + rng.normal(0, 0.2, 100)
        conf = SplitConformal(alpha=0.1)
        q = conf.calibrate(y_true, y_pred)
        assert q > 0, "Conformal quantile must be positive"

    def test_coverage_meets_target(self):
        rng = np.random.default_rng(42)
        n = 500
        y_true = rng.normal(3.0, 0.5, n)
        y_pred = y_true + rng.normal(0, 0.2, n)

        # Split: first half = calibration, second half = test
        conf = SplitConformal(alpha=0.1)
        conf.calibrate(y_true[:250], y_pred[:250])
        lo, hi = conf.interval(y_pred[250:])
        coverage = ((y_true[250:] >= lo) & (y_true[250:] <= hi)).mean()
        assert coverage >= 0.85, f"Coverage {coverage:.2%} is too low (target ≥ 90%)"

    def test_interval_raises_without_calibration(self):
        conf = SplitConformal(alpha=0.1)
        with pytest.raises(RuntimeError):
            conf.interval(np.array([1.0, 2.0]))

    def test_symmetric_intervals(self):
        conf = SplitConformal(alpha=0.1)
        conf.calibrate(np.array([1.0, 2.0, 3.0]), np.array([1.1, 2.1, 3.1]))
        lo, hi = conf.interval(np.array([5.0]))
        assert abs((hi[0] - 5.0) - (5.0 - lo[0])) < 1e-10


class TestEnsembleIntervals:

    def test_mean_and_bounds(self):
        ens = np.random.default_rng(42).normal(4.0, 0.5, 100)
        mean, lo, hi = ensemble_intervals(ens, alpha=0.1)
        assert lo < mean < hi
        assert lo > 0  # yields should be positive

    def test_narrower_with_low_variance(self):
        ens_tight = np.random.default_rng(42).normal(4.0, 0.1, 100)
        ens_wide = np.random.default_rng(42).normal(4.0, 1.0, 100)
        _, lo_t, hi_t = ensemble_intervals(ens_tight)
        _, lo_w, hi_w = ensemble_intervals(ens_wide)
        assert (hi_t - lo_t) < (hi_w - lo_w)
