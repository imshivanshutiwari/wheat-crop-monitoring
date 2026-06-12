"""Smoke tests for phenology fitting and NDVI integral."""
import numpy as np
from src.phenology import double_logistic, fit_pixel, ndvi_integral


class TestDoubleLogistic:

    def test_shape_matches_input(self):
        t = np.arange(0, 180, 8)
        result = double_logistic(t, 0.15, 0.85, 35, 140, 0.12, 0.10)
        assert result.shape == t.shape

    def test_output_within_bounds(self):
        t = np.arange(0, 180, 8)
        result = double_logistic(t, 0.15, 0.85, 35, 140, 0.12, 0.10)
        assert np.all(result >= 0.14), "Should not drop below vmin"
        assert np.all(result <= 0.86), "Should not exceed vmax"

    def test_peak_in_middle(self):
        t = np.arange(0, 180, 1)
        result = double_logistic(t, 0.1, 0.9, 40, 140, 0.15, 0.15)
        peak_idx = np.argmax(result)
        assert 40 < t[peak_idx] < 140, "Peak should occur between SOS and EOS"


class TestFitPixel:

    def test_successful_fit(self):
        t = np.arange(0, 180, 8)
        true_ndvi = double_logistic(t, 0.15, 0.85, 35, 140, 0.12, 0.10)
        obs = true_ndvi + np.random.default_rng(42).normal(0, 0.03, len(t))
        result = fit_pixel(t, obs)
        assert result is not None
        assert "sowing_doy" in result
        assert "senescence_doy" in result
        assert "amplitude" in result
        assert result["season_length"] > 0

    def test_recovers_approximate_sos_eos(self):
        t = np.arange(0, 180, 8)
        true_ndvi = double_logistic(t, 0.15, 0.85, 35, 140, 0.12, 0.10)
        result = fit_pixel(t, true_ndvi)
        assert abs(result["sowing_doy"] - 35) < 10
        assert abs(result["senescence_doy"] - 140) < 10

    def test_returns_none_for_insufficient_data(self):
        t = np.array([0, 10, 20])
        ndvi = np.array([0.2, 0.3, 0.4])
        assert fit_pixel(t, ndvi) is None

    def test_handles_nan_values(self):
        t = np.arange(0, 180, 8)
        true_ndvi = double_logistic(t, 0.15, 0.85, 35, 140, 0.12, 0.10)
        noisy = true_ndvi.copy()
        noisy[3] = np.nan
        noisy[7] = np.nan
        result = fit_pixel(t, noisy)
        assert result is not None


class TestNdviIntegral:

    def test_positive_integral(self):
        t = np.arange(0, 180, 8)
        ndvi = double_logistic(t, 0.15, 0.85, 35, 140, 0.12, 0.10)
        val = ndvi_integral(t, ndvi)
        assert val > 0

    def test_higher_ndvi_gives_higher_integral(self):
        t = np.arange(0, 180, 8)
        low = double_logistic(t, 0.1, 0.5, 35, 140, 0.12, 0.10)
        high = double_logistic(t, 0.1, 0.9, 35, 140, 0.12, 0.10)
        assert ndvi_integral(t, high) > ndvi_integral(t, low)
