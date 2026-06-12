"""Smoke tests for FCLS linear spectral unmixing."""
import numpy as np
from src.unmixing import fcls_unmix, wheat_fraction_area


# Realistic AWiFS endmembers: (Green, Red, NIR, SWIR)
WHEAT = np.array([0.05, 0.04, 0.45, 0.20])
SOIL = np.array([0.15, 0.22, 0.28, 0.35])
OTHER = np.array([0.06, 0.05, 0.38, 0.25])
ENDMEMBERS = np.vstack([WHEAT, SOIL, OTHER])


class TestFCLS:

    def test_pure_pixel_recovers_identity(self):
        """A pure wheat pixel should yield ~100% wheat fraction."""
        ab = fcls_unmix(WHEAT, ENDMEMBERS)
        assert ab.shape == (1, 3)
        assert ab[0, 0] > 0.9, f"Pure wheat pixel got wheat fraction {ab[0, 0]:.3f}"

    def test_fractions_sum_to_one(self):
        rng = np.random.default_rng(42)
        true_fracs = rng.dirichlet([3, 2, 1], size=10)
        pixels = true_fracs @ ENDMEMBERS + rng.normal(0, 0.002, (10, 4))
        ab = fcls_unmix(pixels, ENDMEMBERS)
        sums = ab.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=0.05)

    def test_fractions_non_negative(self):
        rng = np.random.default_rng(42)
        pixels = rng.dirichlet([3, 2, 1], size=20) @ ENDMEMBERS
        ab = fcls_unmix(pixels, ENDMEMBERS)
        assert np.all(ab >= -0.01), "FCLS fractions should be non-negative"

    def test_known_mixture(self):
        """60% wheat / 30% soil / 10% other should be roughly recovered."""
        true_frac = np.array([0.6, 0.3, 0.1])
        mix = true_frac @ ENDMEMBERS
        ab = fcls_unmix(mix, ENDMEMBERS)
        np.testing.assert_allclose(ab[0], true_frac, atol=0.1)


class TestWheatFractionArea:

    def test_area_computation(self):
        ab = np.array([[0.6, 0.3, 0.1], [0.8, 0.1, 0.1]])
        area = wheat_fraction_area(ab, wheat_idx=0, pixel_area_ha=0.3136)
        expected = (0.6 + 0.8) * 0.3136
        assert abs(area - expected) < 0.01

    def test_all_wheat_area(self):
        ab = np.ones((100, 3)) / 3
        ab[:, 0] = 1.0  # make all wheat
        ab[:, 1:] = 0.0
        area = wheat_fraction_area(ab, wheat_idx=0, pixel_area_ha=1.0)
        assert abs(area - 100.0) < 0.01
