"""Smoke tests for the wheat crop growth model and EnKF assimilation."""
import numpy as np
import pytest
from src.crop_model import WheatGrowthModel, EnKF


@pytest.fixture
def drivers():
    """Synthetic 150-day temperature + radiation drivers."""
    rng = np.random.default_rng(42)
    days = np.arange(150)
    temps = 18 + 6 * np.sin(2 * np.pi * (days - 30) / 300) + rng.normal(0, 1, len(days))
    rads = 14 + 4 * np.sin(2 * np.pi * days / 200) + rng.normal(0, 0.8, len(days))
    rads = np.clip(rads, 5, None)
    return temps, rads


class TestWheatGrowthModel:

    def test_simulate_returns_trajectory(self, drivers):
        model = WheatGrowthModel()
        traj = model.simulate(*drivers)
        assert traj.ndim == 2
        assert traj.shape[1] == 3  # tt, lai, biomass
        assert len(traj) > 1

    def test_thermal_time_increases(self, drivers):
        model = WheatGrowthModel()
        traj = model.simulate(*drivers)
        tt = traj[:, 0]
        assert np.all(np.diff(tt) >= 0), "Thermal time must be monotonically non-decreasing"

    def test_biomass_non_negative(self, drivers):
        model = WheatGrowthModel()
        traj = model.simulate(*drivers)
        assert np.all(traj[:, 2] >= 0), "Biomass must never be negative"

    def test_yield_reasonable(self, drivers):
        model = WheatGrowthModel()
        traj = model.simulate(*drivers)
        y = model.yield_t_ha(traj[-1, 2])
        assert 0.5 < y < 25, f"Yield {y:.2f} t/ha is outside a plausible range"

    def test_heat_stress_reduces_yield(self, drivers):
        temps, rads = drivers
        model_normal = WheatGrowthModel()
        traj_normal = model_normal.simulate(temps, rads)

        # Add extreme heat
        temps_hot = temps.copy()
        temps_hot[100:] += 20
        traj_hot = model_normal.simulate(temps_hot, rads)
        yield_hot = model_normal.yield_t_ha(traj_hot[-1, 2])
        yield_normal = model_normal.yield_t_ha(traj_normal[-1, 2])
        assert yield_hot < yield_normal


class TestEnKF:

    def test_assimilation_corrects_lai(self, drivers):
        temps, rads = drivers
        # "True" model with lower RUE → lower LAI
        true_model = WheatGrowthModel(rue=2.3, harvest_index=0.40)
        true_traj = true_model.simulate(temps, rads)

        obs_days = [30, 50, 70, 90, 110]
        lai_obs = [true_traj[min(d, len(true_traj) - 1), 1] for d in obs_days]

        enkf = EnKF(WheatGrowthModel(), n_ens=40, lai_obs_std=0.25)
        mean_traj, ens_yields = enkf.run(temps, rads, lai_obs, obs_days)

        assert mean_traj.shape[1] == 3
        assert len(ens_yields) == 40

    def test_ensemble_yield_spread(self, drivers):
        enkf = EnKF(WheatGrowthModel(), n_ens=50, lai_obs_std=0.3)
        mean_traj, yields = enkf.run(*drivers, lai_obs=[2.0, 3.5], obs_days=[40, 80])
        assert yields.std() > 0, "Ensemble should have non-zero spread"

    def test_forecast_historical(self, drivers):
        temps, rads = drivers
        # Make dummy historical data
        hist_data = [
            (temps + np.random.normal(0, 1, len(temps)), rads + np.random.normal(0, 0.5, len(rads)))
            for _ in range(5)
        ]
        from src.crop_model import EnsembleWeatherGenerator
        gen = EnsembleWeatherGenerator(hist_data, seed=123)
        enkf = EnKF(WheatGrowthModel(), n_ens=20, seed=123)

        T = 80
        total_days = 150
        lai_obs = [1.5, 2.5]
        obs_days = [30, 60]

        mean_traj, yields = enkf.forecast(
            temps, rads, lai_obs, obs_days, T, total_days,
            gen, method='historical'
        )
        assert mean_traj.shape == (total_days, 3)
        assert len(yields) == 20
        assert np.all(mean_traj[:T, 1] >= 0)
        assert np.all(mean_traj[T:, 1] >= 0)

    def test_forecast_ar1(self, drivers):
        temps, rads = drivers
        from src.crop_model import EnsembleWeatherGenerator
        gen = EnsembleWeatherGenerator(seed=456)
        enkf = EnKF(WheatGrowthModel(), n_ens=15, seed=456)

        T = 90
        total_days = 150
        lai_obs = [1.2, 2.2]
        obs_days = [40, 70]

        mean_traj, yields = enkf.forecast(
            temps, rads, lai_obs, obs_days, T, total_days,
            gen, method='ar1'
        )
        assert mean_traj.shape == (total_days, 3)
        assert len(yields) == 15
        assert np.all(mean_traj[:, 2] >= 0)


class TestEnsembleWeatherGenerator:

    def test_historical_resampling_shape_and_values(self):
        from src.crop_model import EnsembleWeatherGenerator
        # 3 years, 10 days each
        hist_data = [
            (np.full(10, float(i)), np.full(10, float(10 * i)))
            for i in range(1, 4)
        ]
        gen = EnsembleWeatherGenerator(hist_data, seed=7)
        obs_t = np.zeros(10)
        obs_r = np.zeros(10)

        t_ens, r_ens = gen.generate_historical_resampling(obs_t, obs_r, T=5, total_days=10, n_ens=5)
        assert t_ens.shape == (5, 10)
        assert r_ens.shape == (5, 10)
        # Check that observed part is 0
        assert np.all(t_ens[:, :5] == 0.0)
        assert np.all(r_ens[:, :5] == 0.0)
        # Check that forecasted part is from historical years (1.0, 2.0, or 3.0)
        for i in range(5):
            val = t_ens[i, 5]
            assert val in [1.0, 2.0, 3.0]
            assert np.all(t_ens[i, 5:] == val)
            assert np.all(r_ens[i, 5:] == 10 * val)

    def test_stochastic_ar1_shape_and_values(self):
        from src.crop_model import EnsembleWeatherGenerator
        gen = EnsembleWeatherGenerator(seed=8)
        obs_t = np.ones(10) * 15.0
        obs_r = np.ones(10) * 12.0

        t_ens, r_ens = gen.generate_stochastic_ar1(obs_t, obs_r, T=6, total_days=10, n_ens=4)
        assert t_ens.shape == (4, 10)
        assert r_ens.shape == (4, 10)
        assert np.all(t_ens[:, :6] == 15.0)
        assert np.all(r_ens[:, :6] == 12.0)
        # Radiation should be clipped to positive values
        assert np.all(r_ens >= 1.0)
