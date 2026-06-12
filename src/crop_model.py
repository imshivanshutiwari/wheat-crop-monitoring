"""Simplified WOFOST-style wheat growth model + Ensemble Kalman Filter
assimilation of satellite-derived LAI.

This is a light, dependency-free crop simulator capturing the core
dynamics used in operational yield work: thermal-time driven phenology,
LAI development, biomass accumulation via radiation-use efficiency, and
yield as a harvest-index fraction of biomass. The EnKF nudges the model
state towards remotely sensed LAI to correct for unmodelled stress.

NOTE: For regulatory/operational deployment use the validated PCSE/WOFOST
or DSSAT engines. This module provides the assimilation *framework* and a
runnable, transparent surrogate.
"""
import numpy as np


class WheatGrowthModel:
    """Daily-step wheat growth simulator.

    State: thermal_time, LAI, biomass (kg/ha).
    Drivers: daily mean temperature, solar radiation (MJ/m2/day).
    """

    def __init__(self, t_base=0.0, t_sum_maturity=2000.0, rue=2.8,
                 sla=0.0022, k_ext=0.6, harvest_index=0.45,
                 heat_threshold=34.0):
        self.t_base = t_base                 # base temperature (C)
        self.t_sum_maturity = t_sum_maturity  # GDD to maturity
        self.rue = rue                       # radiation-use efficiency g/MJ
        self.sla = sla                       # specific leaf area ha/kg
        self.k_ext = k_ext                   # canopy extinction coefficient
        self.harvest_index = harvest_index
        self.heat_threshold = heat_threshold  # terminal heat-stress (C)

    def step(self, state, temp, radiation):
        """Advance one day. state = [tt, lai, biomass]."""
        tt, lai, biomass = state
        gdd = max(temp - self.t_base, 0.0)
        tt_new = tt + gdd
        # fraction of intercepted radiation (Beer-Lambert)
        fapar = 1.0 - np.exp(-self.k_ext * lai)
        # heat-stress penalty on assimilation
        stress = 1.0 if temp < self.heat_threshold else \
            max(0.0, 1.0 - 0.08 * (temp - self.heat_threshold))
        dbiomass = self.rue * radiation * fapar * stress * 10.0  # g/m2->kg/ha
        biomass_new = biomass + dbiomass
        # LAI grows during vegetative phase, senesces after ~60% maturity
        if tt_new < 0.6 * self.t_sum_maturity:
            lai_new = lai + self.sla * dbiomass
        else:
            lai_new = max(0.0, lai - 0.03 * lai)
        return np.array([tt_new, lai_new, biomass_new])

    def simulate(self, temps, rads, lai0=0.05):
        state = np.array([0.0, lai0, 0.0])
        traj = [state.copy()]
        for temp, rad in zip(temps, rads):
            if state[0] >= self.t_sum_maturity:
                break
            state = self.step(state, temp, rad)
            traj.append(state.copy())
        return np.array(traj)

    def yield_t_ha(self, biomass_kg_ha):
        return self.harvest_index * biomass_kg_ha / 1000.0


class EnKF:
    """Ensemble Kalman Filter for assimilating satellite LAI into the model.

    Updates the LAI state (index 1) when an observation arrives, then the
    biomass/yield trajectory reflects the corrected canopy.
    """

    def __init__(self, model, n_ens=50, lai_obs_std=0.3,
                 param_std=0.1, seed=0):
        self.model = model
        self.n_ens = n_ens
        self.lai_obs_std = lai_obs_std
        self.rng = np.random.default_rng(seed)
        self.param_std = param_std

    def _perturb_drivers(self, temps, rads):
        ens = []
        for _ in range(self.n_ens):
            tp = temps * (1 + self.rng.normal(0, 0.03, len(temps)))
            rp = rads * (1 + self.rng.normal(0, 0.05, len(rads)))
            ens.append((tp, rp))
        return ens

    def run(self, temps, rads, lai_obs, obs_days):
        """Assimilate LAI observations at obs_days.

        Returns (mean_trajectory, ensemble_yields).
        """
        drivers = self._perturb_drivers(np.asarray(temps, float),
                                        np.asarray(rads, float))
        states = np.array([[0.0, 0.05 + abs(self.rng.normal(0, 0.01)), 0.0]
                           for _ in range(self.n_ens)])
        obs_map = dict(zip(obs_days, lai_obs))
        n_days = len(temps)
        mean_traj = []
        for day in range(n_days):
            for k in range(self.n_ens):
                tp, rp = drivers[k]
                states[k] = self.model.step(states[k], tp[day], rp[day])
            if day in obs_map:                       # EnKF update on LAI
                lai_ens = states[:, 1]
                obs = obs_map[day] + self.rng.normal(0, self.lai_obs_std,
                                                     self.n_ens)
                var_f = lai_ens.var() + 1e-6
                kal = var_f / (var_f + self.lai_obs_std ** 2)
                states[:, 1] = lai_ens + kal * (obs - lai_ens)
                states[:, 1] = np.clip(states[:, 1], 0, None)
            mean_traj.append(states.mean(0))
        yields = np.array([self.model.yield_t_ha(s[2]) for s in states])
        return np.array(mean_traj), yields

    def forecast(
        self, temps, rads, lai_obs, obs_days, T, total_days,
        weather_generator, method='historical', **kwargs
    ):
        """Runs the EnKF up to day T (assimilating observations on days < T),
        then projects the states forward to total_days using forecast ensembles.

        method: 'historical' or 'ar1'
        kwargs: passed to weather generator method

        Returns:
            mean_trajectory: shape (total_days, 3)
            ensemble_yields: shape (n_ens,)
        """
        states = np.array([[0.0, 0.05 + abs(self.rng.normal(0, 0.01)), 0.0]
                           for _ in range(self.n_ens)])
        drivers = self._perturb_drivers(np.asarray(temps, float),
                                        np.asarray(rads, float))
        obs_map = dict(zip(obs_days, lai_obs))
        mean_traj = []

        for day in range(T):
            for k in range(self.n_ens):
                tp, rp = drivers[k]
                states[k] = self.model.step(states[k], tp[day], rp[day])
            if day in obs_map:
                lai_ens = states[:, 1]
                obs = obs_map[day] + self.rng.normal(0, self.lai_obs_std,
                                                     self.n_ens)
                var_f = lai_ens.var() + 1e-6
                kal = var_f / (var_f + self.lai_obs_std ** 2)
                states[:, 1] = lai_ens + kal * (obs - lai_ens)
                states[:, 1] = np.clip(states[:, 1], 0, None)
            mean_traj.append(states.mean(0))

        if method == 'historical':
            forecast_temps, forecast_rads = weather_generator.generate_historical_resampling(
                temps, rads, T, total_days, n_ens=self.n_ens, **kwargs
            )
        elif method == 'ar1':
            forecast_temps, forecast_rads = weather_generator.generate_stochastic_ar1(
                temps, rads, T, total_days, n_ens=self.n_ens, **kwargs
            )
        else:
            raise ValueError(f"Unknown forecast method: {method}")

        for day in range(T, total_days):
            for k in range(self.n_ens):
                states[k] = self.model.step(
                    states[k], forecast_temps[k, day], forecast_rads[k, day]
                )
            mean_traj.append(states.mean(0))

        yields = np.array([self.model.yield_t_ha(s[2]) for s in states])
        return np.array(mean_traj), yields


class EnsembleWeatherGenerator:
    """Generates weather ensembles for crop modeling forecasts."""

    def __init__(self, historical_data=None, seed=42):
        """
        historical_data: list of tuples/arrays [(temps, rads), ...] of shape (n_days, 2)
        """
        self.historical_data = historical_data
        self.rng = np.random.default_rng(seed)
        if historical_data is not None:
            self.hist_temps = [np.asarray(h[0], float) for h in historical_data]
            self.hist_rads = [np.asarray(h[1], float) for h in historical_data]
            self.n_years = len(historical_data)
            max_days = max(len(t) for t in self.hist_temps)
            self.climatology_temp = np.zeros(max_days)
            self.climatology_rad = np.zeros(max_days)
            for d in range(max_days):
                t_vals = [t[d] for t in self.hist_temps if d < len(t)]
                r_vals = [r[d] for r in self.hist_rads if d < len(r)]
                self.climatology_temp[d] = np.mean(t_vals)
                self.climatology_rad[d] = np.mean(r_vals)
        else:
            self.hist_temps = None
            self.hist_rads = None
            self.climatology_temp = None
            self.climatology_rad = None

    def generate_historical_resampling(
        self, observed_temps, observed_rads, T, total_days, n_ens=50
    ):
        """Generates n_ens weather sequences.
        For each member, days 0 to T-1 are observed weather.
        Days T to total_days-1 are randomly resampled from historical years.
        """
        observed_temps = np.asarray(observed_temps, float)
        observed_rads = np.asarray(observed_rads, float)
        ensemble_temps = []
        ensemble_rads = []

        if self.historical_data is None or len(self.historical_data) == 0:
            raise ValueError("Historical data must be provided for historical resampling.")

        for _ in range(n_ens):
            year_idx = self.rng.choice(self.n_years)
            hist_t = self.hist_temps[year_idx]
            hist_r = self.hist_rads[year_idx]

            forecast_len = total_days - T
            hist_t_slice = hist_t[T:total_days]
            hist_r_slice = hist_r[T:total_days]

            if len(hist_t_slice) < forecast_len:
                padding_len = forecast_len - len(hist_t_slice)
                hist_t_slice = np.concatenate([hist_t_slice, np.full(padding_len, hist_t[-1])])
                hist_r_slice = np.concatenate([hist_r_slice, np.full(padding_len, hist_r[-1])])

            t_member = np.concatenate([observed_temps[:T], hist_t_slice])
            r_member = np.concatenate([observed_rads[:T], hist_r_slice])
            ensemble_temps.append(t_member)
            ensemble_rads.append(r_member)

        return np.array(ensemble_temps), np.array(ensemble_rads)

    def generate_stochastic_ar1(
        self, observed_temps, observed_rads, T, total_days, n_ens=50,
        phi_temp=0.7, phi_rad=0.5, sigma_temp=2.0, sigma_rad=1.5
    ):
        """Generates n_ens weather sequences using AR(1) around the climatology.
        If no historical data is provided, climatology is estimated as the mean of observed.
        """
        observed_temps = np.asarray(observed_temps, float)
        observed_rads = np.asarray(observed_rads, float)

        if self.climatology_temp is not None:
            clim_t = self.climatology_temp
            clim_r = self.climatology_rad
        else:
            clim_t = np.full(total_days, np.mean(observed_temps))
            clim_r = np.full(total_days, np.mean(observed_rads))

        if len(clim_t) < total_days:
            clim_t = np.concatenate([clim_t, np.full(total_days - len(clim_t), clim_t[-1])])
            clim_r = np.concatenate([clim_r, np.full(total_days - len(clim_r), clim_r[-1])])

        ensemble_temps = []
        ensemble_rads = []

        for _ in range(n_ens):
            t_seq = np.zeros(total_days)
            r_seq = np.zeros(total_days)

            t_seq[:T] = observed_temps[:T]
            r_seq[:T] = observed_rads[:T]

            last_t = observed_temps[T-1] if T > 0 else clim_t[0]
            last_r = observed_rads[T-1] if T > 0 else clim_r[0]

            for d in range(T, total_days):
                dev_t = phi_temp * (last_t - clim_t[d-1]) + self.rng.normal(0, sigma_temp)
                dev_r = phi_rad * (last_r - clim_r[d-1]) + self.rng.normal(0, sigma_rad)

                last_t = clim_t[d] + dev_t
                last_r = clim_r[d] + dev_r

                last_r = max(1.0, last_r)

                t_seq[d] = last_t
                r_seq[d] = last_r

            ensemble_temps.append(t_seq)
            ensemble_rads.append(r_seq)

        return np.array(ensemble_temps), np.array(ensemble_rads)
