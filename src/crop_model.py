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
        self.t_sum_maturity = t_sum_maturity # GDD to maturity
        self.rue = rue                       # radiation-use efficiency g/MJ
        self.sla = sla                       # specific leaf area ha/kg
        self.k_ext = k_ext                   # canopy extinction coefficient
        self.harvest_index = harvest_index
        self.heat_threshold = heat_threshold # terminal heat-stress (C)

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
