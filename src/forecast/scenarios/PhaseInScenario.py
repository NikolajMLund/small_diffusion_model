"""
Phase-in scenario: ramp BEV share of purchases over the forecast horizon.

Disappearance rates are unchanged from the baseline.
Purchase inflows are rescaled so that BEV/ICEV totals follow bev_share_schedule,
preserving the within-type age distribution shape from the base year.
"""

from dataclasses import dataclass

import numpy as np

from forecast.BaseScenario import BaseScenario

@dataclass
class PhaseInScenarioConfig:
    """
    bev_share_schedule : array of length n_forecast_years
        BEV fraction of total purchases in each forecast year. Values in [0, 1].
    """
    expected_sales : dict[int, float]       #  (n_forecast_years,) - total number of cars flowing in each year
    bev_share_schedule : dict[int, np.ndarray]   #  (n_forecast_years,) - BE
    age_dist_schedule : dict[int, np.ndarray]    #  (n_forecast_years,) - age distribution of inflows in each year


class PhaseInScenario(BaseScenario):

    def phase_in_inflows_schedule(
            self, 
            PhaseInScenarioConfig: PhaseInScenarioConfig,
    ) -> np.ndarray:
        """
        """
        # Setting objects
        total_inflows = np.zeros(self.n_forecast_years) + np.nan   
        engine_shares = np.zeros((self.n_forecast_years, len(self.car_types))) + np.nan  
        age_dist = np.zeros((self.n_forecast_years, self.model_config.purchase_age_limit + 1)) + np.nan        

        # -------------------------------------------
        #   Car inflow schedule (total number of cars flowing in each year)
        # -------------------------------------------
        purchase_inflows = np.full(
            (self.n_forecast_years, len(self.car_types), self.model_config.purchase_age_limit + 1), 
            np.nan
        )

        for i, t in enumerate(np.arange(self.base_year, self.forecast_config.target_year)):
            total_inflows[i] = PhaseInScenarioConfig.expected_sales[t]
            engine_shares[i, :] = PhaseInScenarioConfig.bev_share_schedule[t]
            age_dist[i, :] = PhaseInScenarioConfig.age_dist_schedule[t]

            purchase_inflows[i, :, :] = total_inflows[i] * engine_shares[i, :][:, np.newaxis] * age_dist[i, :][np.newaxis, :]
        
        breakpoint()

    def get_dis_rates(self, config: PhaseInScenarioConfig) -> np.ndarray:
        return self._baseline_dis_rates()

    def get_purchase_inflows(self, config: PhaseInScenarioConfig) -> np.ndarray:
        return self.phase_in_inflows_schedule(config)

