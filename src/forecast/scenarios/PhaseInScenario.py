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
    projected_sales : np.ndarray       #  (n_forecast_years,) - total number of cars flowing in each year
    market_shares: np.ndarray          # (n_forecast_years, len(engine_types), max_purchase_age + 1)

class PhaseInScenario(BaseScenario):

    def phase_in_inflows_schedule(
            self, 
            PhaseInScenarioConfig: PhaseInScenarioConfig,
    ) -> np.ndarray:
        """
        """
        # -------------------------------------------
        #   Unpacking and validating dimensions
        # -------------------------------------------
        market_shares=PhaseInScenarioConfig.market_shares
        projected_sales=PhaseInScenarioConfig.projected_sales

        expected_market_shares_shape = (
            self.n_forecast_years,
            len(self.model_config.engine_types),
            self.model_config.purchase_age_limit + 1,
        )
        assert market_shares.shape == expected_market_shares_shape, (
            f"market_shares.shape is {market_shares.shape}, expected {expected_market_shares_shape}"
        )
        assert projected_sales.shape == (self.n_forecast_years,), (
            f"projected_sales.shape is {projected_sales.shape}, expected ({self.n_forecast_years},)"
        )

        # -------------------------------------------
        #   Car inflow schedule (total number of cars flowing in each year)
        # -------------------------------------------
        projected_inflows = np.full(
            (self.n_forecast_years, len(self.car_types), self.model_config.purchase_age_limit + 1), 
            np.nan
        )

        projected_inflows[...] = market_shares * projected_sales[:, np.newaxis, np.newaxis]
        
        return projected_inflows
    
    def get_dis_rates(self, config: PhaseInScenarioConfig) -> np.ndarray:
        return self._baseline_dis_rates()

    def get_purchase_inflows(self, config: PhaseInScenarioConfig) -> np.ndarray:
        return self.phase_in_inflows_schedule(config)

