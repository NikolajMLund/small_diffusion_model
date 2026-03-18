"""
Phase-in scenario: ramp BEV share of purchases over the forecast horizon.

Disappearance rates are unchanged from the baseline.
Purchase inflows are rescaled so that BEV/ICEV totals follow bev_share_schedule,
preserving the within-type age distribution shape from the base year.
"""

from dataclasses import dataclass

import numpy as np

from forecast.base_wrangler import BaseWrangler
from forecast.wrangler_helpers import rescale_bev_icev_split


@dataclass
class PhaseInScenarioConfig:
    """
    bev_share_schedule : array of length n_forecast_years
        BEV fraction of total purchases in each forecast year. Values in [0, 1].
    """
    bev_share_schedule: np.ndarray


class PhaseInScenario(BaseWrangler):

    def get_dis_rates(self, config: PhaseInScenarioConfig) -> np.ndarray:
        return self._baseline_dis_rates()

    def get_purchase_inflows(self, config: PhaseInScenarioConfig) -> np.ndarray:
        return rescale_bev_icev_split(
            baseline_inflows=self._baseline_purchase_inflows(),
            bev_share_schedule=config.bev_share_schedule,
            bev_idx=self.car_types.index('BEV'),
            icev_idx=self.car_types.index('ICEV'),
        )
