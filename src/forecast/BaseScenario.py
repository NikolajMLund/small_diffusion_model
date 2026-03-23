"""
Abstract base class for scenario wranglers.

Each scenario subclass implements get_dis_rates and get_purchase_inflows using
whatever logic that scenario requires. The base class handles the invariant
parts (initial state) and provides protected helpers for the common baseline
computations that scenarios can reuse.
"""

from abc import ABC, abstractmethod

import numpy as np
from pandas import IndexSlice as idx

from forecast.ModelConfig import ModelConfig
from forecast.ForecastConfig import ForecastConfig

class BaseScenario(ABC):

    def __init__(
        self,
        data: dict,
        model_config: ModelConfig,
        forecast_config: ForecastConfig,
        scenario_config
    ) -> None:
        self.data = data
        self.model_config = model_config
        self.forecast_config = forecast_config
        self.scenario_config = scenario_config
        self.base_year = forecast_config.base_year
        self.target_year = forecast_config.target_year
        self.n_forecast_years = self.target_year - self.base_year
        self.car_types = model_config.engine_types
        self.n_ages = model_config.max_car_age + 2

    # ------------------------------------------------------------------
    # Concrete: initial state is always historical holdings at base_year
    # ------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        """Returns shape (n_car_types, n_ages)."""
        holdings_dist = self.data['holdings_dist']
        state = np.full((len(self.car_types), self.n_ages), np.nan)
        for i, car_type in enumerate(self.car_types):
            state[i, :] = holdings_dist.loc[self.base_year, car_type, :].values
        return state

    # ------------------------------------------------------------------
    # Protected helpers — reusable baselines for subclass methods
    # ------------------------------------------------------------------

    def _baseline_dis_rates(self) -> np.ndarray:
        """
        Returns shape (n_forecast_years, n_car_types, n_ages).
        Tiles the fitted scrap_profile across all years and car types.
        """
        ages = np.arange(self.n_ages)
        baseline = self.data['scrap_profile'].reindex(ages, fill_value=0).values
        dis_rates = np.full(
            (self.n_forecast_years, len(self.car_types), self.n_ages), np.nan
        )
        for i in range(len(self.car_types)):
            for t in range(self.n_forecast_years):
                dis_rates[t, i, :] = baseline
        return dis_rates

    def _baseline_purchase_inflows(self) -> np.ndarray:
        """
        Returns shape (n_forecast_years, n_car_types, max_purchase_age+1).
        Tiles base_year purchase shares across all forecast years.
        """
        max_purchase_age = self.model_config.purchase_age_limit
        purchase_ages = np.arange(max_purchase_age + 1)
        purchase_inflows = np.full(
            (self.n_forecast_years, len(self.car_types), max_purchase_age + 1), np.nan
        )
        for i, car_type in enumerate(self.car_types):
            base = (
                self.data['car_purchases_market_shares']
                .loc[idx[self.base_year, car_type, :]]
                .reindex(purchase_ages, fill_value=0)
                .values
            )
            for t in range(self.n_forecast_years):
                purchase_inflows[t, i, :] = base
        return purchase_inflows

    # ------------------------------------------------------------------
    # Abstract: each scenario subclass defines these
    # ------------------------------------------------------------------

    @abstractmethod
    def get_dis_rates(self, config) -> np.ndarray:
        """Returns shape (n_forecast_years, n_car_types, n_ages)."""
        ...

    @abstractmethod
    def get_purchase_inflows(self, config) -> np.ndarray:
        """Returns shape (n_forecast_years, n_car_types, max_purchase_age+1)."""
        ...

    # ------------------------------------------------------------------
    # Convenience: assembles the dict expected by core.forecast()
    # ------------------------------------------------------------------

    def prepare(self) -> dict:
        return {
            'state': self.get_state(),
            'dis_rates': self.get_dis_rates(self.scenario_config),
            'purchase_inflows': self.get_purchase_inflows(self.scenario_config),
        }
