"""
Phase-in scenario: ramp BEV share of purchases over the forecast horizon.

Disappearance rates are unchanged from the baseline.
Purchase inflows are rescaled so that BEV/ICEV totals follow bev_share_schedule,
preserving the within-type age distribution shape from the base year.
"""

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
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

    def __init__(self, data, model_config, forecast_config, scenario_config):
        super().__init__(data, model_config, forecast_config, scenario_config)
        self.projected_inflows = self.get_projected_inflows(scenario_config)
        self.dis_rates = self.get_dis_rates(scenario_config)

    def _compute_projected_inflows(
            self,
            PhaseInScenarioConfig: PhaseInScenarioConfig,
    ) -> np.ndarray:
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

    def get_projected_inflows(self, config: PhaseInScenarioConfig) -> np.ndarray:
        return self._compute_projected_inflows(config)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_projected_total_inflow(self, output_dir: str | None = None):
        fig, ax = plt.subplots()
        ax.plot(self.projection_years, self.scenario_config.projected_sales, marker='o')
        ax.axvline(self.base_year, color='grey', linestyle=':', linewidth=0.8)
        ax.set_xlabel('Year')
        ax.set_ylabel('Total cars flowing in')
        ax.set_title('Projected total inflow')

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, 'projected_total_inflow.png'), dpi=150, bbox_inches='tight')
        plt.show()
        return fig

    def plot_projected_used_car_sales(self, output_dir: str | None = None):
        market_shares = self.scenario_config.market_shares
        engine_types = self.model_config.engine_types
        cmap = plt.get_cmap('tab10')

        fig, axes = plt.subplots(len(engine_types), 1, figsize=(9, 7), sharex=True)
        if len(engine_types) == 1:
            axes = [axes]

        for ax, (i, engine) in zip(axes, enumerate(engine_types)):
            imports = market_shares[:, i, 1:]  # ages 1+
            ages = np.arange(1, imports.shape[1] + 1)
            bottom = np.zeros(len(self.projection_years))
            for j, age in enumerate(ages):
                ax.bar(self.projection_years, imports[:, j], bottom=bottom,
                       color=cmap(age), label=f'Age {age}')
                bottom += imports[:, j]
            ax.set_title(engine)
            ax.set_ylabel('Share of total sales')

        axes[-1].set_xlabel('Year')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, 0))
        fig.suptitle('Projected used car sales (age ≥ 1) as share of total inflow')
        fig.tight_layout(rect=(0, 0.06, 1, 1))

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, 'projected_used_car_sales.png'), dpi=150, bbox_inches='tight')
        plt.show()
        return fig

    def plot_projected_new_registrations(self, output_dir: str | None = None):
        market_shares = self.scenario_config.market_shares
        engine_types = self.model_config.engine_types
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

        fig, ax = plt.subplots()
        width = 0.8 / len(engine_types)
        offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, len(engine_types))

        for i, engine in enumerate(engine_types):
            new_reg = market_shares[:, i, 0]
            ax.bar(self.projection_years + offsets[i], new_reg, width=width,
                   color=colors[i % len(colors)], label=engine)

        ax.set_xlabel('Year')
        ax.set_ylabel('Share of total sales')
        ax.set_title('Projected new registrations (age 0) as share of total inflow')
        ax.legend()

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, 'projected_new_registrations.png'), dpi=150, bbox_inches='tight')
        plt.show()
        return fig

    def plot_projected_inflows(self, output_dir: str | None = None):
        engine_types = self.model_config.engine_types
        n_ages = self.projected_inflows.shape[2]
        cmap = plt.get_cmap('tab10')

        fig, axes = plt.subplots(len(engine_types), 1, figsize=(9, 7), sharex=True)
        if len(engine_types) == 1:
            axes = [axes]

        for ax, (i, engine) in zip(axes, enumerate(engine_types)):
            bottom = np.zeros(len(self.projection_years))
            for age in range(n_ages):
                ax.bar(self.projection_years, self.projected_inflows[:, i, age], bottom=bottom,
                       color=cmap(age), label=f'Age {age}')
                bottom += self.projected_inflows[:, i, age]
            ax.set_title(engine)
            ax.set_ylabel('Cars flowing in')

        axes[-1].set_xlabel('Year')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, 0))
        fig.suptitle('Projected inflows by engine type and purchase age (absolute counts)')
        fig.tight_layout(rect=(0, 0.06, 1, 1))

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, 'projected_inflows.png'), dpi=150, bbox_inches='tight')
        plt.show()
        return fig

    def plot_inflow_by_engine_type(self, output_dir: str | None = None):
        inflow_by_engine = self.projected_inflows.sum(axis=2)  # (n_years, 2)
        engine_types = self.model_config.engine_types
        colors = {'ICEV': 'tab:orange', 'BEV': 'tab:blue'}

        fig, ax = plt.subplots()
        bottom = np.zeros(len(self.projection_years))
        for i, engine in enumerate(engine_types):
            ax.bar(self.projection_years, inflow_by_engine[:, i], bottom=bottom,
                   color=colors.get(engine, f'C{i}'), label=engine)
            bottom += inflow_by_engine[:, i]

        ax.set_xlabel('Year')
        ax.set_ylabel('Cars flowing in')
        ax.set_title('Projected total inflow by engine type')
        ax.legend()

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, 'inflow_by_engine_type.png'), dpi=150, bbox_inches='tight')
        plt.show()
        return fig

    def plot_fleet_composition(self, forecasted_distributions: np.ndarray, output_dir: str | None = None):
        fleet = forecasted_distributions.sum(axis=2)  # (n_years, 2)
        fleet_pct = fleet / fleet.sum(axis=1, keepdims=True) * 100
        engine_types = self.model_config.engine_types
        colors = {'ICEV': 'tab:orange', 'BEV': 'tab:blue'}

        fig, ax = plt.subplots()
        bottom = np.zeros(len(self.projection_years))
        for i, engine in enumerate(engine_types):
            ax.bar(self.projection_years, fleet_pct[:, i], bottom=bottom,
                   color=colors.get(engine, f'C{i}'), label=engine)
            bottom += fleet_pct[:, i]

        ax.set_xlabel('Year')
        ax.set_ylabel('Share of fleet (%)')
        ax.set_ylim(0, 100)
        ax.set_title('Fleet composition by engine type')
        ax.legend()

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, 'fleet_composition.png'), dpi=150, bbox_inches='tight')
        plt.show()
        return fig

    def plot_all(self, output_dir: str | None = None, forecasted_distributions: np.ndarray | None = None) -> None:
        self.plot_projected_total_inflow(output_dir=output_dir)
        self.plot_projected_used_car_sales(output_dir=output_dir)
        self.plot_projected_new_registrations(output_dir=output_dir)
        self.plot_projected_inflows(output_dir=output_dir)
        self.plot_inflow_by_engine_type(output_dir=output_dir)
        if forecasted_distributions is not None:
            self.plot_fleet_composition(forecasted_distributions, output_dir=output_dir)

