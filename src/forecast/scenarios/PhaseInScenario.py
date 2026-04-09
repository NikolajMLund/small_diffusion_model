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

    alpha_hist: float = 0.4  # opacity for historical (≤ base_year) data in plots

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

    def get_state(self, config: PhaseInScenarioConfig):
        return self._baseline_get_state()

    def get_dis_rates(self, config: PhaseInScenarioConfig) -> np.ndarray:
        return self._baseline_dis_rates()

    def get_projected_inflows(self, config: PhaseInScenarioConfig) -> np.ndarray:
        return self._compute_projected_inflows(config)

    # ------------------------------------------------------------------
    # Historical data helpers
    # ------------------------------------------------------------------

    def _hist_inflows(self):
        """
        Returns car_purchases_market_shares for years <= base_year.
        Multi-index: (year, engine_type, car_age).  Values are absolute counts.
        """
        ms = self.data['car_purchases_market_shares']
        mask = ms.index.get_level_values('year') <= self.base_year
        return ms[mask]

    def _hist_fleet(self):
        """
        Returns holdings_dist summed over car ages for years <= base_year.
        Result indexed by (year, engine_type).  Values are absolute fleet counts.
        """
        hd = self.data['holdings_dist']
        mask = hd.index.get_level_values('year') <= self.base_year
        return hd[mask].groupby(['year', 'engine_type']).sum()

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_projected_total_inflow(self, output_dir: str | None = None):
        fig, ax = plt.subplots()

        # historical
        hist = self._hist_inflows()
        hist_total = hist.groupby('year').sum()
        ax.plot(hist_total.index, hist_total.values, marker='o', color='tab:blue', alpha=self.alpha_hist)

        # projection
        ax.plot(self.projection_years, self.scenario_config.projected_sales, marker='o')
        ax.axvline(self.base_year, color='grey', linestyle=':', linewidth=0.8)
        ax.set_xlabel('Year')
        ax.set_ylabel('Total cars flowing in')
        ax.set_title('Projected total inflow')

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, 'projected_total_inflow.png'), dpi=150, bbox_inches='tight')
        plt.close()
        return fig

    def plot_projected_used_car_sales(self, output_dir: str | None = None):
        market_shares = self.scenario_config.market_shares
        engine_types = self.model_config.engine_types
        cmap = plt.get_cmap('tab10')

        fig, axes = plt.subplots(len(engine_types), 1, figsize=(9, 7), sharex=True)
        if len(engine_types) == 1:
            axes = [axes]

        # historical
        hist = self._hist_inflows()
        hist_total_by_year = hist.groupby('year').sum()
        hist_years = hist_total_by_year.index
        hist_ages1plus = hist[hist.index.get_level_values('car_age') >= 1]

        for ax, (i, engine) in zip(axes, enumerate(engine_types)):
            imports = market_shares[:, i, 1:]  # ages 1+
            ages = np.arange(1, imports.shape[1] + 1)

            # historical bars
            bottom_h = np.zeros(len(hist_years))
            eng_hist = hist_ages1plus.xs(engine, level='engine_type')
            for j, age in enumerate(ages):
                try:
                    vals_h = (eng_hist.xs(age, level='car_age')
                                      .reindex(hist_years, fill_value=0)
                                      .values / hist_total_by_year.values)
                except KeyError:
                    vals_h = np.zeros(len(hist_years))
                ax.bar(hist_years, vals_h, bottom=bottom_h, color=cmap(age), alpha=self.alpha_hist)
                bottom_h += vals_h

            # projection bars
            bottom = np.zeros(len(self.projection_years))
            for j, age in enumerate(ages):
                ax.bar(self.projection_years, imports[:, j], bottom=bottom,
                       color=cmap(age), label=f'Age {age}')
                bottom += imports[:, j]

            ax.axvline(self.base_year, color='grey', linestyle=':', linewidth=0.8)
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
        plt.close()
        return fig

    def plot_projected_new_registrations(self, output_dir: str | None = None):
        market_shares = self.scenario_config.market_shares
        engine_types = self.model_config.engine_types
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

        fig, ax = plt.subplots()
        width = 0.8 / len(engine_types)
        offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, len(engine_types))

        # historical
        hist = self._hist_inflows()
        hist_total_by_year = hist.groupby('year').sum()
        hist_years = hist_total_by_year.index
        hist_age0 = hist[hist.index.get_level_values('car_age') == 0]

        for i, engine in enumerate(engine_types):
            # historical bars
            try:
                vals_h = (hist_age0.xs(engine, level='engine_type')
                                   .droplevel('car_age')
                                   .reindex(hist_years, fill_value=0)
                                   .values / hist_total_by_year.values)
            except KeyError:
                vals_h = np.zeros(len(hist_years))
            ax.bar(hist_years + offsets[i], vals_h, width=width,
                   color=colors[i % len(colors)], alpha=self.alpha_hist)

            # projection bars
            new_reg = market_shares[:, i, 0]
            ax.bar(self.projection_years + offsets[i], new_reg, width=width,
                   color=colors[i % len(colors)], label=engine)

        ax.axvline(self.base_year, color='grey', linestyle=':', linewidth=0.8)
        ax.set_xlabel('Year')
        ax.set_ylabel('Share of total sales')
        ax.set_title('Projected new registrations (age 0) as share of total inflow')
        ax.legend()

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, 'projected_new_registrations.png'), dpi=150, bbox_inches='tight')
        plt.close()
        return fig

    def plot_projected_inflows(self, output_dir: str | None = None):
        engine_types = self.model_config.engine_types
        n_ages = self.projected_inflows.shape[2]
        cmap = plt.get_cmap('tab10')

        fig, axes = plt.subplots(len(engine_types), 1, figsize=(9, 7), sharex=True)
        if len(engine_types) == 1:
            axes = [axes]

        # historical
        hist = self._hist_inflows()
        hist_years = hist.index.get_level_values('year').unique().sort_values()

        for ax, (i, engine) in zip(axes, enumerate(engine_types)):
            # historical bars
            eng_hist = hist.xs(engine, level='engine_type')
            bottom_h = np.zeros(len(hist_years))
            for age in range(n_ages):
                try:
                    vals_h = (eng_hist.xs(age, level='car_age')
                                      .reindex(hist_years, fill_value=0)
                                      .values)
                except KeyError:
                    vals_h = np.zeros(len(hist_years))
                ax.bar(hist_years, vals_h, bottom=bottom_h, color=cmap(age), alpha=self.alpha_hist)
                bottom_h += vals_h

            # projection bars
            bottom = np.zeros(len(self.projection_years))
            for age in range(n_ages):
                ax.bar(self.projection_years, self.projected_inflows[:, i, age], bottom=bottom,
                       color=cmap(age), label=f'Age {age}')
                bottom += self.projected_inflows[:, i, age]

            ax.axvline(self.base_year, color='grey', linestyle=':', linewidth=0.8)
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
        plt.close()
        return fig

    def plot_inflow_by_engine_type(self, output_dir: str | None = None):
        inflow_by_engine = self.projected_inflows.sum(axis=2)  # (n_years, 2)
        engine_types = self.model_config.engine_types
        colors = {'ICEV': 'tab:orange', 'BEV': 'tab:blue'}

        fig, ax = plt.subplots()

        # historical
        hist = self._hist_inflows()
        hist_by_eng = hist.groupby(['year', 'engine_type']).sum().unstack('engine_type')
        hist_years = hist_by_eng.index
        bottom_h = np.zeros(len(hist_years))
        for i, engine in reversed(list(enumerate(engine_types))):
            vals_h = hist_by_eng[engine].values if engine in hist_by_eng.columns else np.zeros(len(hist_years))
            ax.bar(hist_years, vals_h, bottom=bottom_h,
                   color=colors.get(engine, f'C{i}'), alpha=self.alpha_hist)
            bottom_h += vals_h

        # projection
        bottom = np.zeros(len(self.projection_years))
        for i, engine in reversed(list(enumerate(engine_types))):
            ax.bar(self.projection_years, inflow_by_engine[:, i], bottom=bottom,
                   color=colors.get(engine, f'C{i}'), label=engine)
            bottom += inflow_by_engine[:, i]

        ax.axvline(self.base_year, color='grey', linestyle=':', linewidth=0.8)
        ax.set_xlabel('Year')
        ax.set_ylabel('Cars flowing in')
        ax.set_title('Projected total inflow by engine type')
        ax.legend()

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, 'inflow_by_engine_type.png'), dpi=150, bbox_inches='tight')
        plt.close()
        return fig

    def plot_fleet_composition(self, forecasted_distributions: np.ndarray, output_dir: str | None = None):
        fleet = forecasted_distributions.sum(axis=2)  # (n_years, 2)
        fleet_pct = fleet / fleet.sum(axis=1, keepdims=True) * 100
        engine_types = self.model_config.engine_types
        colors = {'ICEV': 'tab:orange', 'BEV': 'tab:blue'}

        fig, ax = plt.subplots()

        # historical
        hist = self._hist_fleet()
        hist_pct = hist.unstack('engine_type')
        hist_pct = hist_pct.div(hist_pct.sum(axis=1), axis=0) * 100
        hist_years = hist_pct.index
        bottom_h = np.zeros(len(hist_years))
        for i, engine in reversed(list(enumerate(engine_types))):
            vals_h = hist_pct[engine].values if engine in hist_pct.columns else np.zeros(len(hist_years))
            ax.bar(hist_years, vals_h, bottom=bottom_h,
                   color=colors.get(engine, f'C{i}'), alpha=self.alpha_hist)
            bottom_h += vals_h

        # projection
        bottom = np.zeros(len(self.projection_years))
        for i, engine in reversed(list(enumerate(engine_types))):
            ax.bar(self.projection_years, fleet_pct[:, i], bottom=bottom,
                   color=colors.get(engine, f'C{i}'), label=engine)
            bottom += fleet_pct[:, i]

        ax.axvline(self.base_year, color='grey', linestyle=':', linewidth=0.8)
        ax.set_xlabel('Year')
        ax.set_ylabel('Share of fleet (%)')
        ax.set_ylim(0, 100)
        ax.set_title('Fleet composition by engine type')
        ax.legend()

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, 'fleet_composition.png'), dpi=150, bbox_inches='tight')
        plt.close()
        return fig

    def plot_total_fleet_stock(self, forecasted_distributions: np.ndarray, output_dir: str | None = None):
        fleet = forecasted_distributions.sum(axis=2)  # (n_years, 2)
        engine_types = self.model_config.engine_types
        colors = {'ICEV': 'tab:orange', 'BEV': 'tab:blue'}

        fig, ax = plt.subplots()

        # historical
        hist = self._hist_fleet()
        hist_abs = hist.unstack('engine_type')
        hist_years = hist_abs.index
        bottom_h = np.zeros(len(hist_years))
        for i, engine in reversed(list(enumerate(engine_types))):
            vals_h = hist_abs[engine].values if engine in hist_abs.columns else np.zeros(len(hist_years))
            ax.bar(hist_years, vals_h, bottom=bottom_h,
                   color=colors.get(engine, f'C{i}'), alpha=self.alpha_hist)
            bottom_h += vals_h

        # projection
        bottom = np.zeros(len(self.projection_years))
        for i, engine in reversed(list(enumerate(engine_types))):
            ax.bar(self.projection_years, fleet[:, i], bottom=bottom,
                   color=colors.get(engine, f'C{i}'), label=engine)
            bottom += fleet[:, i]

        ax.axvline(self.base_year, color='grey', linestyle=':', linewidth=0.8)
        ax.set_xlabel('Year')
        ax.set_ylabel('Total fleet stock (cars)')
        ax.set_title('Total fleet stock by engine type')
        ax.legend()

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, 'total_fleet_stock.png'), dpi=150, bbox_inches='tight')
        plt.close()
        return fig

    def plot_engine_share_of_total_inflow(self, output_dir: str | None = None):
        market_shares = self.scenario_config.market_shares
        engine_types = self.model_config.engine_types
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

        fig, ax = plt.subplots()
        width = 0.8 / len(engine_types)
        offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, len(engine_types))

        # historical
        hist = self._hist_inflows()
        hist_total_by_year = hist.groupby('year').sum()
        hist_years = hist_total_by_year.index

        for i, engine in enumerate(engine_types):
            try:
                vals_h = (hist.xs(engine, level='engine_type')
                              .groupby('year').sum()
                              .reindex(hist_years, fill_value=0)
                              .values / hist_total_by_year.values)
            except KeyError:
                vals_h = np.zeros(len(hist_years))
            ax.bar(hist_years + offsets[i], vals_h, width=width,
                   color=colors[i % len(colors)], alpha=self.alpha_hist)

            # projection: sum over all ages for this engine
            proj_share = market_shares[:, i, :].sum(axis=1)
            ax.bar(self.projection_years + offsets[i], proj_share, width=width,
                   color=colors[i % len(colors)], label=engine)

        ax.axvline(self.base_year, color='grey', linestyle=':', linewidth=0.8)
        ax.set_xlabel('Year')
        ax.set_ylabel('Share of total inflow')
        ax.set_title('Engine type share of total inflow (all ages)')
        ax.legend()

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, 'engine_share_of_total_inflow.png'),
                        dpi=150, bbox_inches='tight')
        plt.close()
        return fig

    def plot_fleet_age_distribution(self, forecasted_distributions: np.ndarray, output_dir: str | None = None):
        ages = np.arange(self.model_config.max_car_age + 2)
        engine_types = self.model_config.engine_types
        colors = {'ICEV': 'tab:orange', 'BEV': 'tab:blue'}
        target_year = self.forecast_config.target_year

        fig, ax = plt.subplots()
        # Plot BEV first (bottom), then ICEV (top)
        ordered = sorted(enumerate(engine_types), key=lambda x: (x[1] != 'BEV'))
        bottom = np.zeros(len(ages))
        for i, engine in ordered:
            vals = forecasted_distributions[-1, i, :]
            ax.bar(ages, vals, bottom=bottom, color=colors.get(engine, f'C{i}'), label=engine)
            bottom += vals

        ax.set_xlabel('Car age')
        ax.set_ylabel('Number of cars')
        ax.set_title(f'Fleet age distribution ({target_year})')
        ax.legend()

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, 'fleet_age_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()
        return fig

    def plot_all(self, output_dir: str | None = None, forecasted_distributions: np.ndarray | None = None) -> None:
        self.plot_projected_total_inflow(output_dir=output_dir)
        self.plot_projected_used_car_sales(output_dir=output_dir)
        self.plot_projected_new_registrations(output_dir=output_dir)
        self.plot_engine_share_of_total_inflow(output_dir=output_dir)
        self.plot_projected_inflows(output_dir=output_dir)
        self.plot_inflow_by_engine_type(output_dir=output_dir)
        if forecasted_distributions is not None:
            self.plot_fleet_composition(forecasted_distributions, output_dir=output_dir)
            self.plot_total_fleet_stock(forecasted_distributions, output_dir=output_dir)
            self.plot_fleet_age_distribution(forecasted_distributions, output_dir=output_dir)
