import sys
import os

import numpy as np
from pandas import IndexSlice as idx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from forecast.ModelConfig import ModelConfig
from forecast.ForecastConfig import ForecastConfig
from forecast.data_wrangler import load_data
from forecast.scenarios.ConstantScenario import ConstantScenario, ConstantScenarioConfig
from forecast.scenarios.PhaseInScenario import PhaseInScenario, PhaseInScenarioConfig
from forecast.core import forecast

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'processed_data.pkl')


def test_phase_in_replicates_constant():
    data = load_data(DATA_PATH)
    model_config = ModelConfig()
    forecast_config = ForecastConfig(base_year=2023, target_year=2024)

    # ConstantScenario baseline
    constant = ConstantScenario(data, model_config, forecast_config, ConstantScenarioConfig())
    expected = constant.projected_inflows  # (n_years, n_engine, n_ages)

    # Build equivalent PhaseInScenarioConfig
    base_purchases = data['car_purchases_market_shares']
    total = base_purchases.loc[forecast_config.base_year].sum()
    n_years = forecast_config.target_year - forecast_config.base_year

    projected_sales = np.full(n_years, total)

    market_shares = np.full(
        (n_years, len(model_config.engine_types), model_config.purchase_age_limit + 1),
        np.nan
    )
    for i, engine in enumerate(model_config.engine_types):
        shares = (
            base_purchases.loc[idx[forecast_config.base_year, engine, :]]
            .reindex(np.arange(model_config.purchase_age_limit + 1), fill_value=0)
            .values / total
        )
        market_shares[:, i, :] = shares[np.newaxis, :]

    config = PhaseInScenarioConfig(projected_sales=projected_sales, market_shares=market_shares)
    phase_in = PhaseInScenario(data, model_config, forecast_config, config)

    constant_data = constant.prepare()
    phase_in_data = phase_in.prepare()
    
    # Forecasting
    forecasted_constant = forecast(
        state=constant_data['state'],
        dis_rates=constant_data['dis_rates'],
        purchase_inflows=constant_data['projected_inflows'],
        model_config=model_config,
        forecast_config=forecast_config,
    )

    forecasted_phase_in = forecast(
        state=phase_in_data['state'],
        dis_rates=phase_in_data['dis_rates'],
        purchase_inflows=phase_in_data['projected_inflows'],
        model_config=model_config,
        forecast_config=forecast_config,
    )

    # 
    
    breakpoint()
    assert np.allclose(phase_in.projected_inflows, expected)
    assert np.allclose(forecasted_phase_in, forecasted_constant) 
