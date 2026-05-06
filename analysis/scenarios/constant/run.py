import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'plots'))

from forecast.ModelConfig import ModelConfig
from forecast.ForecastConfig import ForecastConfig
from forecast.data_wrangler import load_data
from forecast.core import forecast
from forecast.plotting import plot_forecast_vs_actual
from forecast.scenarios.ConstantScenario import ConstantScenario, ConstantScenarioConfig

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'plots')
DATA_PATH = 'processed_data.pkl'

model_config = ModelConfig()
forecast_config = ForecastConfig(
    base_year=2023, 
    target_year=2100 #2035
)

data = load_data(DATA_PATH)

Scenario = ConstantScenario(
    data=data,
    model_config=model_config,
    forecast_config=forecast_config,
    scenario_config=ConstantScenarioConfig(),
)
prepared = Scenario.prepare()

forecasted_distributions = forecast(
    state=prepared['state'],
    dis_rates=prepared['dis_rates'],
    purchase_inflows=prepared['projected_inflows'],
    model_config=model_config,
    forecast_config=forecast_config,
)

plot_forecast_vs_actual(
    forecasted_distributions=forecasted_distributions,
    holdings_dist=data['holdings_dist'],
    model_config=model_config,
    forecast_config=forecast_config,
    output_dir=OUTPUT_DIR,
    file_name='forecast_vs_actual.png',
)

# ---------------------------------------------------------------
# Diagnostics: age-specific holdings by year and engine type
# ---------------------------------------------------------------

# Historic (from observed data) — MultiIndex (year, engine_type, car_age)
_name = data['holdings_dist'].name
hist = (
    data['holdings_dist']
    .reset_index()
    .rename(columns={_name: 'value'} if _name else {0: 'value'})
)
hist['source'] = 'historic'

# Predicted (from forecast) — unpack 3-D array (n_years, n_types, n_ages)
_projection_years = Scenario.projection_years
_n_ages_pred = forecasted_distributions.shape[2]
pred_rows = [
    {
        'year':        int(year),
        'engine_type': model_config.engine_types[i],
        'car_age':     age,
        'value':       forecasted_distributions[t, i, age],
    }
    for t, year in enumerate(_projection_years)
    for i in range(len(model_config.engine_types))
    for age in range(_n_ages_pred)
]
pred = pd.DataFrame(pred_rows)
pred['source'] = 'predicted'

pred_years = pred.year.unique()
hist = hist[~hist['year'].isin(pred_years)]


holdings_diagnostics = pd.concat([hist, pred], ignore_index=True)
holdings_diagnostics = holdings_diagnostics[['year', 'engine_type', 'car_age', 'value', 'source']]

holdings_diagnostics=holdings_diagnostics.groupby(['year','engine_type','source']).sum()

# scale to by stock in baseline year. 
scale_stock = holdings_diagnostics.loc[forecast_config.base_year].sum()
holdings_diagnostics /= scale_stock

holdings_diagnostics=holdings_diagnostics.reset_index()[['year','engine_type','value']].pivot(index='year', columns=['engine_type'])
holdings_diagnostics['source'] = 'historic'
holdings_diagnostics.loc[holdings_diagnostics.index.get_level_values('year') > forecast_config.base_year, 'source'] = 'predicted'

holdings_diagnostics.to_excel(
    os.path.join(OUTPUT_DIR, 'holdings_diagnostics.xlsx'),
)
