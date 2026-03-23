import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from forecast.ModelConfig import ModelConfig
from forecast.ForecastConfig import ForecastConfig
from forecast.data_wrangler import load_data
from forecast.core import forecast
from forecast.plotting import plot_forecast_vs_actual
from forecast.scenarios.PhaseInScenario import PhaseInScenario, PhaseInScenarioConfig
from plots import plot_total_sales_forecast, plot_engine_share_over_time

BASE_YEAR = 2025
TARGET_YEAR = 2030
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'plots')
DATA_PATH = 'processed_data.pkl'

model_config = ModelConfig()
forecast_config = ForecastConfig(
    base_year=BASE_YEAR, 
    target_year=TARGET_YEAR
)

data = load_data(DATA_PATH)

# ----------------------------------------------------------------------
#  Scenario configuration
# - Disappearance rates are unchanged from the baseline.
# - Amount of cars sold needs to be determined.
# - BEV/ICEV split of purchases needs to be determined.
# - Age distribution of inflows needs to be determined.
# ------------------------------------------------------------------------

# ------------------------------
# Cars Sold Schedule (total number of cars flowing in each year)
# ------------------------------

breakpoint()
data.keys()
market_shares = data['car_purchases_market_shares']
infer_sales_year = np.arange(
    market_shares.index.get_level_values('year').min(),
    forecast_config.base_year + 1 
)

new_car_sales=market_shares.groupby('year').sum().loc[infer_sales_year]

# regression to extrapolate sales into forecast horizon
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

x = np.arange(-len(new_car_sales), 0) + 1    
x = sm.add_constant(x)

y = new_car_sales.values
model = OLS(y, x)
# add constant to x for intercept
model_fit = model.fit()
sales_forecast = model_fit.predict(x)
print(model_fit.summary())

# Plot raw data, regression fit, and projection into forecast horizon
historical_years = infer_sales_year  # calendar years for observed data
projection_years = np.arange(forecast_config.base_year + 1, forecast_config.target_year + 1)
x_proj = (projection_years - forecast_config.base_year)
x_proj = sm.add_constant(x_proj, has_constant='add')
sales_projection = model_fit.predict(x_proj)

DATA_PLOT_DIR = os.path.join(OUTPUT_DIR, 'data')
SCENARIO_PLOT_DIR = os.path.join(OUTPUT_DIR, 'scenario')

plot_total_sales_forecast(
    historical_years=historical_years,
    y=y,
    sales_forecast=sales_forecast,
    projection_years=projection_years,
    sales_projection=sales_projection,
    base_year=forecast_config.base_year,
    output_dir=DATA_PLOT_DIR,
)

# --------------------------------------------------------------
# BEV/ICEV split of purchases
# --------------------------------------------------------------
plot_engine_share_over_time(
    market_shares=data['car_purchases_market_shares'],
    new_reg_market_shares=data['market_shares'],
    base_year=forecast_config.base_year,
    output_dir=DATA_PLOT_DIR,
)



# --------------------------------------------------------------
# Age distribution of inflows
# --------------------------------------------------------------




# ---------------------------------------------------------------
# Packing scenario config
# ---------------------------------------------------------------

scenario_config = PhaseInScenarioConfig(
    expected_sales = dict(zip(projection_years, sales_projection)),
    bev_share_schedule = {
        2023: np.array([0.3, 0.7]),   # TODO: Should develop according to some schedule. 
    },
    age_dist_schedule = {
        2023: np.array([0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1]),   # TODO: Should we assume the age distribution of inflows remains constant, or changes over time (e.g., due to changing import patterns)?
    }
)

Scenario = PhaseInScenario(
    data=data,
    model_config=model_config,
    forecast_config=forecast_config,
    scenario_config=scenario_config,
)

prepared = Scenario.prepare()
breakpoint()
forecasted_distributions = forecast(
    state=prepared['state'],
    dis_rates=prepared['dis_rates'],
    purchase_inflows=prepared['purchase_inflows'],
    model_config=model_config,
    forecast_config=forecast_config,
)

plot_forecast_vs_actual(
    forecasted_distributions=forecasted_distributions,
    holdings_dist=data['holdings_dist'],
    model_config=model_config,
    forecast_config=forecast_config,
    output_dir=SCENARIO_PLOT_DIR,
    file_name='forecast_vs_actual_phase_in_test.png',
)
