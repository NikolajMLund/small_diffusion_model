import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'plots'))

from forecast.ModelConfig import ModelConfig
from forecast.ForecastConfig import ForecastConfig
from forecast.data_wrangler import load_data
from forecast.core import forecast
from forecast.plotting import plot_forecast_vs_actual
from analysis.scenarios.constant.scenario import ConstantScenario, ConstantScenarioConfig

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'plots')
DATA_PATH = 'processed_data.pkl'

model_config = ModelConfig()
forecast_config = ForecastConfig(
    base_year=2024, 
    target_year=2030
)

data = load_data(DATA_PATH)

Scenario = ConstantScenario(
    data=data,
    model_config=model_config,
    forecast_config=forecast_config,
)
prepared = Scenario.prepare()

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
    output_dir=OUTPUT_DIR,
    file_name='forecast_vs_actual.png',
)
