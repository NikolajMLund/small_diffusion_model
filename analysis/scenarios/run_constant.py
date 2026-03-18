
from forecast import ForecastConfig, ModelConfig
from forecast.data_wrangler import load_data, prepare_data_for_forecast
from forecast.core import forecast, markov_step
import os
from forecast.plotting import plot_forecast_vs_actual

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '.', 'plots', 'constant')

data_path = 'processed_data.pkl'
model_config = ModelConfig()
forecast_config = ForecastConfig(
    base_year=2023,
    target_year=2024,
    invariant_disappearance_rates=True,
    invariant_inflows=True
)

data = load_data(data_path)
processed_data = prepare_data_for_forecast(data_path, model_config, forecast_config)
forecasted_distributions = forecast(
    state=processed_data['state'],
    dis_rates=processed_data['dis_rates'],
    purchase_inflows=processed_data['purchase_inflows'],
    model_config=model_config,
    forecast_config=forecast_config
)

plot_forecast_vs_actual(
    forecasted_distributions=forecasted_distributions,
    holdings_dist=data['holdings_dist'],
    model_config=model_config,
    forecast_config=forecast_config,
    output_dir=OUTPUT_DIR,
    file_name='forecast_vs_actual.png'
)
