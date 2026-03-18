
from forecast import ForecastConfig, ModelConfig
from forecast.data_wrangler import load_data, prepare_data_for_forecast
from forecast.core import forecast, markov_step
    
data_path = 'processed_data.pkl'
model_config = ModelConfig()
forecast_config = ForecastConfig(
    target_year=2024,
    base_year=2023,
    car_types=['ICEV', 'BEV'],
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


import matplotlib.pyplot as plt
import os
import numpy as np
ages = np.arange(model_config.max_car_age + 2)  # +2 to account for age 0 and the forced scrappage age
q_t = forecasted_distributions[0, 1, :]  # Year 1 forecast for the first car type (e.g., BEV)
width = 0.35
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.bar(ages - width/2, q_t, width=width, label=f'q_t forecast ({forecast_config.target_year})')
#ax1.bar(ages + width/2, actual_year, width=width, label=f'Actual ({forecast_config.target_year})')
ax1.set_xlabel('Car age')
ax1.set_ylabel('Share of total holdings')
ax1.set_title(f'{forecast_config.car_types[1]}: 1-year forecast vs actual ({forecast_config.target_year})')
ax1.legend()

#ax2.bar(ages - width/2, q_tt, width=width, label=f'q_tt forecast ({forecast_config.target_year + 1})')
#ax2.bar(ages + width/2, actual_year_plus1, width=width, label=f'Actual ({forecast_config.target_year + 1})')
ax2.set_xlabel('Car age')
ax2.set_ylabel('Share of total holdings')
ax2.set_title(f'{forecast_config.car_types[1]}: 2-year forecast vs actual ({forecast_config.target_year + 1})')
ax2.legend()

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'analysis', 'plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.savefig(os.path.join(OUTPUT_DIR, f'forecast_vs_actual_test.png'))
plt.tight_layout()
plt.show()
