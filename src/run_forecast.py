import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from forecast import  markov_step
from forecast.ModelConfig import ModelConfig
from forecast.ForecastConfig import ForecastConfig




# Import processed data 
with open('processed_data.pkl', 'rb') as f:
    processed_data = pickle.load(f)

# extract the generated data
holdings_dist = processed_data['holdings_dist']
dis_rate = processed_data['disappearance_rate']
market_shares = processed_data['market_shares']
ncpurch_prob = processed_data['ncpurch_prob']
car_purchases_market_shares = processed_data['car_purchases_market_shares']
dis_rates = processed_data['scrap_profile'].values

# Age transition probabilities and survival rates. 
n_ages = holdings_dist.index.get_level_values('car_age').max()


model_config=ModelConfig() # uses default values

forecast_config=ForecastConfig(
    target_year=2024,
    car_type='BEV'
)

max_age_car_traded=car_purchases_market_shares.index.get_level_values('car_age').unique().max()
prob_buying = car_purchases_market_shares

holdings_dist_base = holdings_dist.loc[forecast_config.target_year - 1, forecast_config.car_type, :].values

q_t=markov_step(
    state=holdings_dist_base,
    dis_rates=dis_rates,
    purchase_inflows=prob_buying.loc[forecast_config.target_year - 1, forecast_config.car_type, :].values,
    model_config=model_config,
    forecast_config=forecast_config,
)

q_tt = markov_step(
    state=q_t,
    dis_rates=dis_rates,
    purchase_inflows=prob_buying.loc[forecast_config.target_year, forecast_config.car_type, :].values,
    model_config=model_config,
    forecast_config=forecast_config,
)

# Compare q_t (forecast) with actual holdings distribution in year+1
actual_year = holdings_dist.loc[forecast_config.target_year, forecast_config.car_type, :].values
actual_year_plus1 = holdings_dist.loc[forecast_config.target_year + 1, forecast_config.car_type, :].values
ages = np.arange(model_config.max_car_age + 2)  # +2 to account for age 0 and the forced scrappage age

width = 0.35
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.bar(ages - width/2, q_t, width=width, label=f'q_t forecast ({forecast_config.target_year})')
ax1.bar(ages + width/2, actual_year, width=width, label=f'Actual ({forecast_config.target_year})')
ax1.set_xlabel('Car age')
ax1.set_ylabel('Share of total holdings')
ax1.set_title(f'{forecast_config.car_type}: 1-year forecast vs actual ({forecast_config.target_year})')
ax1.legend()

ax2.bar(ages - width/2, q_tt, width=width, label=f'q_tt forecast ({forecast_config.target_year + 1})')
ax2.bar(ages + width/2, actual_year_plus1, width=width, label=f'Actual ({forecast_config.target_year + 1})')
ax2.set_xlabel('Car age')
ax2.set_ylabel('Share of total holdings')
ax2.set_title(f'{forecast_config.car_type}: 2-year forecast vs actual ({forecast_config.target_year + 1})')
ax2.legend()

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'analysis', 'plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.savefig(os.path.join(OUTPUT_DIR, f'forecast_vs_actual_test.png'))
plt.tight_layout()
plt.show()

