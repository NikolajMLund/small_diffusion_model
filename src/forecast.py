import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import processed data 
with open('processed_data.pkl', 'rb') as f:
    processed_data = pickle.load(f)

# extract the generated data
holdings_dist = processed_data['holdings_dist']
dis_rate = processed_data['disappearance_rate']
market_shares = processed_data['market_shares']
ncpurch_prob = processed_data['ncpurch_prob']


# Age transition probabilities and survival rates. 
n_years = holdings_dist.index.get_level_values('year').nunique()
n_engine_types = holdings_dist.index.get_level_values('engine_type').nunique()
n_ages = holdings_dist.index.get_level_values('car_age').nunique()
age_transition_probs = np.zeros((n_years, n_engine_types, n_ages, n_ages))


year = 2024
car_type = 'BEV'
# constructing the age transition prob.
# np.eye(n, k=-1) has ones on the sub-diagonal:
#   A @ v shifts every element in v one position down (ages by 1 year).
#   The first element (age 0) becomes 0 — new entrants must be added separately.
age_step_matrix = np.eye(n_ages, k=-1)

# Multiply by the survival rates picking a (year, engine)-pair to exemplify
dis_rate_cleaned=dis_rate.loc[year, car_type, :].values 

dis_rate_cleaned = np.nan_to_num(dis_rate_cleaned, nan=0.0)  # Replace NaN with 0 for multiplication
dis_rate_clipped = np.clip(dis_rate_cleaned, 0, 1)  # Ensure values are between 0 and 1
dis_rate_clipped = dis_rate_clipped[1:]
dis_rate_clipped[-1] = 1.0  # Set the last age's disappearance rate to 1 (certain disappearance/ forced scrappage)
survival_age_matrix= (1 - dis_rate_clipped) * age_step_matrix

## THIS can be done so much more elegant by first assuming cars are scrapped/imported/exported and then calculating the transition probabilities.
# Q AGE 

# Next step is to create the behaviour matrix.
prob_buying_new=ncpurch_prob.loc[year, :].values
prob_buying_given_new = market_shares.loc[year, :].values 
prob_buying = prob_buying_new * prob_buying_given_new


holdings_dist_prev_year = holdings_dist.loc[year - 1, car_type, :].values


survival_age_matrix[0,:]
survival_age_matrix@holdings_dist_prev_year


#survival_and_buying_matrix = survival_age_matrix.copy()
#survival_and_buying_matrix[0, :] = prob_buying[1]

# transitioning
q_t=survival_age_matrix@holdings_dist_prev_year
q_t[0] += prob_buying[1]

q_tt = age_step_matrix@q_t
q_tt[0] += prob_buying[1]

# Compare q_t (forecast) with actual holdings distribution in year+1
actual_year = holdings_dist.loc[year, car_type, :].values
actual_year_plus1 = holdings_dist.loc[year + 1, car_type, :].values
ages = np.arange(n_ages)

width = 0.35
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.bar(ages - width/2, q_t, width=width, label=f'q_t forecast ({year})')
ax1.bar(ages + width/2, actual_year, width=width, label=f'Actual ({year})')
ax1.set_xlabel('Car age')
ax1.set_ylabel('Share of total holdings')
ax1.set_title(f'{car_type}: 1-year forecast vs actual ({year})')
ax1.legend()

ax2.bar(ages - width/2, q_tt, width=width, label=f'q_tt forecast ({year + 1})')
ax2.bar(ages + width/2, actual_year_plus1, width=width, label=f'Actual ({year + 1})')
ax2.set_xlabel('Car age')
ax2.set_ylabel('Share of total holdings')
ax2.set_title(f'{car_type}: 2-year forecast vs actual ({year + 1})')
ax2.legend()

plt.tight_layout()
plt.show()

