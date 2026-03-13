import pickle
import numpy as np
import pandas as pd

# Import processed data 
with open('processed_data.pkl', 'rb') as f:
    processed_data = pickle.load(f)

# extract the generated data
holdings_dist = processed_data['holdings_dist']
dis_rate = processed_data['disappearance_rate']
market_shares = processed_data['market_shares']
ncpurch_prob = processed_data['ncpurch_prob']

breakpoint()

# Age transition probabilities and survival rates. 
n_years = holdings_dist.index.get_level_values('year').nunique()
n_engine_types = holdings_dist.index.get_level_values('engine_type').nunique()
n_ages = holdings_dist.index.get_level_values('car_age').nunique()
age_transition_probs = np.zeros((n_years, n_engine_types, n_ages, n_ages))

# constructing the age transition prob.
# np.eye(n, k=-1) has ones on the sub-diagonal:
#   A @ v shifts every element in v one position down (ages by 1 year).
#   The first element (age 0) becomes 0 — new entrants must be added separately.
age_step_matrix = np.eye(n_ages, k=-1)

# Multiply by the survival rates picking one year, engine to exemplify
dis_rate_cleaned=dis_rate.loc[2024, 'ICEV', :].values 

dis_rate_cleaned = np.nan_to_num(dis_rate_cleaned, nan=0.0)  # Replace NaN with 0 for multiplication
dis_rate_clipped = np.clip(dis_rate_cleaned, 0, 1)  # Ensure values are between 0 and 1
dis_rate_clipped = dis_rate_clipped[1:]
dis_rate_clipped[-1] = 1.0  # Set the last age's disappearance rate to 1 (certain disappearance/ forced scrappage)
survival_age_matrix= (1 - dis_rate_clipped) * age_step_matrix

## THIS can be done so much more elegant by first assuming cars are scrapped/imported/exported and then calculating the transition probabilities.
# Q AGE 







# Now we can multiply onto a holdings vector.




#dis_rates = dis_rate.unstack('engine_type').values  # shape (n_years, n_engine_types, n_ages)



