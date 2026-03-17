import pickle

import numpy as np

def load_data(data_path: str) -> dict:
    with open(data_path, 'rb') as f:
        return pickle.load(f)

def markov_step(
    state: np.ndarray,
    dis_rates: np.ndarray,
    purchase_inflows: np.ndarray,
    model_config,
    forecast_config
) -> np.ndarray:
    """
    One discrete-time Markov transition step.

    Replicates the transition logic from the original forecast.py:
    1. Build an age-shift matrix (subdiagonal).
    2. Scale each column by its survival probability (1 - disappearance rate).
    3. Force the oldest cohort to scrap with certainty.
    4. Add purchase inflows for ages 0..max_purchase_age.

    Parameters
    ----------
    state : (n_ages,) current age distribution
    dis_rates : (n_ages,) raw disappearance rates for the step year
    purchase_inflows : (max_purchase_age+1,) purchase shares by age
    max_purchase_age : int
    n_ages : int
    """
    age_step_matrix = np.eye(model_config.max_car_age+2, k=-1) # +2 to account for age 0 and the forced scrappage age

    survival_matrix = (1 - dis_rates) * age_step_matrix

    next_state = survival_matrix @ state
    
    next_state[0:(model_config.purchase_age_limit + 1)] += purchase_inflows

    return next_state
