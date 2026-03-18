import numpy as np

from forecast.ModelConfig import ModelConfig
from forecast.ForecastConfig import ForecastConfig

def forecast(
    state: np.ndarray,
    dis_rates: np.ndarray,
    purchase_inflows: np.ndarray,
    model_config,
    forecast_config
) -> dict:  
    """
    state: (n_car_types, n_ages) current age distribution
    dis_rates: (n_forecast_years, n_car_types, n_ages) raw disappearance rates for the step year
    purchase_inflows: (n_forecast_years, n_car_types, max_purchase_age+1) purchase shares by car_type, age
    """

    target_year = forecast_config.target_year
    n_periods = target_year - forecast_config.base_year + 1
    n_car_types = len(model_config.engine_types)
    n_ages = model_config.max_car_age + 2  # +2 to account for age 0 and the forced scrappage age

    # Initialize a np.array to hold the forecasted age distributions for each car type and year
    forecasted_distributions = np.zeros((n_periods, n_car_types, n_ages)) + np.nan

    # Set the initial state for the first year
    for i, car_type in enumerate(model_config.engine_types):        
        for t in np.arange(n_periods):
            if t == 0: # base year: just set the state, no transition
                forecasted_distributions[t, i, :] = state[i, :]
            else: 
                forecasted_distributions[t, i, :] = markov_step(
                    state=forecasted_distributions[t-1, i, :],
                    dis_rates=dis_rates[t-1, i, :],
                    purchase_inflows=purchase_inflows[t-1, i, :],
                    model_config=model_config,
                    forecast_config=forecast_config
                )

    return forecasted_distributions[1:, :, :]  # return only the forecasted years, excluding the initial state

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

if __name__ == "__main__":
    from forecast.data_wrangler import load_data, prepare_data_for_forecast
    data_path = 'processed_data.pkl'
    model_config = ModelConfig()
    forecast_config = ForecastConfig(
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
    breakpoint()