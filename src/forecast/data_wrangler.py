"""
This file contains the core functionality for getting the data as it currently is in pandas into the right format,
based on the options made in the ForecastConfig and ModelConfig dataclasses. 
"""

import pickle
import numpy as np
import pandas as pd
from pandas import IndexSlice as idx
from forecast.ModelConfig import ModelConfig
from forecast.ForecastConfig import ForecastConfig
def load_data(data_path: str) -> dict:
    with open(data_path, 'rb') as f:
        return pickle.load(f)


def prepare_data_for_forecast(
        data_path: str, 
        model_config: ModelConfig, 
        forecast_config: ForecastConfig
)-> dict:
    """ 
    Loads the processed data and prepares it for the forecast.
    inputs: 
    the pickled processed data file.

    Outputs and their expected shapes:     
    state: (n_car_types, n_ages) current age distribution
    dis_rates: (n_forecast_years, n_car_types, n_ages) raw disappearance rates for the step year
    purchase_inflows: (n_forecast_years, n_car_types, max_purchase_age+1) purchase shares by car_type, age
    """    

    # unpack config
    base_year = forecast_config.base_year
    target_year = forecast_config.target_year
    n_forecast_years = target_year - base_year
    invariant_disappearance_rates = forecast_config.invariant_disappearance_rates
    invariant_inflows = forecast_config.invariant_inflows
    car_types = model_config.engine_types
    
    # load data
    data = load_data(data_path)
    
    # ----------------------------------------------------------------------
    # Holdings distribution: 
    # wrapping holdings distribution into a np.array of shape (n_car_types, n_ages)
    # ----------------------------------------------------------------------
    holdings_dist = data['holdings_dist']
    state = np.zeros((len(car_types), model_config.max_car_age + 2)) + np.nan
    for i, car_type in enumerate(car_types):
        state[i, :] = holdings_dist.loc[base_year, car_type, :].values

    # ----------------------------------------------------------------------
    # Disappearance rates: 
    # wrapping disappearance rates into a np.array of shape (n_forecast_years, n_car_types, n_ages)
    # STATUS: HACK since there both exist a scrap_profile (by age) and the actual disappearance rates.
    # ----------------------------------------------------------------------
    ages = np.arange(model_config.max_car_age + 2)
    dis_rates = np.zeros(
        (n_forecast_years, len(car_types), model_config.max_car_age + 2)
        ) + np.nan
    for i, car_type in enumerate(car_types):
        for t, year in enumerate(np.arange(base_year, target_year)):
            if invariant_disappearance_rates:
                dis_rates[t, i, :] = (
                    data['scrap_profile']
                    .reindex(ages, fill_value=0)
                    .values
                )
            else:
                dis_rates[t, i, :] = (
                    data['scrap_profile']
                    .reindex(ages, fill_value=0)
                    .values
                )

    # ----------------------------------------------------------------------
    # Purchase inflows: 
    # wrapping purchase inflows into a np.array of shape (n_forecast_years, n_car_types, max_purchase_age+1)
    # ----------------------------------------------------------------------
    max_purchase_age = model_config.purchase_age_limit
    purchase_ages = np.arange(max_purchase_age + 1)
    purchase_inflows = np.zeros(
        (n_forecast_years, len(car_types), max_purchase_age + 1)
        ) + np.nan
    for i, car_type in enumerate(car_types):
        for t, year in enumerate(np.arange(base_year, target_year)):
            if invariant_inflows:
                purchase_inflows[t, i, :] = (
                    data['car_purchases_market_shares']
                    .loc[idx[base_year, car_type, :]]
                    .reindex(purchase_ages, fill_value=0)
                    .values
                )
            else:
                purchase_inflows[t, i, :] = (
                    data['car_purchases_market_shares']
                    .loc[idx[year, car_type, :]]
                    .reindex(purchase_ages, fill_value=0)
                    .values
                )

    return {
        'state': state,
        'dis_rates': dis_rates,
        'purchase_inflows': purchase_inflows
    }

if __name__ == "__main__":
    model_config = ModelConfig()
    forecast_config = ForecastConfig(
        target_year=2024,
        invariant_disappearance_rates=True,
        invariant_inflows=True
    )
    data_path = 'processed_data.pkl'
    prepared_data = prepare_data_for_forecast(data_path, model_config, forecast_config)
    
    state = prepared_data['state']
    dis_rates = prepared_data['dis_rates']
    purchase_inflows = prepared_data['purchase_inflows']
