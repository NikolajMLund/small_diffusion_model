import os
import sys

import numpy as np
from pandas import IndexSlice as idx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from forecast.ModelConfig import ModelConfig
from forecast.ForecastConfig import ForecastConfig
from forecast.data_wrangler import load_data
from forecast.core import forecast
from forecast.plotting import plot_forecast_vs_actual
from forecast.scenarios.PhaseInScenario import PhaseInScenario, PhaseInScenarioConfig
from plots import plot_total_sales_forecast, plot_engine_share_over_time

#BASE_YEAR = 2024
#TARGET_YEAR = 2035
BASE_YEAR = 2023
TARGET_YEAR = 2035
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

market_shares = data['car_purchases_market_shares']
infer_sales_year = np.arange(
    market_shares.index.get_level_values('year').min(),
    forecast_config.base_year + 1 
)

new_car_sales=market_shares.groupby('year').sum().loc[infer_sales_year]
actual_car_sales=market_shares.groupby('year').sum()

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
projected_sales = model_fit.predict(x_proj)

DATA_PLOT_DIR = os.path.join(OUTPUT_DIR, 'data')
SCENARIO_PLOT_DIR = os.path.join(OUTPUT_DIR, 'scenario')

plot_total_sales_forecast(
    historical_years=historical_years,
    y=y,
    sales_forecast=sales_forecast,
    actual_car_sales = actual_car_sales,
    projection_years=projection_years,
    sales_projection=projected_sales,
    base_year=forecast_config.base_year,
    output_dir=DATA_PLOT_DIR,
)

# --------------------------------------------------------------
# BEV/ICEV split of purchases
# --------------------------------------------------------------
plot_engine_share_over_time(
    market_shares=data['car_purchases_market_shares'],
    new_reg_market_shares=data['new_car_registrations_market_shares'],
    base_year=forecast_config.base_year,
    output_dir=DATA_PLOT_DIR,
)


# --------------------------------------------------------------
# BEV diffusion curves (logistic fit)
# --------------------------------------------------------------
from scipy.optimize import curve_fit
from pandas import IndexSlice as idx
from plots import plot_bev_diffusion_fit

L = 0.95  # fixed saturation ceiling

def logistic(t, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))

def logistic_restricted(t, k):
    return L / (1 + np.exp(-k * (t - 2025)))

# Observed BEV share — total inflow incl. implied imports
by_year_eng_total = market_shares.groupby(['year', 'engine_type']).sum()
total_total_by_year = by_year_eng_total.groupby('year').sum()
bev_share_total = (by_year_eng_total.loc[idx[:, 'BEV']] / total_total_by_year).dropna()

# Observed BEV share — new registrations only
new_reg = data['new_car_registrations_market_shares']
by_year_eng_reg = new_reg.groupby(['year', 'engine_type']).sum()
total_reg_by_year = by_year_eng_reg.groupby('year').sum()
bev_share_reg = (by_year_eng_reg.loc[idx[:, 'BEV']] / total_total_by_year).dropna()
breakpoint()


data_limit_year = 2024
bev_share_total_est = bev_share_total[bev_share_total.index <= data_limit_year ]  # only fit to data prior to the limit year
bev_share_reg_est = bev_share_reg[bev_share_reg.index <= data_limit_year ]  # only fit to data prior to the limit year

years_reg   = bev_share_reg_est.index.values.astype(float)
years_total = bev_share_total_est.index.values.astype(float)
popt_reg, _   = curve_fit(logistic, years_reg,   bev_share_reg_est.values,
                           p0=[0.4, 2025], bounds=([0.01, 2015], [5.0, 2032]))
popt_total, _ = curve_fit(logistic, years_total, bev_share_total_est.values,
                           p0=[0.4, 2025], bounds=([0.01, 2015], [5.0, 2032]))

popt_reg_manual_inflection, _   = curve_fit(logistic_restricted, years_reg,   bev_share_reg_est.values,
                           p0=[0.4,], bounds=([0.01], [5.0,]))
popt_total_manual_inflection, _ = curve_fit(logistic_restricted, years_total, bev_share_total_est.values,
                           p0=[0.4,], bounds=([0.01], [5.0]))


print(f"New-reg fit:      k={popt_reg[0]:.3f}, t0={popt_reg[1]:.1f}")
print(f"Total-inflow fit: k={popt_total[0]:.3f}, t0={popt_total[1]:.1f}")

print(f"New-reg fit:      k={popt_reg_manual_inflection[0]:.3f}, t0 fixed at 2025")
print(f"Total-inflow fit: k={popt_total_manual_inflection[0]:.3f}, t0 fixed at 2025")


# Evaluate over historical + projection window
all_years = np.arange(years_reg.min(), forecast_config.target_year + 1, dtype=float)
bev_fit_reg   = logistic(all_years, *popt_reg)
bev_fit_total = logistic(all_years, *popt_total)
bev_fit_reg_manual_inflection = logistic_restricted(all_years, *popt_reg_manual_inflection)
bev_fit_total_manual_inflection = logistic_restricted(all_years, *popt_total_manual_inflection)

plot_bev_diffusion_fit(
    all_years=all_years,
    bev_fit_reg=bev_fit_reg,
    bev_fit_total=bev_fit_total,
    bev_share_reg=bev_share_reg,
    bev_share_total=bev_share_total,
    base_year=forecast_config.base_year,
    data_limit_year=data_limit_year,
    saturation=L,
    output_dir=DATA_PLOT_DIR,
    file_name = 'bev_diffusion_fit.png'
)

plot_bev_diffusion_fit(
    all_years=all_years,
    bev_fit_reg=bev_fit_reg_manual_inflection,
    bev_fit_total=bev_fit_total_manual_inflection,
    bev_share_reg=bev_share_reg,
    bev_share_total=bev_share_total,
    base_year=forecast_config.base_year,
    data_limit_year=data_limit_year,
    saturation=L,
    output_dir=DATA_PLOT_DIR,
    file_name = 'bev_diffusion_fit_2025_inflection.png'
)

predicted_market_shares = np.full(
    shape= (
        len(projection_years), 
        len(model_config.engine_types), 
        model_config.purchase_age_limit + 1
        ), 
        fill_value=np.nan
)
projected_inflows = predicted_market_shares.copy()

# BEV inflows (as a share of total sales)
# Predicted inflow - new registrations 
predicted_market_shares_reg_bev = bev_fit_reg[np.isin(all_years, projection_years)]
# predicted inflow - total (incl. imports)
predicted_market_shares_total_bev = bev_fit_total[np.isin(all_years, projection_years)]
# Predicted inflow - import 
predicted_market_shares_import_bev = predicted_market_shares_total_bev - predicted_market_shares_reg_bev

# ICEV inflows (as a share of total sales)
# Documentation: ICEVs is a residual of the market share fraction of BEVs. 
predicted_market_shares_reg_icev = (1 - bev_fit_total)[np.isin(all_years, projection_years)]

# sanity check
sanity=predicted_market_shares_reg_icev + predicted_market_shares_reg_bev + predicted_market_shares_import_bev 
assert np.all(np.isclose(sanity,1.0)), 'Shares should sum to 1.0' 

# --------------------------------------------------------------
# Age distribution of predicted inflows - imports distributed by age
# --------------------------------------------------------------
# age distribution:
# For simplicity, we assume the age distribution of inflows remains constant over time, BEVs
# There is no imports for ICEVs (Probably wrong but is masked by higher exports than imports.)

from plots import plot_age_distribution_of_inflows
plot_age_distribution_of_inflows(
    car_purchases_market_shares=data['car_purchases_market_shares'],
    #new_reg_market_shares=data['market_shares'],
    base_year=forecast_config.base_year,
    output_dir=DATA_PLOT_DIR,
)

imports = data['car_purchases_market_shares'][
    data['car_purchases_market_shares'].index.get_level_values('car_age') >= 1
]
imports_age_dist = (
    imports.groupby(['year', 'engine_type', 'car_age']).sum() /
    imports.groupby(['year', 'engine_type']).sum()
)

# I just pick out 2024. Modelling this seems rather complex otherwise. 
assumption_age_dist_import_year = 2024

imports_age_dist_chosen_bev = imports_age_dist.loc[idx[assumption_age_dist_import_year, 'BEV', np.arange(1, model_config.purchase_age_limit+1)]].values

# create age distribution of sales:
predicted_market_shares_import_by_age_bev = predicted_market_shares_import_bev[:, np.newaxis] * imports_age_dist_chosen_bev[np.newaxis, :]

# now we append the new car sales as projected. (5,) (5,6) -> (5,7)
predicted_market_shares_of_bevs=np.concatenate([predicted_market_shares_reg_bev[:,np.newaxis],predicted_market_shares_import_by_age_bev], axis=1)

assert np.all(np.isclose(predicted_market_shares_of_bevs.sum(axis=1), predicted_market_shares_total_bev)), 'the inflows should match'

# ICEVs: 
# Assumption is no imports for ICEVs so we only need to insert for the first value  
predicted_market_shares_of_icevs = predicted_market_shares_reg_icev.copy()

# storing (Done this way to make sure that BEV and ICEVs are indexed correctly.)
for i, engine_type in enumerate(model_config.engine_types): 
    if engine_type == 'BEV':
        predicted_market_shares[:,i,:] = predicted_market_shares_of_bevs
    elif engine_type == 'ICEV':
        predicted_market_shares[:,i,1:] = 0.0
        predicted_market_shares[:,i,0] = predicted_market_shares_of_icevs

assert np.all(np.isclose(predicted_market_shares.sum(axis=(1,2)), 1.0)), 'should sum to 1 in each year (at this stage)'

projected_inflows[...] = predicted_market_shares * projected_sales[:, np.newaxis, np.newaxis]

# This should give us the schedule we are looking for. 

# Create a plot of that age distribution, to sanity check it and to communicate assumptions.
## Plot that shows what happens in the forecast under these assumptions, showing the age distribution and the size of the inflows.
## A plot that shows the inflow of cars over the projected horizon. - Stacked maybe

# ---------------------------------------------------------------
# Packing scenario config
# ---------------------------------------------------------------

scenario_config = PhaseInScenarioConfig(
    market_shares=predicted_market_shares,
    projected_sales=projected_sales
)

Scenario = PhaseInScenario(
    data=data,
    model_config=model_config,
    forecast_config=forecast_config,
    scenario_config=scenario_config,
)

prepared = Scenario.prepare()

forecasted_distributions = forecast(
    state=prepared['state'],
    dis_rates=prepared['dis_rates'],
    purchase_inflows=prepared['projected_inflows'],
    model_config=model_config,
    forecast_config=forecast_config,
)

Scenario.plot_all(output_dir=SCENARIO_PLOT_DIR, forecasted_distributions=forecasted_distributions)

plot_forecast_vs_actual(
    forecasted_distributions=forecasted_distributions,
    holdings_dist=data['holdings_dist'],
    model_config=model_config,
    forecast_config=forecast_config,
    output_dir=SCENARIO_PLOT_DIR,
    file_name='forecast_vs_actual_phase_in_test.png',
)
