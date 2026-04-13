import os
import sys
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from pandas import IndexSlice as idx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase_in'))

from forecast.ModelConfig import ModelConfig
from forecast.ForecastConfig import ForecastConfig
from forecast.data_wrangler import load_data
from forecast.core import forecast
from forecast.plotting import plot_forecast_vs_actual
from forecast.scenarios.KFScenario import KFScenario, KFScenarioConfig
from plots import (
    plot_total_regression_fit, 
    plot_engine_share_over_time,
    wrangle_kf_to_engine_types,
)

#BASE_YEAR = 2024
#TARGET_YEAR = 2035
BASE_YEAR = 2024
TARGET_YEAR = 2035
data_limit_year = 2024 # Used in the projection 
scale_years = {2026: 12/2}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'plots')
DATA_PATH = 'processed_data.pkl'

from data_import import import_FAM55N
FAM55N = import_FAM55N()
#FIXTHIS: Should just be part of the dta imported. 
denom_choice = 2 * FAM55N[FAM55N['TID'] == 2020]['INDHOLD'].values[0]
n_oldcars = 72_000

model_config = ModelConfig(
    engine_types = ['BEV', 'ICEV', 'Old'],
    synthetic_engine_types = ['Old'],
)

forecast_config = ForecastConfig(
    base_year=BASE_YEAR, 
    target_year=TARGET_YEAR,
)

# TODO: This should happen in the data_process file.
data = load_data(DATA_PATH)
for key in ['car_purchases_market_shares', 'new_car_registrations_market_shares']:
    years = data[key].index.get_level_values('year')
    data[key] = data[key][~years.isin(scale_years.keys())]

# ----------------------------------------------------------------------
#  Scenario configuration
# - Disappearance rates are unchanged from the baseline.
# - Amount of cars sold is matched to KF25.
# - BEV/ICEV split Follows the existing approach.
# - Age distribution of inflows also follow existing approach.
# ------------------------------------------------------------------------

# ------------------------------
# Cars Sold Schedule (total number of cars flowing in each year)
# We use KF 2025 to match the stock going in each year. 
# ------------------------------

bestand = pd.read_excel('./analysis/data_kf/KF25 Transport Extract.xlsx', sheet_name='Bestand')
salg    = pd.read_excel('./analysis/data_kf/KF25 Transport Extract.xlsx', sheet_name='Salg')

bestand = bestand.rename(columns={'Unnamed: 0': 'year'}).set_index('year')
salg    = salg.rename(columns={'Unnamed: 0': 'year'}).set_index('year')

# KF bestand is "ultimo" (31 Dec), model is "primo" (1 Jan).
# Shift bestand years forward by 1 so KF end-of-Y aligns with model start-of-(Y+1).
# Salg (inflow) is not shifted — it represents activity during year Y in both conventions.
bestand.index = bestand.index + 1
#salg.index = salg.index + 1

KF_ENGINE_MAP_BESTAND = {
    'El':             'BEV',
    'Diesel':         'ICEV',
    'Benzin':         'ICEV',
    'Plug-in Hybrid': 'ICEV',
    'Øvrige':         'ICEV',
}

KF_ENGINE_MAP_SALG = {
    'El':             'BEV',
    'Diesel':         'ICEV',
    'Benzin':         'ICEV',
    'Plug-in Hybrid': 'ICEV',
    #'Øvrige':         'ICEV',
}

bestand = wrangle_kf_to_engine_types(bestand, KF_ENGINE_MAP_BESTAND)
salg = wrangle_kf_to_engine_types(salg, KF_ENGINE_MAP_SALG)

market_shares = data['car_purchases_market_shares']

projection_years = np.arange(forecast_config.base_year + 1, forecast_config.target_year + 1)

bestand_to_match = bestand / denom_choice

# The goal is to get projected sales such that it matches bestand 
# The challenge is that we have to add the amount of cars needed to match bestand in each year, after disappearances.
# This creates a chicken and egg problem because I have written the code to take inflow as given and then compute the distribution. 
#projected_sales

DATA_PLOT_DIR = os.path.join(OUTPUT_DIR, 'data')
SCENARIO_PLOT_DIR = os.path.join(OUTPUT_DIR, 'scenario')
COMPARISON_PLOT_DIR = os.path.join(OUTPUT_DIR, 'comparison')

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
    elif engine_type == 'Old':
        predicted_market_shares[:,i,:] = 0.0

assert np.all(np.isclose(predicted_market_shares.sum(axis=(1,2)), 1.0)), 'should sum to 1 in each year (at this stage)'

# ---------------------------------------------------------------
# Derive projected_sales by inverting the Markov model to match KF25 stock
# ---------------------------------------------------------------

from forecast.core import markov_step

# Instantiating a preliminary scenario to extract dis_rates
n_years = forecast_config.target_year -forecast_config.base_year
n_types = len(model_config.engine_types)
n_maxage = model_config.max_car_age + 2

dummy_config = KFScenarioConfig(
    projected_sales=np.zeros(n_years),
    market_shares=np.zeros((n_years, n_types, model_config.purchase_age_limit + 1)),
    n_oldcars=n_oldcars/denom_choice,
)
dummy_Scenario = KFScenario(data, model_config, forecast_config, dummy_config)
dummy_outcomes = dummy_Scenario.prepare()

shjt = np.full((n_years+1, n_types, n_maxage), np.nan)
shjt[0,...] = dummy_outcomes['state']
purchase_inflows = np.full((n_years+1, n_types, model_config.purchase_age_limit+1), 0.0)
projected_sales = np.zeros(n_years)
model_scrappage_fc = np.zeros(n_years)  # scrapped cars per forecast year
for t, year in enumerate(range(forecast_config.base_year,forecast_config.target_year)):
    shj=np.zeros((n_types, model_config.max_car_age+ 2))
    for i in range(len(model_config.engine_types)):
        # shjt is surviving cars in t+1 on the basis of holdings at t.
        shj[i,:] = markov_step(
            state=shjt[t, i, :],
            dis_rates=dummy_Scenario.dis_rates[t,i,:],
            purchase_inflows=np.zeros((model_config.purchase_age_limit+1)),
            model_config=model_config,
            forecast_config=forecast_config,
        )
    inflow = bestand_to_match.loc[idx[year+1,:]].sum() - shj.sum()
    purchase_inflows[t+1, ...] = predicted_market_shares[t, ...]*inflow
    shjt[t+1, ...] = shj
    shjt[t+1,:, 0:(model_config.purchase_age_limit + 1)] += purchase_inflows[t+1, ...]
    projected_sales[t] = inflow
    # Cars that disappeared during year: state before survival minus state after survival
    model_scrappage_fc[t] = (shjt[t,...].sum() - shj.sum()) * denom_choice
# Build initial state (mirrors KFScenario.get_state)
# This should give us the schedule we are looking for. 

# Create a plot of that age distribution, to sanity check it and to communicate assumptions.
## Plot that shows what happens in the forecast under these assumptions, showing the age distribution and the size of the inflows.
## A plot that shows the inflow of cars over the projected horizon. - Stacked maybe
# ---------------------------------------------------------------
# Packing scenario config
# ---------------------------------------------------------------

scenario_config = KFScenarioConfig(
    market_shares=predicted_market_shares,
    projected_sales=projected_sales,
    n_oldcars=n_oldcars/denom_choice,
)

Scenario = KFScenario(
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


################################################################
# Comparison with KF 2025
################################################################

# Read in data
from plots import wrangle_kf_to_engine_types, plot_kf_stock_total, plot_kf_stock_by_engine, plot_kf_inflow, plot_kf_inflow_by_engine, plot_kf_stock_difference, plot_kf_stock_difference_total, plot_base_year_stock_vs_inflow_diff, plot_kf_vs_model_scrappage, plot_kf_inflow_with_scrappage_gap


def create_kf_comparison(bestand, salg, model_config, forecast_config, denom_choice):
    """
    Extracts KF stock (bestand) and sales (salg) data into arrays aligned with
    the model's year convention.

    Note: bestand years have already been shifted +1 upstream (ultimo → primo
    alignment) before this function is called. Salg is unchanged.
    """
    kf_years         = bestand.index.get_level_values('year').unique()
    historical_years = kf_years[kf_years <= forecast_config.base_year]
    forecast_years   = kf_years[(kf_years > forecast_config.base_year) &
                                 (kf_years <= forecast_config.target_year)]
    engine_types = list(model_config.engine_types)

    def extract(series, years, engine_types):
        def safe_get(series, y, et):
            try:
                return series.loc[y, et]
            except KeyError as e:
                if et not in np.atleast_1d(model_config.synthetic_engine_types):
                    raise
                return 0.0
        return np.array([
            [safe_get(series, y, et) for et in engine_types]
            for y in years
        ]) / denom_choice  # → (n_periods, n_engine_types), same scale as model

    return (
        extract(bestand, historical_years, engine_types),
        extract(salg,    historical_years, engine_types),
        extract(bestand, forecast_years,   engine_types),
        extract(salg,    forecast_years,   engine_types),
        historical_years,
        forecast_years,
    )

(historical_distributions_kf, historical_inflows_kf,
 forecasted_distributions_kf, projected_inflows_kf,
 kf_historical_years, kf_forecast_years) = create_kf_comparison(
    bestand,
    salg,
    model_config,
    forecast_config,
    denom_choice,
)

plot_kf_stock_total(
    historical_distributions_kf=historical_distributions_kf,
    forecasted_distributions_kf=forecasted_distributions_kf,
    kf_historical_years=kf_historical_years,
    kf_forecast_years=kf_forecast_years,
    forecasted_distributions=forecasted_distributions,
    holdings_dist=data['holdings_dist'],
    forecast_config=forecast_config,
    output_dir=COMPARISON_PLOT_DIR,
)

plot_kf_stock_by_engine(
    historical_distributions_kf=historical_distributions_kf,
    forecasted_distributions_kf=forecasted_distributions_kf,
    kf_historical_years=kf_historical_years,
    kf_forecast_years=kf_forecast_years,
    forecasted_distributions=forecasted_distributions,
    holdings_dist=data['holdings_dist'],
    model_config=model_config,
    forecast_config=forecast_config,
    output_dir=COMPARISON_PLOT_DIR,
)

plot_kf_inflow(
    historical_inflows_kf=historical_inflows_kf,
    projected_inflows_kf=projected_inflows_kf,
    kf_historical_years=kf_historical_years,
    kf_forecast_years=kf_forecast_years,
    projected_inflows=prepared['projected_inflows'],
    car_purchases_market_shares=data['car_purchases_market_shares'],
    forecast_config=forecast_config,
    output_dir=COMPARISON_PLOT_DIR,
)

plot_kf_inflow_by_engine(
    historical_inflows_kf=historical_inflows_kf,
    projected_inflows_kf=projected_inflows_kf,
    kf_historical_years=kf_historical_years,
    kf_forecast_years=kf_forecast_years,
    projected_inflows=prepared['projected_inflows'],
    car_purchases_market_shares=data['car_purchases_market_shares'],
    model_config=model_config,
    forecast_config=forecast_config,
    output_dir=COMPARISON_PLOT_DIR,
)

plot_kf_stock_difference(
    historical_distributions_kf=historical_distributions_kf,
    forecasted_distributions_kf=forecasted_distributions_kf,
    kf_historical_years=kf_historical_years,
    kf_forecast_years=kf_forecast_years,
    forecasted_distributions=forecasted_distributions,
    holdings_dist=data['holdings_dist'],
    denom_choice=denom_choice,
    model_config=model_config,
    forecast_config=forecast_config,
    output_dir=COMPARISON_PLOT_DIR,
)

plot_kf_stock_difference_total(
    historical_distributions_kf=historical_distributions_kf,
    forecasted_distributions_kf=forecasted_distributions_kf,
    kf_historical_years=kf_historical_years,
    kf_forecast_years=kf_forecast_years,
    forecasted_distributions=forecasted_distributions,
    holdings_dist=data['holdings_dist'],
    denom_choice=denom_choice,
    forecast_config=forecast_config,
    output_dir=COMPARISON_PLOT_DIR,
)

plot_base_year_stock_vs_inflow_diff(
    historical_distributions_kf=historical_distributions_kf,
    projected_inflows_kf=projected_inflows_kf,
    kf_historical_years=kf_historical_years,
    kf_forecast_years=kf_forecast_years,
    projected_inflows=prepared['projected_inflows'],
    holdings_dist=data['holdings_dist'],
    denom_choice=denom_choice,
    forecast_config=forecast_config,
    output_dir=COMPARISON_PLOT_DIR,
)

# KF implied scrappage: stock(Y) + inflow(Y) - stock(Y+1)
_kf_all_years  = np.concatenate([kf_historical_years, kf_forecast_years])
_kf_all_stock  = np.concatenate([historical_distributions_kf.sum(axis=1),
                                  forecasted_distributions_kf.sum(axis=1)])
_kf_all_inflow = np.concatenate([historical_inflows_kf.sum(axis=1),
                                  projected_inflows_kf.sum(axis=1)])
_ks_years, _ks_values = [], []
for _i, _y in enumerate(_kf_all_years[:-1]):
    _y_next = _y + 1
    if _y_next not in _kf_all_years:
        continue
    _j = np.where(_kf_all_years == _y_next)[0][0]
    _ks_values.append((_kf_all_stock[_i] + _kf_all_inflow[_i] - _kf_all_stock[_j]) * denom_choice)
    _ks_years.append(_y)
kf_scrap_years  = np.array(_ks_years)
kf_scrap_values = np.array(_ks_values)
scrappage_fc_years = np.arange(forecast_config.base_year,
                                forecast_config.base_year + len(model_scrappage_fc))

plot_kf_vs_model_scrappage(
    kf_scrap_years=kf_scrap_years,
    kf_scrap_values=kf_scrap_values,
    scrappage_fc_years=scrappage_fc_years,
    model_scrappage_fc=model_scrappage_fc,
    denom_choice=denom_choice,
    forecast_config=forecast_config,
    output_dir=COMPARISON_PLOT_DIR,
)

plot_kf_inflow_with_scrappage_gap(
    historical_inflows_kf=historical_inflows_kf,
    projected_inflows_kf=projected_inflows_kf,
    kf_historical_years=kf_historical_years,
    kf_forecast_years=kf_forecast_years,
    projected_inflows=prepared['projected_inflows'],
    car_purchases_market_shares=data['car_purchases_market_shares'],
    kf_scrap_years=kf_scrap_years,
    kf_scrap_values=kf_scrap_values,
    model_scrappage_fc=model_scrappage_fc,
    scrappage_fc_years=scrappage_fc_years,
    denom_choice=denom_choice,
    forecast_config=forecast_config,
    output_dir=COMPARISON_PLOT_DIR,
)


