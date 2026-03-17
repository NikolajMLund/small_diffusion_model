# This file creates the cleans and process the relevant data series. 

import pandas as pd
import numpy as np
import data_import
import logging
import matplotlib.pyplot as plt
import visualisation
import pickle


def track(fn):
    def wrapper(df, *args, **kwargs):
        before = len(df)
        result = fn(df, *args, **kwargs)
        after = len(result)
        print(f"{fn.__name__}: {before} → {after} rows ({before - after} dropped)")
        return result
    return wrapper


# Importing data: 
BIL21=data_import.import_BIL21()
BIL51=data_import.import_BIL51()
FAM55N=data_import.import_FAM55N() 

########################
## BIL21: PROCESSING ###
########################

## Create car groups: ##
Drivmidler = {
    'Benzin': 'ICEV',
    'Diesel': 'ICEV',
    'El': 'BEV',
    'Hybrid': 'ICEV',
    'Pluginhybrid': 'ICEV',
    'F-gas': np.nan, # Drop rows with 'F-gas'
    'N-gas': np.nan, # Drop rows with 'N-gas'
    'Petroleum': np.nan, # Drop rows with 'Petroleum'
    'Brint': np.nan, # Drop rows with 'Brint'
    'Metanol': np.nan, # Drop rows with 'Metanol'
    'Ætanol': np.nan, # Drop rows with 'Ætanol'
}

BIL21['engine_type'] = BIL21['DRIV'].map(Drivmidler)
BIL21.dropna(subset=['engine_type'], inplace=True)

ages_to_remap = {
    'Mere end 24 år': '25 år',
    'Alder i alt': np.nan, # Drop rows with 'Alder i alt'
}

BIL21=BIL21.replace({'ALDER1': ages_to_remap})
BIL21.dropna(subset=['ALDER1'], inplace=True)
#regex for extracting the number of years from the 'ALDER1' column
BIL21['car_age'] = BIL21['ALDER1'].str.extract(r'(\d+)').astype(int)
BIL21 = (
    BIL21[['TID', 'engine_type', 'car_age', 'INDHOLD']]
    .rename(columns={'TID': 'year', 'INDHOLD': 'count'})
    .groupby(['year', 'engine_type', 'car_age'], as_index=False)['count'].sum()
)
BIL21.set_index(['year', 'engine_type', 'car_age'], inplace=True)

########################
## BIL51: PROCESSING ###
########################
EJER_mapping = {
    'Husholdningerne': 'Alle',
    'Erhvervene': 'Alle', 
}

BIL51['owner_type'] = BIL51['EJER'].map(EJER_mapping)
BIL51.dropna(subset=['owner_type'], inplace=True)

BIL51['engine_type'] = BIL51['DRIV'].map(Drivmidler)
BIL51.dropna(subset=['engine_type'], inplace=True)
BIL51['year'] = BIL51['TID'].str.extract(r'(\d{4})')
BIL51['month'] = BIL51['TID'].str.extract(r'(M\d{2})')
BIL51['month'] = BIL51['month'].str.extract(r'(\d{2})')

# sum over months to get annual data

BIL51 = (
    BIL51.rename(columns={'INDHOLD': 'count'})
    .astype({'year': int, 'month': int, 'count': int})
)
BIL51 = BIL51.groupby(['year', 'owner_type', 'engine_type'], as_index=False)['count'].sum()
BIL51 = BIL51.set_index(['year', 'owner_type', 'engine_type'])
#########################
## FAM55N: PROCESSING ###
#########################

FAM55N.rename(columns={'TID': 'year', 'INDHOLD': 'count'}, inplace=True)

##########################
## Create market shares: #
##########################

# Using BIL51 to extract new registrations.
market_shares=BIL51['count']/BIL51.groupby(['year', 'owner_type'], as_index=True)['count'].transform('sum')

###################################
## prob. of purchasing a new car: #
###################################
denom_choice  = 2*FAM55N[FAM55N['year'] == 2020]['count'].values[0]

ncpurch_prob=(
    BIL51.groupby(['year', 'owner_type'], as_index=True)['count'].sum()/
    # shift +1: households observed beginning of t are denominator for purchases in t + 1
    #FAM55N.groupby(['year'], as_index=True)['count'].sum().rename(index=lambda x: x - 1)
    #BIL21.groupby('year', as_index=True)['count'].sum()
    denom_choice
)

###################################
## Scrap rate:                  ###
###################################
"""
Based on BIL21.
for each vintage we should be able to use:
stock(t) = stock(t-1) + Import(t) - Export(t) - Scrap(t)
We do not observe import and exports so let's instead assume 
stock(t) = stock(t-1)  - disapperance(t)
and then calculate the disappearance rate as 
disappearance(t)/stock(t-1) = 1 - stock(t)/stock(t-1) 
"""

# Vintage-level disappearance rate:
# disappearance_rate(age=a, year=t) = 1 - N(t+1, a+1) / N(t, a)
# Purchases drop out since no new cars join a cohort after age 0.
# Align N(t+1, a+1) with N(t, a) by shifting year and car_age back by 1.

stockt0 = BIL21['count']
stockt1=(
    BIL21
    .rename(index=lambda x: x - 1, level='year')
    .rename(index=lambda x: x - 1, level='car_age')
    )['count']

# this calculates 
#           age=0   age=1   age=2   age=3
# year=2010  N(10,0) N(10,1) N(10,2) N(10,3)
# year=2011  N(11,0) N(11,1) N(11,2) N(11,3)
# year=2012  N(12,0) N(12,1) N(12,2) N(12,3)
# So the code gives dis(10,0) = N(10,0) - N(11,1)

#stockt0_p = pd.DataFrame(stockt0).xs('ICEV', level='engine_type')['count'].unstack('car_age')
#stockt1_p = pd.DataFrame(stockt1).xs('ICEV', level='engine_type')['count'].unstack('car_age')

dis = stockt0 - stockt1
dis_rate = 1 - stockt1 / stockt0

# Conclusions: You can only really use the first 6 ages for BEVs and excluding the first one in year 0. 
# For ICEVs it is better but you still have to skip the first year

###########################################
# Infer new car sales + netto imports: ####
###########################################

inflow = -(stockt0 - stockt1)

# Reindex Reason: the stock at t1 was reindexed to t0.
# they are 0 year olds becoming 1 year olds. So I want them to register as 1 years olds not 0 years olds.
inflow = inflow.rename(index=lambda x: x + 1, level='car_age')

#drop isnan values
inflow = inflow.dropna()
inflow = inflow.drop(index=24, level='car_age')
# remove negative values (net exports) 
inflow = inflow[inflow > 0]
# only consider first 10 years
inflow = inflow[inflow.index.get_level_values('car_age') <= 10]
#inflow = inflow.rename('inflow')


##########################################
## Calculate market shares [0, A]:     ###
# Sales are inferred from new registrations in BIL51
# and from changes in the stock of BIL21.
A=5 # considers purchases of cars from age 0 up to age A.

# New registrations 
new_registrations = BIL51.groupby(['year', 'engine_type'], as_index=True)['count'].sum()

new_registrations = (new_registrations
    .reset_index()
    .assign(car_age=0)
    .set_index(['year', 'engine_type', 'car_age'])['count']
)

# Inflow of new (a=0) cars in addition to new registrations 
# and shifting the year ahead one year to align with the year of purchase (cars bought in year t are observed as inflow in year t+1)
# the numbers is quite wild
primo_stock_t1=stockt0.loc[:,:, 0].rename(index=lambda x: x - 1, level='year')
new_car_imports = primo_stock_t1 - new_registrations
new_car_imports = new_car_imports.reset_index()
new_car_imports['car_age'] = 0
new_car_imports = new_car_imports.set_index(['year', 'engine_type', 'car_age'])['count']
visualisation.plot_new_car_imports(new_registrations, new_car_imports)

# Inflow of cars from age 1 to A
from pandas import IndexSlice as idx
inflow_1_to_A = inflow.loc[idx[:, :, 1:A+1]]

# Now append the new_car_imports to the inflow of cars from age 1 to A. 
total_inflow = pd.concat([inflow_1_to_A, new_car_imports])

# To avoid complications I will only count 1-A stock changes into purchase probabilities.
# This means i'm not including the inflow of new cars not explained by new registrations in BIL51.

car_purchases = (
    pd.concat([inflow_1_to_A, new_registrations], axis=0)
    .sort_index()
)


# Constructing car_purchase market shares:

car_purchases_market_shares = (
    car_purchases
    .groupby(['year', 'engine_type', 'car_age']).sum()
    /denom_choice
)


##########################################
## Calculate holdings distribution     ###
##########################################
# For now I will model the distribution of holdings of the cars instead of the distribution of cars in the population. 

denom = BIL21.groupby('year', as_index=True)['count'].sum()
num = BIL21.groupby(['year', 'engine_type', 'car_age'], as_index=True)['count'].sum()

holdings_dist = num / denom_choice

engine_shares = BIL21.groupby(['year', 'engine_type'], as_index=True)['count'].sum() / denom
engine_shares_df = engine_shares.unstack('engine_type')

##########################################
## Generate all plots                  ###
##########################################
visualisation.run_all(dis_rate, holdings_dist, engine_shares_df, market_shares, ncpurch_prob, inflow, BIL51.reset_index(), BIL21)

##########################################
## Save processed data for later use   ###
## stored as a pickle file for easy loading in the future.         ###
##########################################

processed_data = {
    'disappearance_rate': dis_rate,
    'holdings_dist': holdings_dist,
    'engine_shares': engine_shares_df,
    'market_shares': market_shares,
    'ncpurch_prob': ncpurch_prob,
    'car_purchases_market_shares': car_purchases_market_shares,
}

with open('processed_data.pkl', 'wb') as f:
    pickle.dump(processed_data, f)

