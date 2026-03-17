import pickle
import pandas as pd
from pandas import IndexSlice as idx
import numpy as np
import statsmodels.api as sm
import visualisation

# import data 
with open('processed_data.pkl', 'rb') as f:
    processed_data = pickle.load(f)

dis_full=processed_data['disappearance_rate']


# Focus on ICEVs and ages above 7
car_type, age_floor = 'ICEV', 7

dis=dis_full.loc[idx[:, car_type, age_floor:]]

dis = dis.loc[(dis<1) & (dis>0)]

# Now build a regression model that can accomodate this profile.

# Y: log(dis_rate) - log(1 - dis_rate)
# X:
## X_1: age
## X_2: 1(a>=4 and a even)

reg_data = dis.dropna().reset_index()
reg_data.columns = [*reg_data.columns[:-1], 'dis_rate']

reg_data['Y'] = np.log(reg_data['dis_rate']) - np.log(1 - reg_data['dis_rate'])
reg_data['inspection_year'] = ((reg_data['car_age'] >= 4) & (reg_data['car_age'] % 2 == 0)).astype(int)
reg_data['car_age_sq'] = reg_data['car_age']**2

X = sm.add_constant(reg_data[['car_age', 'inspection_year']])
y = reg_data['Y']

model = sm.OLS(y, X).fit()
print(model.summary())


# make a prediction scrap-"profile"

max_age = reg_data['car_age'].max()
pred_ages = np.arange(0, max_age + 2)


pred_df = pd.DataFrame({
    'car_age': pred_ages,
    'car_age_sq': pred_ages**2,
    'inspection_year': ((pred_ages >= 4) & (pred_ages % 2 == 0)).astype(int),
})
X_pred = sm.add_constant(pred_df[['car_age', 'inspection_year']], has_constant='add')
logit_pred = model.predict(X_pred)
scrap_profile = pd.Series(
    1 / (1 + np.exp(-logit_pred.values)),
    index=pred_ages,
    name='scrap_rate',
)
scrap_profile.index.name = 'car_age'

# add forced scrappage at age max_age+1: (it's a pandas dataframe)

scrap_profile.loc[max_age + 2] = 1.0

# store this prediction in processed data
processed_data['scrap_profile'] = scrap_profile
with open('processed_data.pkl', 'wb') as f:
    pickle.dump(processed_data, f)
print(scrap_profile)

visualisation.plot_scrap_profile(dis_full.loc[(dis_full<1) & (dis_full>0)], scrap_profile, car_type, age_floor)
