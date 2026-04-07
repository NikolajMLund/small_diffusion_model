import sys
sys.path.insert(0, "src")
import pandas as pd
import matplotlib.pyplot as plt

from data_import import import_NAHL2, import_BEFOLK1, import_BIL8

NAHL2   = import_NAHL2()
BEFOLK1 = import_BEFOLK1()
BIL8    = import_BIL8()



BNP=NAHL2[['TID', 'INDHOLD']].set_index('TID')
BNP['INDHOLD'] = pd.to_numeric(BNP['INDHOLD'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False))
BNP['INDHOLD'] = BNP['INDHOLD']#/1e6

BEF=BEFOLK1[['TID','INDHOLD']].set_index('TID')
BEF['INDHOLD'] = pd.to_numeric(BEF['INDHOLD'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False))

BIL=BIL8[['TID','INDHOLD']].set_index('TID')
BIL['INDHOLD'] = pd.to_numeric(BIL['INDHOLD'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False))

df = pd.concat([BNP,BEF,BIL],axis=1)
df.columns = ['BNP', 'BEF', 'BIL']


df['bnp/cap'] = df['BNP']/df['BEF'] 
df['bil/cap'] = df['BIL']/df['BEF']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

years = df.index.astype(int)

ax1.plot(years, df['bnp/cap'])
ax1.scatter(years, df['bnp/cap'], s=20)
for year, val in zip(years, df['bnp/cap']):
    ax1.annotate(f'{val:.0f}', (year, val), fontsize=6, alpha=0.7)
ax1.set_xlabel('År')
ax1.set_ylabel('BNP per capita (DKK)')
ax1.set_title('BNP per capita')

ax2.plot(years, df['bil/cap'])
ax2.scatter(years, df['bil/cap'], s=20)
for year, val in zip(years, df['bil/cap']):
    ax2.annotate(f'{val:.3f}', (year, val), fontsize=6, alpha=0.7)
ax2.set_xlabel('År')
ax2.set_ylabel('Biler per capita')
ax2.set_title('Biler per capita')

plt.tight_layout()
plt.savefig('analysis/assorted/fleetsize_timeseries.png', dpi=150)
plt.close()

breakpoint()
