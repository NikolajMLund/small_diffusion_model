import sys
sys.path.insert(0, "src")
import numpy as np
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

# BIL8 is primo (1 Jan year Y) — a stock measure.
# GDP is a flow over year Y. Shift BIL forward by 1 so primo-Y stock
# is compared against GDP of year Y (i.e. the year the cars were driven).
BIL.index = BIL.index + 1

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

# Scatter: bil/cap vs bnp/cap, annotated by year
for xscale, yscale, fname in [
    ('linear', 'linear', 'fleetsize_scatter.png'),
    ('log',    'linear', 'fleetsize_scatter_log_x.png'),
    ('linear', 'log',    'fleetsize_scatter_log_y.png'),
    ('log',    'log',    'fleetsize_scatter_log_log.png'),
]:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['bnp/cap'], df['bil/cap'], s=20)
    for year, row in df.iterrows():
        ax.annotate(str(year), (row['bnp/cap'], row['bil/cap']), fontsize=6, alpha=0.7)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel('BNP per capita (DKK)')
    ax.set_ylabel('Biler per capita')
    ax.set_title('Biler per capita vs BNP per capita')
    plt.tight_layout()
    plt.savefig(f'analysis/assorted/{fname}', dpi=150)
    plt.close()

# First differences scatter: Δbil/cap vs Δbnp/cap
df_diff = df[['bnp/cap', 'bil/cap']].diff().dropna()

# Log first differences scatter: Δlog(bil/cap) vs Δlog(bnp/cap) ≈ growth rates
df_ldiff = df[['bnp/cap', 'bil/cap']].apply(np.log).diff().dropna()

# Colour by decade to reveal regime shifts
ldiff_years = df_ldiff.index.astype(int)
decade_colors = {1980: 'tab:blue', 1990: 'tab:orange', 2000: 'tab:green',
                 2010: 'tab:red',  2020: 'tab:purple'}
point_colors = [decade_colors[(int(y) // 10) * 10] for y in ldiff_years]

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df_ldiff['bnp/cap'], df_ldiff['bil/cap'], c=point_colors, s=30, zorder=3)
for year, row in df_ldiff.iterrows():
    ax.annotate(str(year), (row['bnp/cap'], row['bil/cap']), fontsize=6, alpha=0.7)
ax.axhline(0, color='grey', linewidth=0.8, linestyle=':')
ax.axvline(0, color='grey', linewidth=0.8, linestyle=':')
ax.set_xlabel('Δ log(BNP per capita)')
ax.set_ylabel('Δ log(Biler per capita)')
ax.set_title('Log first differences: growth rates of biler/cap vs bnp/cap')
for decade, color in decade_colors.items():
    ax.scatter([], [], c=color, label=f'{decade}s')
ax.legend(title='Decade')
plt.tight_layout()
plt.savefig('analysis/assorted/fleetsize_scatter_ldiff.png', dpi=150)
plt.close()

# Log-differenced series against time
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
axes[0].bar(ldiff_years, df_ldiff['bnp/cap'], color=point_colors, width=0.8)
axes[0].axhline(0, color='grey', linewidth=0.8, linestyle=':')
axes[0].set_ylabel('Δ log(BNP per capita)')
axes[0].set_title('Log first differences over time')
axes[1].bar(ldiff_years, df_ldiff['bil/cap'], color=point_colors, width=0.8)
axes[1].axhline(0, color='grey', linewidth=0.8, linestyle=':')
axes[1].set_ylabel('Δ log(Biler per capita)')
axes[1].set_xlabel('År')
plt.tight_layout()
plt.savefig('analysis/assorted/fleetsize_ldiff_timeseries.png', dpi=150)
plt.close()

for xscale, yscale, fname in [
    ('linear', 'linear', 'fleetsize_scatter_diff.png'),
    ('symlog', 'linear', 'fleetsize_scatter_diff_log_x.png'),
    ('linear', 'symlog', 'fleetsize_scatter_diff_log_y.png'),
    ('symlog', 'symlog', 'fleetsize_scatter_diff_log_log.png'),
]:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df_diff['bnp/cap'], df_diff['bil/cap'], s=20)
    for year, row in df_diff.iterrows():
        ax.annotate(str(year), (row['bnp/cap'], row['bil/cap']), fontsize=6, alpha=0.7)
    ax.axhline(0, color='grey', linewidth=0.8, linestyle=':')
    ax.axvline(0, color='grey', linewidth=0.8, linestyle=':')
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel('Δ BNP per capita (DKK)')
    ax.set_ylabel('Δ Biler per capita')
    ax.set_title('First differences: Δbiler/cap vs Δbnp/cap')
    plt.tight_layout()
    plt.savefig(f'analysis/assorted/{fname}', dpi=150)
    plt.close()

