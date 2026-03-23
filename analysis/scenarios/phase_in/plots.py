import os

import matplotlib.pyplot as plt
import numpy as np
from pandas import IndexSlice as idx


def plot_total_sales_forecast(
    historical_years, y, sales_forecast,
    projection_years, sales_projection,
    base_year, output_dir,
):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots()
    ax.scatter(historical_years, y, label='Raw data', zorder=3)
    ax.plot(historical_years, sales_forecast, label='Regression fit')
    ax.plot(projection_years, sales_projection, linestyle='--', label='Projection')
    ax.scatter(projection_years, sales_projection, zorder=3)
    ax.axvline(base_year, color='grey', linestyle=':', linewidth=0.8)
    ax.set_xlabel('Year')
    ax.set_ylabel('New car sales (share of 2020 households)')
    ax.set_title('Total new car sales: regression fit and projection\n'
                 'Includes new registrations + implied imports (age 1–6 stock changes); excludes exports')
    ax.legend()
    fig.savefig(os.path.join(output_dir, 'total_sales_forecast.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_engine_share_over_time(market_shares, new_reg_market_shares, base_year, output_dir):
    """
    Plots BEV and ICEV shares of purchases over time, comparing:
      - Full series: new registrations + implied imports (car_age 1-6 stock changes)
      - New registrations only: from data['market_shares'] (BIL51 direct, no 0-age imports)

    Notes:
      - age>=1 inflows are inferred from stock changes — they capture imports but NOT exports
      - The last observed year (base_year) is missing age>=1 data since t+1 stock is not yet
        observed, making totals appear artificially low
      - new_reg_market_shares uses BIL51 directly and excludes 0-age car imports
    """
    os.makedirs(output_dir, exist_ok=True)

    # Full series (all ages): sum over age, then compute shares per year
    by_year_engine = market_shares.groupby(['year', 'engine_type']).sum()
    total_by_year = by_year_engine.groupby('year').sum()
    bev_share_full = by_year_engine.loc[idx[:, 'BEV']] / total_by_year
    icev_share_full = by_year_engine.loc[idx[:, 'ICEV']] / total_by_year

    # New registrations only: sum over owner_type, then compute shares per year
    by_year_engine_reg = new_reg_market_shares.groupby(['year', 'engine_type']).sum()
    total_by_year_reg = by_year_engine_reg.groupby('year').sum()
    bev_share_reg = by_year_engine_reg.loc[idx[:, 'BEV']] / total_by_year_reg
    icev_share_reg = by_year_engine_reg.loc[idx[:, 'ICEV']] / total_by_year_reg

    bev_color = 'tab:blue'
    icev_color = 'tab:orange'

    fig, ax = plt.subplots()
    ax.plot(bev_share_full.index, bev_share_full.values, marker='o', color=bev_color,
            label='BEV — incl. implied imports')
    ax.plot(icev_share_full.index, icev_share_full.values, marker='o', color=icev_color,
            label='ICEV — incl. implied imports')
    ax.plot(bev_share_reg.index, bev_share_reg.values, marker='s', linestyle='--', color=bev_color,
            label='BEV — new registrations only (BIL51)')
    ax.plot(icev_share_reg.index, icev_share_reg.values, marker='s', linestyle='--', color=icev_color,
            label='ICEV — new registrations only (BIL51)')

    # Annotate last observed year
    ax.axvline(base_year, color='grey', linestyle=':', linewidth=0.8)
    ax.annotate(
        f'{base_year}: imports\nnot yet observed',
        xy=(base_year, bev_share_full.loc[base_year]),
        xytext=(base_year - 1.5, bev_share_full.loc[base_year] + 0.02),
        arrowprops=dict(arrowstyle='->', color='grey'),
        fontsize=8, color='grey',
    )

    ax.set_xlabel('Year')
    ax.set_ylabel('Engine type share of purchases')
    ax.set_title('BEV/ICEV share of car purchases\n'
                 'Solid: incl. implied imports (age 1–6 stock changes, excl. exports) | '
                 'Dashed: new registrations only (BIL51, no 0-age imports)')
    ax.legend()
    fig.savefig(os.path.join(output_dir, 'engine_share_over_time.png'), dpi=150, bbox_inches='tight')
    plt.show()
