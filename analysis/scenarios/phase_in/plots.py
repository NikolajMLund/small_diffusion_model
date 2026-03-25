import os

import matplotlib.pyplot as plt
import numpy as np
from pandas import IndexSlice as idx


def plot_total_sales_forecast(
    historical_years, y, sales_forecast,
    projection_years, sales_projection, actual_car_sales,
    base_year, output_dir,
):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots()
    ax.scatter(actual_car_sales.index.values, actual_car_sales.values, label='Raw data', zorder=3)
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


def plot_bev_diffusion_fit(
    all_years, bev_fit_reg, bev_fit_total,
    bev_share_reg, bev_share_total,
    base_year, data_limit_year, saturation, output_dir, file_name='bev_diffusion_fit.png'
):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots()

    # Observed data
    ax.scatter(bev_share_reg.index, bev_share_reg.values,
               label='Observed — new reg only', zorder=3, marker='s', color='tab:blue')
    ax.scatter(bev_share_total.index, bev_share_total.values,
               label='Observed — incl. imports', zorder=3, marker='o', color='tab:orange')

    # Fitted S-curves
    hist_mask = all_years <= data_limit_year
    proj_mask = all_years >= data_limit_year
    ax.plot(all_years[hist_mask], bev_fit_reg[hist_mask], color='tab:blue', linewidth=1.5)
    ax.plot(all_years[proj_mask], bev_fit_reg[proj_mask], color='tab:blue', linewidth=1.5,
            linestyle='--', label='Logistic fit — new reg only')
    ax.plot(all_years[hist_mask], bev_fit_total[hist_mask], color='tab:orange', linewidth=1.5)
    ax.plot(all_years[proj_mask], bev_fit_total[proj_mask], color='tab:orange', linewidth=1.5,
            linestyle='--', label='Logistic fit — incl. imports')

    # Saturation line and base-year marker
    ax.axhline(saturation, color='grey', linestyle=':', linewidth=0.8,
               label=f'Saturation = {saturation:.0%}')
    #ax.axvline(base_year, color='grey', linestyle=':', linewidth=0.8)
    ax.axvline(base_year, color='grey', linestyle=':', linewidth=0.8)

    ax.set_xlabel('Year')
    ax.set_ylabel('BEV share of purchases')
    ax.set_title('BEV adoption — logistic diffusion fit\n'
                 'Solid: historical fit | Dashed: projection')
    ax.set_ylim(0, 1)
    ax.legend()
    fig.savefig(os.path.join(output_dir, file_name), dpi=150, bbox_inches='tight')
    plt.show()

def plot_age_distribution_of_inflows(car_purchases_market_shares, base_year, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Filter to imports (car_age >= 1) and sum over any remaining dimensions
    imports = car_purchases_market_shares[
        car_purchases_market_shares.index.get_level_values('car_age') >= 1
    ]
    by_year_engine_age = imports.groupby(['year', 'engine_type', 'car_age']).sum()

    engine_types = ['BEV', 'ICEV']
    cmap = plt.get_cmap('tab10')
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for ax, engine in zip(axes, engine_types):
        subset = by_year_engine_age.loc[idx[:, engine, :]]
        pivot = subset.unstack('car_age').fillna(0)
        pivot.index = pivot.index.get_level_values('year')
        years = pivot.index.values
        ages = pivot.columns.get_level_values('car_age')

        bottom = np.zeros(len(years))
        for age in ages:
            values = pivot[age].values
            ax.bar(years, values, bottom=bottom, color=cmap(age), label=f'Age {age}')
            bottom += values

        ax.axvline(base_year, color='grey', linestyle=':', linewidth=0.8)
        ax.set_title(engine)
        ax.set_ylabel('Market share')

    axes[-1].set_xlabel('Year')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, 0))
    fig.suptitle('Age distribution of implied car imports (car_age ≥ 1)\n'
                 'Inferred from stock changes; excludes exports')
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(output_dir, 'age_dist_imports.png'), dpi=150, bbox_inches='tight')
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
    bev_share_reg = by_year_engine_reg.loc[idx[:, 'BEV']] / total_by_year
    icev_share_reg = by_year_engine_reg.loc[idx[:, 'ICEV']] / total_by_year

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
