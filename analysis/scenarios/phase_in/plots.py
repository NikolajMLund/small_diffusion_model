import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import IndexSlice as idx


def wrangle_kf_to_engine_types(df, engine_map):
    """
    Convert a KF DataFrame (year-indexed, wide format) to a Series
    with MultiIndex (year, engine_type), matching the structure of
    holdings_dist / car_purchases_market_shares summed over age.
    """
    return (
        df[list(engine_map.keys())]
        .rename(columns=engine_map)
        .T.groupby(level=0).sum().T
        .stack()
        .rename_axis(index=['year', 'engine_type'])
    )


def plot_kf_stock_total(
    historical_distributions_kf,   # (n_hist, n_engine_types)
    forecasted_distributions_kf,   # (n_forecast, n_engine_types)
    kf_historical_years,
    kf_forecast_years,
    forecasted_distributions,      # (n_periods, n_engine_types, n_ages)
    holdings_dist,                  # Series (year, engine_type, car_age)
    forecast_config,
    output_dir,
    file_name='kf_stock_total.png',
):
    forecast_years = np.arange(forecast_config.base_year + 1, forecast_config.target_year + 1)
    width = 0.35

    kf_hist_total     = historical_distributions_kf.sum(axis=1)
    kf_forecast_total = forecasted_distributions_kf.sum(axis=1)
    model_hist_total  = holdings_dist.groupby('year').sum()
    model_fc_total    = forecasted_distributions.sum(axis=(1, 2))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(kf_historical_years, kf_hist_total,
               color='tab:orange', zorder=3, label='KF 2025 (historical)')
    ax.scatter(model_hist_total.index, model_hist_total.values,
               color='tab:blue', zorder=3, marker='s', label='historical')
    ax.bar(kf_forecast_years - width / 2, kf_forecast_total, width=width,
           color='tab:orange', alpha=0.7, label='KF 2025 (forecast)')
    ax.bar(forecast_years + width / 2, model_fc_total, width=width,
           color='tab:blue', alpha=0.7, label='Model (forecast)')
    ax.axvline(forecast_config.base_year, color='grey', linestyle=':', linewidth=0.8)
    ax.set_xlabel('Year')
    ax.set_ylabel('Total stock (norm.)')
    ax.set_title('Total car stock — KF 2025 vs Model')
    ax.legend()
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, file_name), dpi=150, bbox_inches='tight')
    plt.close()


def plot_kf_stock_by_engine(
    historical_distributions_kf,   # (n_hist, n_engine_types)
    forecasted_distributions_kf,   # (n_forecast, n_engine_types)
    kf_historical_years,
    kf_forecast_years,
    forecasted_distributions,      # (n_periods, n_engine_types, n_ages)
    holdings_dist,                  # Series (year, engine_type, car_age)
    model_config,
    forecast_config,
    output_dir,
    file_name='kf_stock_by_engine.png',
):
    forecast_years = np.arange(forecast_config.base_year + 1, forecast_config.target_year + 1)
    engine_types = list(model_config.engine_types)
    width = 0.35

    model_hist_by_engine = holdings_dist.groupby(['year', 'engine_type']).sum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, engine_type in zip(axes, engine_types):
        et_idx = engine_types.index(engine_type)

        kf_hist_et    = historical_distributions_kf[:, et_idx]
        kf_forecast_et = forecasted_distributions_kf[:, et_idx]
        model_hist_et  = model_hist_by_engine.loc[idx[:, engine_type]]
        model_fc_et    = forecasted_distributions[:, et_idx, :].sum(axis=1)

        ax.scatter(kf_historical_years, kf_hist_et,
                   color='tab:orange', zorder=3, label='KF 2025 (historical)')
        ax.scatter(model_hist_et.index, model_hist_et.values,
                   color='tab:blue', zorder=3, marker='s', label='historical')
        ax.bar(kf_forecast_years - width / 2, kf_forecast_et, width=width,
               color='tab:orange', alpha=0.7, label='KF 2025 (forecast)')
        ax.bar(forecast_years + width / 2, model_fc_et, width=width,
               color='tab:blue', alpha=0.7, label='Model (forecast)')
        ax.axvline(forecast_config.base_year, color='grey', linestyle=':', linewidth=0.8)
        ax.set_xlabel('Year')
        ax.set_ylabel(f'{engine_type} stock (norm.)')
        ax.set_title(f'{engine_type} stock — KF 2025 vs Model')
        ax.legend()

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, file_name), dpi=150, bbox_inches='tight')
    plt.close()


def plot_kf_inflow(
    historical_inflows_kf,          # (n_hist, n_engine_types)
    projected_inflows_kf,           # (n_forecast, n_engine_types)
    kf_historical_years,
    kf_forecast_years,
    projected_inflows,              # (n_periods, n_engine_types, n_ages)
    car_purchases_market_shares,    # Series (year, engine_type, car_age)
    forecast_config,
    output_dir,
    file_name='kf_inflow.png',
):
    forecast_years = np.arange(forecast_config.base_year + 1, forecast_config.target_year + 1)
    width = 0.35

    kf_hist_total     = historical_inflows_kf.sum(axis=1)
    kf_forecast_total = projected_inflows_kf.sum(axis=1)
    model_hist_total  = car_purchases_market_shares.groupby('year').sum()
    model_fc_total    = projected_inflows.sum(axis=(1, 2))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(kf_historical_years, kf_hist_total,
               color='tab:orange', zorder=3, label='KF 2025 (historical)')
    ax.scatter(model_hist_total.index, model_hist_total.values,
               color='tab:blue', zorder=3, marker='s', label='Historical')
    ax.bar(kf_forecast_years - width / 2, kf_forecast_total, width=width,
           color='tab:orange', alpha=0.7, label='KF 2025 (forecast)')
    ax.bar(forecast_years + width / 2, model_fc_total, width=width,
           color='tab:blue', alpha=0.7, label='Model (forecast)')
    ax.axvline(forecast_config.base_year, color='grey', linestyle=':', linewidth=0.8)
    ax.set_xlabel('Year')
    ax.set_ylabel('Annual inflow (norm.)')
    ax.set_title('Annual car inflow — KF 2025 vs Model')
    ax.legend()
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, file_name), dpi=150, bbox_inches='tight')
    plt.close()


def plot_kf_inflow_by_engine(
    historical_inflows_kf,          # (n_hist, n_engine_types)
    projected_inflows_kf,           # (n_forecast, n_engine_types)
    kf_historical_years,
    kf_forecast_years,
    projected_inflows,              # (n_periods, n_engine_types, n_ages)
    car_purchases_market_shares,    # Series (year, engine_type, car_age)
    model_config,
    forecast_config,
    output_dir,
    file_name='kf_inflow_by_engine.png',
):
    forecast_years = np.arange(forecast_config.base_year + 1, forecast_config.target_year + 1)
    engine_types = list(model_config.engine_types)
    width = 0.35

    model_hist_by_engine = car_purchases_market_shares.groupby(['year', 'engine_type']).sum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, engine_type in zip(axes, engine_types):
        et_idx = engine_types.index(engine_type)

        kf_hist_et     = historical_inflows_kf[:, et_idx]
        kf_forecast_et = projected_inflows_kf[:, et_idx]
        model_hist_et  = model_hist_by_engine.loc[idx[:, engine_type]]
        model_fc_et    = projected_inflows[:, et_idx, :].sum(axis=1)

        ax.scatter(kf_historical_years, kf_hist_et,
                   color='tab:orange', zorder=3, label='KF 2025 (historical)')
        ax.scatter(model_hist_et.index, model_hist_et.values,
                   color='tab:blue', zorder=3, marker='s', label='historical')
        ax.bar(kf_forecast_years - width / 2, kf_forecast_et, width=width,
               color='tab:orange', alpha=0.7, label='KF 2025 (forecast)')
        ax.bar(forecast_years + width / 2, model_fc_et, width=width,
               color='tab:blue', alpha=0.7, label='Model (forecast)')
        ax.axvline(forecast_config.base_year, color='grey', linestyle=':', linewidth=0.8)
        ax.set_xlabel('Year')
        ax.set_ylabel(f'{engine_type} inflow (norm.)')
        ax.set_title(f'{engine_type} inflow — KF 2025 vs Model')
        ax.legend()

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, file_name), dpi=150, bbox_inches='tight')
    plt.close()


def plot_kf_stock_difference(
    historical_distributions_kf,   # (n_hist, n_engine_types)
    forecasted_distributions_kf,   # (n_forecast, n_engine_types)
    kf_historical_years,
    kf_forecast_years,
    forecasted_distributions,      # (n_periods, n_engine_types, n_ages) — model
    holdings_dist,                  # Series (year, engine_type, car_age) — model historical
    denom_choice,
    model_config,
    forecast_config,
    output_dir,
    file_name='kf_stock_difference.png',
):
    forecast_years = np.arange(forecast_config.base_year + 1, forecast_config.target_year + 1)
    engine_types = list(model_config.engine_types)
    model_hist_by_engine = holdings_dist.groupby(['year', 'engine_type']).sum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, engine_type in zip(axes, engine_types):
        et_idx = engine_types.index(engine_type)

        # Historical difference
        model_hist_et = model_hist_by_engine.loc[idx[:, engine_type]]
        hist_common   = np.intersect1d(kf_historical_years, model_hist_et.index)
        kf_hist_abs   = historical_distributions_kf[np.isin(kf_historical_years, hist_common), et_idx] * denom_choice
        model_hist_abs = model_hist_et.loc[hist_common].values * denom_choice
        hist_diff = kf_hist_abs - model_hist_abs

        # Forecast difference
        common_fc   = np.intersect1d(kf_forecast_years, forecast_years)
        kf_fc_abs   = forecasted_distributions_kf[np.isin(kf_forecast_years, common_fc), et_idx] * denom_choice
        model_fc_abs = forecasted_distributions[np.isin(forecast_years, common_fc), et_idx, :].sum(axis=1) * denom_choice
        fc_diff = kf_fc_abs - model_fc_abs

        ax.bar(hist_common, hist_diff, width=0.7,
               color=np.where(hist_diff >= 0, 'tab:orange', 'tab:blue'), alpha=0.5)
        ax.bar(common_fc,   fc_diff,   width=0.7,
               color=np.where(fc_diff   >= 0, 'tab:orange', 'tab:blue'))
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axvline(forecast_config.base_year, color='grey', linestyle=':', linewidth=0.8)
        ax.set_xlabel('Year')
        ax.set_ylabel('Difference in cars (KF − Model)')
        ax.set_title(
            f'{engine_type} stock difference — KF 2025 minus Model\n'
            f'Note: bars at/before {forecast_config.base_year} show KF 2025 minus Statistics Denmark (observed vs observed)'
        )

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, file_name), dpi=150, bbox_inches='tight')
    plt.close()


def plot_kf_stock_difference_total(
    historical_distributions_kf,   # (n_hist, n_engine_types)
    forecasted_distributions_kf,   # (n_forecast, n_engine_types)
    kf_historical_years,
    kf_forecast_years,
    forecasted_distributions,      # (n_periods, n_engine_types, n_ages) — model
    holdings_dist,                  # Series (year, engine_type, car_age) — model historical
    denom_choice,
    forecast_config,
    output_dir,
    file_name='kf_stock_difference_total.png',
):
    forecast_years = np.arange(forecast_config.base_year + 1, forecast_config.target_year + 1)
    model_hist_total = holdings_dist.groupby('year').sum()

    # Historical difference
    hist_common    = np.intersect1d(kf_historical_years, model_hist_total.index)
    kf_hist_abs    = historical_distributions_kf[np.isin(kf_historical_years, hist_common)].sum(axis=1) * denom_choice
    model_hist_abs = model_hist_total.loc[hist_common].values * denom_choice
    hist_diff = kf_hist_abs - model_hist_abs

    # Forecast difference
    common_fc    = np.intersect1d(kf_forecast_years, forecast_years)
    kf_fc_abs    = forecasted_distributions_kf[np.isin(kf_forecast_years, common_fc)].sum(axis=1) * denom_choice
    model_fc_abs = forecasted_distributions[np.isin(forecast_years, common_fc)].sum(axis=(1, 2)) * denom_choice
    fc_diff = kf_fc_abs - model_fc_abs

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(hist_common, hist_diff, width=0.7,
           color=np.where(hist_diff >= 0, 'tab:orange', 'tab:blue'), alpha=0.5)
    ax.bar(common_fc,   fc_diff,   width=0.7,
           color=np.where(fc_diff   >= 0, 'tab:orange', 'tab:blue'))
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(forecast_config.base_year, color='grey', linestyle=':', linewidth=0.8)
    ax.set_xlabel('Year')
    ax.set_ylabel('Difference in cars (KF − Model)')
    ax.set_title(
        'Total stock difference — KF 2025 minus Model\n'
        f'Note: bars at/before {forecast_config.base_year} show KF 2025 minus Statistics Denmark (observed vs observed)'
    )
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, file_name), dpi=150, bbox_inches='tight')
    plt.close()


def plot_total_regression_fit(
    historical_years, y, regression_fit,
    projection_years, sales_projection, actual_car_sales,
    base_year, output_dir,
):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots()
    ax.scatter(actual_car_sales.index.values, actual_car_sales.values, label='Raw data', zorder=3)
    ax.plot(historical_years, regression_fit, label='Regression fit')
    ax.plot(projection_years, sales_projection, linestyle='--', label='Projection')
    ax.scatter(projection_years, sales_projection, zorder=3)
    ax.axvline(base_year, color='grey', linestyle=':', linewidth=0.8)
    ax.set_xlabel('Year')
    ax.set_ylabel('New car sales (share of 2020 households*2)')
    ax.set_title('Total new car sales: regression fit and projection\n'
                 'Includes new registrations + implied imports (age 1–6 stock changes); excludes exports')
    ax.legend()
    fig.savefig(os.path.join(output_dir, 'total_regression_fit.png'), dpi=150, bbox_inches='tight')
    plt.close()


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
    plt.close()

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
    plt.close()


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
    # ax.axvline(base_year, color='grey', linestyle=':', linewidth=0.8)
    # ax.annotate(
    #     f'{base_year}: imports\nnot yet observed',
    #     xy=(base_year, bev_share_full.loc[base_year]),
    #     xytext=(base_year - 1.5, bev_share_full.loc[base_year] + 0.02),
    #     arrowprops=dict(arrowstyle='->', color='grey'),
    #     fontsize=8, color='grey',
    # )

    ax.set_xlabel('Year')
    ax.set_ylabel('Engine type share of purchases incl. imports')
    ax.set_title('BEV/ICEV share of car purchases incl. imports.')
    ax.legend()
    fig.savefig(os.path.join(output_dir, 'engine_share_over_time.png'), dpi=150, bbox_inches='tight')
    plt.close()


