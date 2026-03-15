import matplotlib.pyplot as plt
import pandas as pd
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'analysis', 'plots')


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_disappearance_rate(dis_rate, engine_type, n_years=6, trim_end=3, output_dir=OUTPUT_DIR):
    _ensure_dir(output_dir)
    plt.figure()
    years = dis_rate.index.get_level_values('year').unique()[-n_years:]
    for year in years:
        plt.plot(dis_rate.loc[year, engine_type, :][0:-trim_end], label=f'{year}')
    plt.xlabel('Car age')
    plt.ylabel(f'Disappearance rate - {engine_type}s')
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'disappearance_rate_by_age_{engine_type}.png'))


def plot_holdings_by_age(holdings_dist, engine_type, n_years=6, output_dir=OUTPUT_DIR):
    _ensure_dir(output_dir)
    plt.figure()
    years = holdings_dist.index.get_level_values('year').unique()[-n_years:]
    for year in years:
        try:
            data = holdings_dist.loc[year, engine_type, :]
            plt.plot(data.index, data.values, label=f'{year}')
        except KeyError:
            pass
    plt.xlabel('Car age')
    plt.ylabel(f'Holdings share - {engine_type}s')
    plt.title(f'Holdings distribution by car age ({engine_type})')
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'holdings_dist_by_age_{engine_type}.png'))


def plot_holdings_by_engine_type(engine_shares_df, output_dir=OUTPUT_DIR):
    _ensure_dir(output_dir)
    plt.figure()
    engine_shares_df.plot(kind='bar', stacked=True)
    plt.xlabel('Year')
    plt.ylabel('Share of total stock')
    plt.title('Holdings share by engine type')
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'holdings_dist_by_engine_type.png'))


def plot_market_shares(market_shares, output_dir=OUTPUT_DIR):
    _ensure_dir(output_dir)
    plt.figure()
    ms_df = market_shares.reset_index()
    ms_df.columns = ['year', 'owner_type', 'engine_type', 'market_share']
    for engine in ms_df['engine_type'].unique():
        subset = ms_df[ms_df['engine_type'] == engine]
        plt.plot(subset['year'], subset['market_share'], marker='o', label=engine)
    plt.xlabel('Year')
    plt.ylabel('Market share')
    plt.title('New car registration market shares (Households)')
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'market_shares_over_time.png'))


def plot_purchase_probability(ncpurch_prob, output_dir=OUTPUT_DIR):
    _ensure_dir(output_dir)
    plt.figure()
    ncp_df = ncpurch_prob.reset_index()
    ncp_df.columns = ['year', 'owner_type', 'purchase_prob']
    ncp_df = ncp_df.dropna(subset=['purchase_prob'])
    plt.plot(ncp_df['year'], ncp_df['purchase_prob'], marker='o', label='Households')
    plt.xlabel('Year')
    plt.ylabel('Purchase probability')
    plt.title('Probability of purchasing a new car (Households)')
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'purchase_probability_over_time.png'))


def plot_sales_by_age(inflow, engine_type, n_years=6, output_dir=OUTPUT_DIR):
    _ensure_dir(output_dir)
    _, ax = plt.subplots()
    years = inflow.index.get_level_values('year').unique()[-n_years:]
    for year in years:
        try:
            data = inflow.loc[year, engine_type, :]
            ax.plot(data.index, data.values, marker='o', label=f'{year}')
        except KeyError:
            pass
    plt.xlabel('Car age')
    plt.ylabel('Car inflow')
    plt.title(f'Car inflow by car age ({engine_type}, ages 0–10)')
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'car_inflow_by_age_{engine_type}.png'))


def plot_sales_by_engine_type(inflow, new_registrations, n_years=6, output_dir=OUTPUT_DIR):
    _ensure_dir(output_dir)
    sales_by_year_engine = inflow.groupby(['year', 'engine_type']).sum().unstack('engine_type')
    if isinstance(sales_by_year_engine.columns, pd.MultiIndex):
        sales_by_year_engine.columns = sales_by_year_engine.columns.droplevel(0)
    last_years = sales_by_year_engine.index[-n_years:]
    subset = sales_by_year_engine.loc[last_years]

    ax = subset.plot(kind='bar', stacked=False, figsize=(10, 5))

    # Map each engine type to its bar colour (first patch per engine)
    n_years_plot = len(last_years)
    n_engines = len(subset.columns)
    engine_colors = {col: ax.patches[i * n_years_plot].get_facecolor()
                     for i, col in enumerate(subset.columns)}

    # Overlay new registrations from BIL51 as dots at the same x-position as each bar
    reg_by_engine = new_registrations.groupby(['year', 'engine_type'])['count'].sum().unstack('engine_type')
    reg_subset = reg_by_engine.reindex(last_years)
    bar_width = 0.8 / n_engines
    for i, engine in enumerate(subset.columns):
        if engine not in reg_subset.columns:
            continue
        x_positions = [pos - 0.4 + bar_width * (i + 0.5) for pos in range(n_years_plot)]
        ax.scatter(x_positions, reg_subset[engine].values,
                   color=engine_colors[engine], edgecolors='black',
                   zorder=5, s=80, label=f'{engine} new reg. (BIL51)')

    plt.xlabel('Year')
    plt.ylabel('Car inflow')
    plt.title('Car inflow by engine type per year (ages 0–10)')
    plt.tight_layout()
    plt.legend(title='Engine type')
    plt.savefig(os.path.join(output_dir, 'car_inflow_by_engine_type.png'))


def plot_stock_inflow_stock(BIL21, new_registrations, engine_type, n_years=6, output_dir=OUTPUT_DIR):
    import numpy as np
    _ensure_dir(output_dir)

    stock_a0_t = BIL21.loc[(slice(None), engine_type, 0), 'count'].copy()
    stock_a0_t.index = stock_a0_t.index.get_level_values('year')           # N(t, a=0)

    stock_a1_t1 = BIL21.loc[(slice(None), engine_type, 1), 'count'].copy()
    stock_a1_t1.index = stock_a1_t1.index.get_level_values('year') - 1    # N(t+1, a=1) indexed by t

    stock_a0_t1 = BIL21.loc[(slice(None), engine_type, 0), 'count'].copy()
    stock_a0_t1.index = stock_a0_t1.index.get_level_values('year') - 1    # N(t+1, a=0) indexed by t

    reg = (new_registrations[new_registrations['engine_type'] == engine_type]
           .groupby('year')['count'].sum())                                # BIL51(t)

    common_years = sorted(
        set(stock_a0_t.index) & set(stock_a1_t1.index) & set(stock_a0_t1.index) & set(reg.index)
    )[-n_years:]

    stock_a0_t  = stock_a0_t.loc[common_years]
    stock_a1_t1 = stock_a1_t1.loc[common_years]
    stock_a0_t1 = stock_a0_t1.loc[common_years]
    reg         = reg.loc[common_years]

    x = np.arange(len(common_years))
    width = 0.35
    _, ax = plt.subplots(figsize=(10, 5))

    # Left stack: N(t, a=0) + new reg
    ax.bar(x - 0.2, stock_a0_t.values, width=width, label='Stock a=0 at t', color='steelblue')
    ax.bar(x - 0.2, reg.values, width=width, bottom=stock_a0_t.values, label='New reg. t (BIL51)', color='darkorange')

    # Right stack: N(t+1, a=1) + N(t+1, a=0)
    ax.bar(x + 0.2, stock_a1_t1.values, width=width, label='Stock a=1 at t+1', color='mediumseagreen')
    ax.bar(x + 0.2, stock_a0_t1.values, width=width, bottom=stock_a1_t1.values, label='Stock a=0 at t+1', color='lightblue')

    ax.set_xticks(x)
    ax.set_xticklabels(common_years)
    ax.set_xlabel('Year')
    ax.set_ylabel('Count')
    ax.set_title(f'Stock consistency check ({engine_type})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'stock_inflow_stock_{engine_type}.png'))


def plot_cohort_survival(BIL21, year, ages=range(1, 11), output_dir=OUTPUT_DIR):
    import numpy as np
    _ensure_dir(output_dir)

    fig, (ax_bev, ax_icev) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for ax, engine_type in [(ax_bev, 'BEV'), (ax_icev, 'ICEV')]:
        stock_t_vals  = []
        stock_t1_vals = []
        net_vals      = []

        for a in ages:
            try:
                s_t  = BIL21.loc[(year,   engine_type, a),   'count']
                s_t1 = BIL21.loc[(year+1, engine_type, a+1), 'count']
                stock_t_vals.append(s_t)
                stock_t1_vals.append(s_t1)
                net_vals.append(s_t1 - s_t)
            except KeyError:
                stock_t_vals.append(np.nan)
                stock_t1_vals.append(np.nan)
                net_vals.append(np.nan)

        x = np.arange(len(list(ages)))
        width = 0.25
        ax.bar(x - width, stock_t_vals,  width=width, label=f'N(t={year}, a)',   color='steelblue')
        ax.bar(x,         stock_t1_vals, width=width, label=f'N(t={year+1}, a+1)', color='mediumseagreen')
        ax.bar(x + width, net_vals,      width=width, label='Net change',         color='darkorange')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(list(ages))
        ax.set_ylabel('Count')
        ax.set_title(f'{engine_type}: cohort survival {year}→{year+1}')
        ax.legend()

    ax_icev.set_xlabel('Car age at year t')
    fig.suptitle(f'Cohort survival by age, {year}→{year+1}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cohort_survival_{year}.png'))
    plt.close(fig)


def run_all(dis_rate, holdings_dist, engine_shares_df, market_shares, ncpurch_prob, inflow, new_registrations, BIL21, year=None, output_dir=OUTPUT_DIR):
    """Generate all standard plots."""
    if year is None:
        year = BIL21.index.get_level_values('year').max() - 1
    plot_disappearance_rate(dis_rate, 'BEV', output_dir=output_dir)
    plot_disappearance_rate(dis_rate, 'ICEV', output_dir=output_dir)
    plot_holdings_by_age(holdings_dist, 'BEV', output_dir=output_dir)
    plot_holdings_by_age(holdings_dist, 'ICEV', output_dir=output_dir)
    plot_holdings_by_engine_type(engine_shares_df, output_dir=output_dir)
    plot_market_shares(market_shares, output_dir=output_dir)
    plot_purchase_probability(ncpurch_prob, output_dir=output_dir)
    plot_sales_by_age(inflow, 'BEV', output_dir=output_dir)
    plot_sales_by_age(inflow, 'ICEV', output_dir=output_dir)
    plot_sales_by_engine_type(inflow, new_registrations, output_dir=output_dir)
    plot_stock_inflow_stock(BIL21, new_registrations, 'BEV', output_dir=output_dir)
    plot_stock_inflow_stock(BIL21, new_registrations, 'ICEV', output_dir=output_dir)
    plot_cohort_survival(BIL21, year, output_dir=output_dir)
    print(f"All plots saved to {os.path.abspath(output_dir)}")
