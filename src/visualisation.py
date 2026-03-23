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


def plot_stock_inflow_stock(BIL21, new_registrations, engine_type, n_years=6, years=None, output_dir=OUTPUT_DIR):
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

    all_common = sorted(
        set(stock_a0_t.index) & set(stock_a1_t1.index) & set(stock_a0_t1.index) & set(reg.index)
    )
    if years is not None:
        common_years = [y for y in all_common if y in years]
    else:
        common_years = all_common[-n_years:]

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
    suffix = f'_{min(years)}-{max(years)}' if years is not None else ''
    plt.savefig(os.path.join(output_dir, f'stock_inflow_stock_{engine_type}{suffix}.png'))


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


def plot_new_car_imports(new_registrations, new_car_imports, output_dir=OUTPUT_DIR):
    import numpy as np
    _ensure_dir(output_dir)

    reg = new_registrations.groupby(['year', 'engine_type']).sum()
    imp = new_car_imports.groupby(['year', 'engine_type']).sum()

    engine_types = ['BEV', 'ICEV']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, engine in zip(axes, engine_types):
        try:
            r = reg.xs(engine, level='engine_type')
            i = imp.xs(engine, level='engine_type')
        except KeyError:
            continue

        common = sorted(set(r.index) & set(i.index))
        r = r.loc[common]
        i = i.loc[common]

        x = np.arange(len(common))
        imp_pos = np.where(i.values >= 0, i.values, 0)
        imp_neg = np.where(i.values < 0, i.values, 0)

        width = 0.4
        ax.bar(x - width / 2, r.values, width=width, label='New registrations (BIL51)', color='steelblue')
        ax.bar(x - width / 2, imp_pos, width=width, bottom=r.values,
               label='Unexplained imports (residual)', color='darkorange', alpha=0.8)
        ax.bar(x + width / 2, np.abs(imp_neg), width=width,
               label='Net exports (negative residual)', color='firebrick', alpha=0.8)

        ymax = (r.values + imp_pos).max() * 1.1
        ax.set_ylim(0, ymax)
        ax.axhline(0, color='black', linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(common, rotation=45)
        ax.set_xlabel('Year')
        ax.set_ylabel('Count')
        ax.set_title(f'New car inflow decomposition — {engine}')
        ax.legend()

    plt.suptitle('Age-0 stock = BIL51 registrations + unexplained imports')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'new_car_imports_decomposition.png'))
    plt.close(fig)


def _draw_inflow_bars(ax, reg_eng, imp_eng, inflow_eng, years, A, bar_width, age_colors):
    """Draw grouped inflow bars onto ax. Returns legend handles dict."""
    import numpy as np
    x = np.arange(len(years))
    n_ages = A + 1
    legend_handles = {}

    for age_idx, age in enumerate(range(n_ages)):
        x_pos = x - 0.35 + bar_width * (age_idx + 0.5)

        if age == 0:
            reg_vals = np.array([reg_eng.get(y, 0) for y in years], dtype=float)
            imp_vals = np.array([imp_eng.get(y, 0) for y in years], dtype=float)

            h1 = ax.bar(x_pos, reg_vals, width=bar_width, color='steelblue', label='Age 0 — new registrations')
            legend_handles.setdefault('reg', h1)

            imp_pos = np.where(imp_vals >= 0, imp_vals, 0)
            imp_neg = np.where(imp_vals < 0, imp_vals, 0)
            if imp_pos.any():
                h2 = ax.bar(x_pos, imp_pos, width=bar_width, bottom=reg_vals,
                            color='darkorange', label='Age 0 — imports (residual)')
                legend_handles.setdefault('imp_pos', h2)
            if imp_neg.any():
                h3 = ax.bar(x_pos, np.abs(imp_neg), width=bar_width, bottom=reg_vals + imp_neg,
                            color='firebrick', alpha=0.7, label='Net export (negative)')
                legend_handles.setdefault('neg', h3)
        else:
            color = age_colors[(age_idx + 2) % len(age_colors)]
            vals = np.array([
                inflow_eng.get((y, age), 0) if not inflow_eng.empty else 0
                for y in years
            ], dtype=float)

            pos_vals = np.where(vals >= 0, vals, 0)
            neg_vals = np.where(vals < 0, vals, 0)

            h = ax.bar(x_pos, pos_vals, width=bar_width, color=color, label=f'Age {age} — used imports')
            legend_handles.setdefault(age_idx, h)
            if neg_vals.any():
                ax.bar(x_pos, np.abs(neg_vals), width=bar_width, bottom=neg_vals,
                       color='firebrick', alpha=0.7)

    return x, legend_handles


def _add_year_secondary_axis(ax, x, n_ages, bar_width, years):
    """Add a secondary x-axis on top showing year labels, one per group centre."""
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    # Centre of each year group
    group_centers = [xi - 0.35 + bar_width * n_ages / 2 for xi in x]
    ax2.set_xticks(group_centers)
    ax2.set_xticklabels(years, fontsize=8, rotation=45, ha='left')
    ax2.tick_params(axis='x', length=3, pad=1)
    ax2.set_xlabel('Year', fontsize=8)
    return ax2


def _finalize_inflow_ax(ax, x, years, engine_type, n_ages, bar_width):
    """Set ticks, labels, y-limit headroom on a completed inflow axis."""
    ax.axhline(0, color='black', linewidth=0.6)
    # Primary (bottom) axis: individual age ticks for every bar
    age_tick_positions = [
        xi - 0.35 + bar_width * (a + 0.5)
        for xi in x
        for a in range(n_ages)
    ]
    age_tick_labels = [str(a) for _ in x for a in range(n_ages)]
    ax.set_xticks(age_tick_positions)
    ax.set_xticklabels(age_tick_labels, fontsize=6)
    ax.set_xlabel('Car age')
    ax.set_ylabel('Number of cars')
    ax.set_title(f'Car inflow by age and year — {engine_type}')
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(bottom=ymin * 1.1 if ymin < 0 else 0, top=ymax * 1.1)
    if 2024 in years and 2025 in years:
        xi_2024 = years.index(2024)
        xi_2025 = years.index(2025)
        boundary = ((xi_2024 - 0.35 + bar_width * n_ages) + (xi_2025 - 0.35)) / 2
        ax.axvline(boundary, color='grey', linestyle='--', linewidth=1.2, alpha=0.4)


def _prep_inflow_series(new_registrations, new_car_imports, inflow_raw, engine_type):
    try:
        reg_eng = new_registrations.xs(engine_type, level='engine_type')
        reg_eng.index = reg_eng.index.get_level_values('year')
    except KeyError:
        reg_eng = pd.Series(dtype=float)
    try:
        imp_eng = new_car_imports.xs(engine_type, level='engine_type')
        imp_eng.index = imp_eng.index.get_level_values('year')
    except KeyError:
        imp_eng = pd.Series(dtype=float)
    try:
        inflow_eng = inflow_raw.xs(engine_type, level='engine_type')
    except KeyError:
        inflow_eng = pd.Series(dtype=float)
    return reg_eng, imp_eng, inflow_eng


def plot_inflow_by_age_stacked(new_registrations, new_car_imports, inflow_raw, engine_type, A=6, output_dir=OUTPUT_DIR):
    import numpy as np
    _ensure_dir(output_dir)

    reg_eng, imp_eng, inflow_eng = _prep_inflow_series(new_registrations, new_car_imports, inflow_raw, engine_type)
    years = sorted(set(reg_eng.index) | set(imp_eng.index))
    n_ages = A + 1
    bar_width = 0.7 / n_ages
    age_colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(max(8, len(years) * 1.1), 5))
    x, legend_handles = _draw_inflow_bars(ax, reg_eng, imp_eng, inflow_eng, years, A, bar_width, age_colors)
    _finalize_inflow_ax(ax, x, years, engine_type, n_ages, bar_width)
    _add_year_secondary_axis(ax, x, n_ages, bar_width, years)
    ax.legend(title='Age / type', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'inflow_age_stacked_{engine_type}.png'))
    plt.close(fig)


def plot_inflow_by_age_combined(new_registrations, new_car_imports, inflow_raw, A=6, output_dir=OUTPUT_DIR):
    """Combined BEV (top) + ICEV (bottom) subplot."""
    import numpy as np
    _ensure_dir(output_dir)

    engine_types = ['BEV', 'ICEV']
    age_colors = plt.cm.tab10.colors
    n_ages = A + 1
    bar_width = 0.7 / n_ages

    # Compute shared year list
    all_years = set()
    for et in engine_types:
        reg_eng, imp_eng, _ = _prep_inflow_series(new_registrations, new_car_imports, inflow_raw, et)
        all_years |= set(reg_eng.index) | set(imp_eng.index)
    years = sorted(all_years)

    fig, axes = plt.subplots(2, 1, figsize=(max(8, len(years) * 1.1), 9), sharex=False)

    for ax, engine_type in zip(axes, engine_types):
        reg_eng, imp_eng, inflow_eng = _prep_inflow_series(new_registrations, new_car_imports, inflow_raw, engine_type)
        eng_years = sorted(set(reg_eng.index) | set(imp_eng.index))
        x, legend_handles = _draw_inflow_bars(ax, reg_eng, imp_eng, inflow_eng, eng_years, A, bar_width, age_colors)
        _finalize_inflow_ax(ax, x, eng_years, engine_type, n_ages, bar_width)
        _add_year_secondary_axis(ax, x, n_ages, bar_width, eng_years)
        ax.legend(title='Age / type', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)

    axes[-1].set_xlabel('Car age')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inflow_age_stacked_combined.png'))
    plt.close(fig)


def _prep_inflow_by_source(new_registrations, new_car_imports, inflow_raw, engine_type, A=6):
    """Return (reg, imp_new, imp_used_by_age) year-indexed Series for one engine type.

    imp_used_by_age: dict {age: Series(year -> value)} for ages 1..A, keeping sign.
    """
    reg_eng, imp_eng, inflow_eng = _prep_inflow_series(new_registrations, new_car_imports, inflow_raw, engine_type)
    imp_used_by_age = {}
    for age in range(1, A + 1):
        if inflow_eng.empty:
            imp_used_by_age[age] = pd.Series(dtype=float)
        else:
            try:
                s = inflow_eng.xs(age, level='car_age')
                s.index = s.index.get_level_values('year')
            except KeyError:
                s = pd.Series(dtype=float)
            imp_used_by_age[age] = s
    return reg_eng, imp_eng, imp_used_by_age


def plot_inflow_by_source_combined(new_registrations, new_car_imports, inflow_raw, A=6, output_dir=OUTPUT_DIR):
    """BEV (top) + ICEV (bottom): per year, two side-by-side bars —
    left = new registrations, right = imports stacked by age (0 residual + ages 1-A).
    """
    import numpy as np
    _ensure_dir(output_dir)

    engine_types = ['BEV', 'ICEV']
    age_colors = plt.cm.tab10.colors  # age 0 residual uses index 0, age 1 → 1, etc.

    all_years = set()
    for et in engine_types:
        reg_eng, imp_eng, _ = _prep_inflow_series(new_registrations, new_car_imports, inflow_raw, et)
        all_years |= set(reg_eng.index) | set(imp_eng.index)
    years = sorted(all_years)

    bar_width = 0.35
    fig, axes = plt.subplots(2, 1, figsize=(max(8, len(years) * 0.8), 9), sharex=False)

    for ax, engine_type in zip(axes, engine_types):
        reg_eng, imp_eng, imp_used_by_age = _prep_inflow_by_source(
            new_registrations, new_car_imports, inflow_raw, engine_type, A)
        eng_years = sorted(
            set(reg_eng.index) | set(imp_eng.index) |
            {y for s in imp_used_by_age.values() for y in s.index}
        )
        x = np.arange(len(eng_years))

        # --- Left bar: new registrations ---
        reg_vals = np.array([reg_eng.get(y, 0) for y in eng_years], dtype=float)
        ax.bar(x - bar_width / 2, reg_vals, width=bar_width,
               color='steelblue', label='New registrations')

        # --- Right bar: imports stacked by age ---
        # Age-0 residual (bottom of import stack)
        imp0_vals = np.array([imp_eng.get(y, 0) for y in eng_years], dtype=float)
        imp0_pos = np.where(imp0_vals >= 0, imp0_vals, 0)
        imp0_neg = np.where(imp0_vals < 0, imp0_vals, 0)

        bottom = np.zeros(len(eng_years))
        ax.bar(x + bar_width / 2, imp0_pos, width=bar_width,
               bottom=bottom, color=age_colors[0], label='Imports age 0 (new-car residual)')
        bottom += imp0_pos

        # Ages 1-A stacked on top
        for age in range(1, A + 1):
            s = imp_used_by_age[age]
            vals = np.array([s.get(y, 0) for y in eng_years], dtype=float)
            pos_vals = np.where(vals >= 0, vals, 0)
            color = age_colors[age % len(age_colors)]
            ax.bar(x + bar_width / 2, pos_vals, width=bar_width,
                   bottom=bottom, color=color, label=f'Imports age {age}')
            bottom += pos_vals

        # Negative age-0 residual as downward red bar
        if imp0_neg.any():
            ax.bar(x + bar_width / 2, np.abs(imp0_neg), width=bar_width,
                   bottom=imp0_neg, color='firebrick', alpha=0.7, label='Net export (negative)')

        ax.axhline(0, color='black', linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(eng_years, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of cars')
        ax.set_title(f'Car inflow by source — {engine_type}')
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(bottom=ymin * 1.1 if ymin < 0 else 0, top=ymax * 1.1)
        ax.legend(title='Source / age', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inflow_source_stacked_combined.png'))
    plt.close(fig)


def plot_scrap_profile(dis_full, scrap_profile, car_type, age_floor, n_years=6, output_dir=OUTPUT_DIR):
    _ensure_dir(output_dir)
    fig, ax = plt.subplots()

    # Observed data: all ages for this car type
    dis_ct = dis_full.xs(car_type, level='engine_type')
    years = dis_ct.index.get_level_values('year').unique()[-n_years:]
    for year in years:
        try:
            data = dis_ct.loc[year]
            ax.plot(data.index.get_level_values('car_age'), data.values,
                    color='steelblue', alpha=0.4, linewidth=1)
        except KeyError:
            pass

    ax.plot(scrap_profile.index, scrap_profile.values,
            color='firebrick', linewidth=2, label='Model fit')
    ax.axvline(age_floor, color='grey', linewidth=0.8, linestyle=':',
               label=f'Estimation floor (age {age_floor})')

    ax.set_xlabel('Car age')
    ax.set_ylabel('Disappearance rate')
    ax.set_title(f'Scrap profile — {car_type}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'scrap_profile_{car_type}.png'))
    plt.close(fig)


def run_all(dis_rate, holdings_dist, engine_shares_df, market_shares, ncpurch_prob, inflow, new_registrations, BIL21, new_car_imports=None, inflow_raw=None, new_registrations_indexed=None, year=None, output_dir=OUTPUT_DIR):
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
    plot_stock_inflow_stock(BIL21, new_registrations, 'BEV', years=range(2020, 2026), output_dir=output_dir)
    plot_stock_inflow_stock(BIL21, new_registrations, 'ICEV', output_dir=output_dir)
    plot_cohort_survival(BIL21, year, output_dir=output_dir)
    for y in [2021, 2022, 2023]:
        plot_cohort_survival(BIL21, y, output_dir=output_dir)
    if new_car_imports is not None and inflow_raw is not None and new_registrations_indexed is not None:
        plot_inflow_by_age_combined(new_registrations_indexed, new_car_imports, inflow_raw, output_dir=output_dir)
        plot_inflow_by_source_combined(new_registrations_indexed, new_car_imports, inflow_raw, output_dir=output_dir)
    print(f"All plots saved to {os.path.abspath(output_dir)}")
