import matplotlib.pyplot as plt
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


def run_all(dis_rate, holdings_dist, engine_shares_df, market_shares, ncpurch_prob, output_dir=OUTPUT_DIR):
    """Generate all standard plots."""
    plot_disappearance_rate(dis_rate, 'BEV', output_dir=output_dir)
    plot_disappearance_rate(dis_rate, 'ICEV', output_dir=output_dir)
    plot_holdings_by_age(holdings_dist, 'BEV', output_dir=output_dir)
    plot_holdings_by_age(holdings_dist, 'ICEV', output_dir=output_dir)
    plot_holdings_by_engine_type(engine_shares_df, output_dir=output_dir)
    plot_market_shares(market_shares, output_dir=output_dir)
    plot_purchase_probability(ncpurch_prob, output_dir=output_dir)
    print(f"All plots saved to {os.path.abspath(output_dir)}")
