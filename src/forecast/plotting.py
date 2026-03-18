import os
import numpy as np
import matplotlib.pyplot as plt


def plot_forecast_vs_actual(
    forecasted_distributions: np.ndarray,
    holdings_dist,
    model_config,
    forecast_config,
    output_dir: str,
    file_name: str,
):
    """
    Bar chart of forecast vs actual age distribution for each engine type.

    forecasted_distributions: (n_periods, n_engine_types, n_ages)
    holdings_dist: pandas Series/DataFrame with MultiIndex (year, car_type, car_age)
    """
    ages = np.arange(model_config.max_car_age + 2)
    width = 0.35

    fig, ax = plt.subplots(1, len(model_config.engine_types), figsize=(14, 5))

    for i, engine_type in enumerate(model_config.engine_types):
        ax[i].bar(ages - width/2, forecasted_distributions[-1, i, :], width=width, label=f'Forecast ({forecast_config.target_year})')
        if forecast_config.target_year in holdings_dist.index.get_level_values('year'):
            actual = holdings_dist.loc[forecast_config.target_year, engine_type, :].values
            ax[i].bar(ages + width/2, actual, width=width, label=f'Actual ({forecast_config.target_year})')
        ax[i].set_xlabel('Car age')
        ax[i].set_ylabel('Share of total holdings')
        ax[i].set_title(f'{engine_type}: forecast vs actual ({forecast_config.target_year})')
        ax[i].legend()

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, file_name))
    plt.show()