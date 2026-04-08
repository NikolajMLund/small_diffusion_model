"""
Tests for markov_step in core.py.

Covers the "sticky last bin" change:
  - age_step_matrix[-1, -1] = 1  (last bin retains cars)
  - dis_rates[-1] = 1.0 for normal cars  → forced scrappage, same as before
  - dis_rates[-1] = 0.0 for old cars     → cars accumulate forever

The regression test verifies that normal cars still vanish at the last bin
after the structural change (i.e. the explicit dis_rate=1.0 replicates the
old implicit behaviour).
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from forecast.core import markov_step
from forecast.ModelConfig import ModelConfig
from forecast.ForecastConfig import ForecastConfig


@pytest.fixture
def configs():
    model_config = ModelConfig(max_car_age=4, engine_types=['ICEV'], purchase_age_limit=1)
    forecast_config = ForecastConfig(base_year=2023, target_year=2025)
    return model_config, forecast_config


def _zero_dis_rates(model_config):
    """All-zero disappearance rates (no scrappage)."""
    return np.zeros(model_config.max_car_age + 2)


def test_cars_at_last_bin_removed_when_dis_rate_is_one(configs):
    """
    Regression: with dis_rates[-1] = 1.0, cars at the last age bin must
    disappear after one step — matching the original implicit forced scrappage.
    """
    model_config, forecast_config = configs
    n_ages = model_config.max_car_age + 2  # 6

    state = np.zeros(n_ages)
    state[-1] = 100.0  # put cars only in the last bin

    dis_rates = _zero_dis_rates(model_config)
    dis_rates[-1] = 1.0  # explicit forced scrappage

    next_state = markov_step(
        state=state,
        dis_rates=dis_rates,
        purchase_inflows=np.zeros(model_config.purchase_age_limit + 1),
        model_config=model_config,
        forecast_config=forecast_config,
    )

    assert next_state[-1] == 0.0, (
        "Cars at the last bin should be scrapped when dis_rates[-1] = 1.0"
    )
    assert next_state.sum() == 0.0


def test_cars_at_last_bin_persist_when_dis_rate_is_zero(configs):
    """
    New behaviour: with dis_rates[-1] = 0.0, cars at the last age bin must
    stay there (old cars that don't disappear).
    """
    model_config, forecast_config = configs
    n_ages = model_config.max_car_age + 2

    state = np.zeros(n_ages)
    state[-1] = 100.0

    dis_rates = _zero_dis_rates(model_config)
    # dis_rates[-1] left at 0.0 — old cars survive

    next_state = markov_step(
        state=state,
        dis_rates=dis_rates,
        purchase_inflows=np.zeros(model_config.purchase_age_limit + 1),
        model_config=model_config,
        forecast_config=forecast_config,
    )

    assert next_state[-1] == 100.0, (
        "Cars at the last bin should persist when dis_rates[-1] = 0.0"
    )


def test_normal_aging_is_unaffected(configs):
    """
    Cars at intermediate ages still shift forward by one and are scaled
    by their survival probability, regardless of last-bin behaviour.
    """
    model_config, forecast_config = configs
    n_ages = model_config.max_car_age + 2

    state = np.zeros(n_ages)
    state[2] = 200.0  # cars at age 2

    dis_rates = _zero_dis_rates(model_config)
    dis_rates[-1] = 1.0  # normal cars: forced scrappage at last bin
    dis_rates[2] = 0.2   # 20 % disappearance at age 2

    next_state = markov_step(
        state=state,
        dis_rates=dis_rates,
        purchase_inflows=np.zeros(model_config.purchase_age_limit + 1),
        model_config=model_config,
        forecast_config=forecast_config,
    )

    assert next_state[2] == 0.0, "Cars should leave age 2"
    assert np.isclose(next_state[3], 200.0 * 0.8), (
        "Surviving cars (80 %) should shift to age 3"
    )