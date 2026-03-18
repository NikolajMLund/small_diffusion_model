"""
Reusable helper functions for scenario wranglers.

Each function implements one scenario lever. Scenarios import whichever
helpers they need and compose them inside their get_dis_rates /
get_purchase_inflows methods.
"""

import numpy as np


def rescale_bev_icev_split(
    baseline_inflows: np.ndarray,
    bev_share_schedule: np.ndarray,
    bev_idx: int,
    icev_idx: int,
) -> np.ndarray:
    """
    Rescale purchase inflows so that BEV/ICEV totals match a given schedule,
    preserving the within-type age distribution shape.

    Parameters
    ----------
    baseline_inflows : (n_years, n_car_types, max_purchase_age+1)
        Baseline purchase inflows to rescale.
    bev_share_schedule : (n_years,)
        Target BEV fraction of total purchases in each year. Values in [0, 1].
    bev_idx : int
        Index of BEV in the car_types dimension.
    icev_idx : int
        Index of ICEV in the car_types dimension.

    Returns
    -------
    np.ndarray  same shape as baseline_inflows
    """
    inflows = baseline_inflows.copy()
    n_years = inflows.shape[0]

    for t in range(n_years):
        total = inflows[t].sum()
        bev_share = bev_share_schedule[t]

        bev_row = inflows[t, bev_idx, :]
        icev_row = inflows[t, icev_idx, :]

        bev_total = bev_row.sum()
        icev_total = icev_row.sum()

        # Normalise to age-distribution shape, then rescale to new totals
        if bev_total > 0:
            inflows[t, bev_idx, :] = (bev_row / bev_total) * (bev_share * total)
        else:
            # No historical BEV data — distribute evenly across ages
            n_ages = bev_row.shape[0]
            inflows[t, bev_idx, :] = (bev_share * total) / n_ages

        if icev_total > 0:
            inflows[t, icev_idx, :] = (icev_row / icev_total) * ((1.0 - bev_share) * total)
        else:
            n_ages = icev_row.shape[0]
            inflows[t, icev_idx, :] = ((1.0 - bev_share) * total) / n_ages

    return inflows


def apply_scrappage_multiplier(
    baseline_dis_rates: np.ndarray,
    multipliers: np.ndarray,
) -> np.ndarray:
    """
    Scale disappearance rates by a per-year multiplier.

    Parameters
    ----------
    baseline_dis_rates : (n_years, n_car_types, n_ages)
    multipliers : (n_years,)
        Per-year scalar applied to all car types and ages. 1.0 = no change.

    Returns
    -------
    np.ndarray  same shape as baseline_dis_rates, clipped to [0, 1]
    """
    dis_rates = baseline_dis_rates.copy()
    for t, m in enumerate(multipliers):
        dis_rates[t] *= m
    return np.clip(dis_rates, 0.0, 1.0)
