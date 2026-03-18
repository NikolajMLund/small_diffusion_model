from dataclasses import dataclass


@dataclass
class ForecastConfig:
    """Configuration for the main settings for the actual forecast."""
    target_year: int = 2025
    base_year: int = 2024
    invariant_disappearance_rates: bool = True
    invariant_inflows: bool = True