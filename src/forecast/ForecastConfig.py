from dataclasses import dataclass, field


@dataclass
class ForecastConfig:
    """Configuration for the main settings for the actual forecast."""
    target_year: int = 2025
    base_year: int = 2024
    car_types: list[str] = field(default_factory=lambda: ['BEV'])
    invariant_disappearance_rates: bool = True
    invariant_inflows: bool = True