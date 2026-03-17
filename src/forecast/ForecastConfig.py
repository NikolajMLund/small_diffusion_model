from dataclasses import dataclass, field


@dataclass
class ForecastConfig:
    """Configuration for the main settings for the actual forecast."""
    target_year: int = 2023
    car_type: list[str] = field(default_factory=lambda: ['BEV'])