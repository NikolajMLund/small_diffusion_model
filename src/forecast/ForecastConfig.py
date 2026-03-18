from dataclasses import dataclass


@dataclass
class ForecastConfig:
    """Timing configuration shared by the wrangler, forecast engine, and plotting."""
    base_year: int = 2024
    target_year: int = 2025
