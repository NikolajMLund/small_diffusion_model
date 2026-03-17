from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for the main settings for the forecasting model."""
    max_car_age: int = 24
    engine_types: list[str] = field(default_factory=lambda: ['ICEV', 'BEV'])
    purchase_age_limit: int = 6