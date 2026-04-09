"""
Constant scenario: no active levers.

Replicates the historical baseline — disappearance rates and purchase inflows
are frozen at base_year values across the entire forecast horizon.
"""

from dataclasses import dataclass

from forecast.BaseScenario import BaseScenario


@dataclass
class ConstantScenarioConfig:
    """No active levers — the wrangler needs no scenario parameters."""
    pass


class ConstantScenario(BaseScenario):

    def __init__(self, data, model_config, forecast_config, scenario_config):
        super().__init__(data, model_config, forecast_config, scenario_config)
        self.projected_inflows = self.get_projected_inflows(scenario_config)
        self.dis_rates = self.get_dis_rates(scenario_config)

    def get_state(self, config: ConstantScenarioConfig=ConstantScenarioConfig()):
        return self._baseline_get_state()

    def get_dis_rates(self, config: ConstantScenarioConfig=ConstantScenarioConfig()):
        return self._baseline_dis_rates()

    def get_projected_inflows(self, config: ConstantScenarioConfig=ConstantScenarioConfig()):
        return self._baseline_projected_inflows()
