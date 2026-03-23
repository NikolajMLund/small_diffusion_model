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

    def get_dis_rates(self, config: ConstantScenarioConfig=ConstantScenarioConfig()):
        return self._baseline_dis_rates()

    def get_purchase_inflows(self, config: ConstantScenarioConfig=ConstantScenarioConfig()):
        return self._baseline_purchase_inflows()
