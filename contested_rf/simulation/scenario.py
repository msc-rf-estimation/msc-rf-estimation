"""Scenario definitions.

A Scenario bundles jammers, a base station and global settings (carrier
frequency, grid extent). SCENARIO_1 is a single static omnidirectional
jammer; SCENARIO_2 adds a directional emitter alongside a known
omnidirectional one. SCENARIO_3 defines a moving emitter and is retained in
the code but is not used by any reported experiment.

The base station defaults to a single 30 dBm omnidirectional transmitter at
the centre of the area.
"""
from dataclasses import dataclass, field
from typing import List, Tuple

from contested_rf.simulation.base_station import BaseStation
from contested_rf.simulation.jammer import Jammer


def _default_base_station() -> BaseStation:
    """Default factory: 30 dBm omni base station at the grid centre."""
    return BaseStation(name="BS", position=(1000.0, 1000.0), power_dBm=30.0)


@dataclass
class Scenario:
    """A simulation scenario.

    Attributes:
        name: human-readable identifier.
        jammers: list of Jammer objects present in the scenario.
        base_station: friendly transmitter providing the signal of interest.
        operating_freq_Hz: carrier frequency, constant across the area.
        grid_xlim: (xmin, xmax) of the simulation area in metres.
        grid_ylim: (ymin, ymax) of the simulation area in metres.
    """
    name: str
    jammers: List[Jammer]
    base_station: BaseStation = field(default_factory=_default_base_station)
    operating_freq_Hz: float = 2.4e9
    grid_xlim: Tuple[float, float] = (0.0, 2000.0)
    grid_ylim: Tuple[float, float] = (0.0, 2000.0)


# Scenario 1: single static omnidirectional jammer at (1200, 800), 30 dBm.
SCENARIO_1 = Scenario(
    name="S1: single static omni",
    jammers=[
        Jammer(name="A", position=(1200.0, 800.0), power_dBm=30.0),
    ],
)


# Scenario 2: two jammers. A is omni at (600, 700) with 30 dBm. B is directional
# at (1400, 1300) with 27 dBm and a 60° beamwidth. B's main beam direction is
# set to 225° (south-west) as a sensible default; in inference experiments this
# may be one of the parameters the SMC estimates.
SCENARIO_2 = Scenario(
    name="S2: two jammers, one omni and one directional",
    jammers=[
        Jammer(name="A", position=(600.0, 700.0), power_dBm=30.0),
        Jammer(
            name="B",
            position=(1400.0, 1300.0),
            power_dBm=27.0,
            is_directional=True,
            theta_main_deg=225.0,
            theta_3db_deg=60.0,
            front_back_ratio_dB=30.0,
        ),
    ],
)


# Scenario 3: single dynamic omnidirectional jammer at (500, 500), 30 dBm,
# moving at 2 m/s toward (1500, 1500). On reaching the target it stops.
SCENARIO_3 = Scenario(
    name="S3: single dynamic omni",
    jammers=[
        Jammer(
            name="A",
            position=(500.0, 500.0),
            power_dBm=30.0,
            velocity_mps=2.0,
            target_position=(1500.0, 1500.0),
        ),
    ],
)


# Lookup by integer for convenience.
SCENARIOS = {
    1: SCENARIO_1,
    2: SCENARIO_2,
    3: SCENARIO_3,
}
