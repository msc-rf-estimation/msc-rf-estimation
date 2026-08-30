"""Base station (legitimate signal source) for a scenario.

Physically just an RF emitter — same construction as a Jammer — but conceptually
distinct: a base station's received power goes in the *numerator* of SINR
(signal of interest), whereas jammers' powers go in the *denominator*
(interference).
"""
import math
from dataclasses import dataclass
from typing import Tuple

from contested_rf.propagation.antenna import directional_gain


@dataclass
class BaseStation:
    """A friendly base station emitting the signal of interest.

    Attributes:
        name: short identifier (e.g. "BS").
        position: (x, y) in metres.
        power_dBm: transmit power in dBm.
        is_directional: if True, gain depends on bearing per Gaussian pattern.
        theta_main_deg: main beam direction in degrees. Used only if directional.
        theta_3db_deg: half-power beamwidth in degrees. Used only if directional.
        peak_gain_dB: peak gain (G0) in dB. Default 0 dB.
    """
    name: str
    position: Tuple[float, float]
    power_dBm: float

    is_directional: bool = False
    theta_main_deg: float = 0.0
    theta_3db_deg: float = 60.0
    peak_gain_dB: float = 0.0

    def gain_toward(self, sensor_position: Tuple[float, float]) -> float:
        """Antenna gain (dB) radiated from this base station toward a sensor."""
        if not self.is_directional:
            return self.peak_gain_dB

        bx, by = self.position
        sx, sy = sensor_position
        bearing_deg = math.degrees(math.atan2(sy - by, sx - bx))
        return directional_gain(
            bearing_deg,
            theta_main=self.theta_main_deg,
            theta_3db=self.theta_3db_deg,
            G0_dB=self.peak_gain_dB,
        )
