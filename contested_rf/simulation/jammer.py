"""Jammer dataclass for emitters in a contested RF scenario."""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from contested_rf.propagation.antenna import directional_gain


@dataclass
class Jammer:
    """A single emitter.

    Coordinates in metres on a 2D grid, power in dBm. Antennas are
    omnidirectional by default or Gaussian directional. Motion is static by
    default or constant-velocity toward target_position, stopping on arrival.

    Attributes:
        name: short identifier.
        position: (x, y) in metres at simulation start.
        power_dBm: transmit power.
        is_directional: if True, gain depends on bearing.
        theta_main_deg: boresight direction, degrees. Directional only.
        theta_3db_deg: half-power beamwidth, degrees. Directional only.
        peak_gain_dB: boresight gain, dB. Default 0.
        front_back_ratio_dB: back-lobe depth at which the Gaussian falloff is
            clamped. None leaves the unphysical deep nulls; a realistic 10-40 dB
            makes the truth antenna match the clamped model the filter assumes.
            Directional only.
        velocity_mps: speed along the line to target_position; 0 is static.
        target_position: (x, y) motion target, or None.
    """
    name: str
    position: Tuple[float, float]
    power_dBm: float

    # Antenna directionality (optional)
    is_directional: bool = False
    theta_main_deg: float = 0.0
    theta_3db_deg: float = 60.0
    peak_gain_dB: float = 0.0
    front_back_ratio_dB: Optional[float] = None

    # Motion (optional)
    velocity_mps: float = 0.0
    target_position: Optional[Tuple[float, float]] = None

    def position_at(self, t_sec: float) -> Tuple[float, float]:
        """Return the jammer's position at time t seconds after start.

        Static jammers (velocity 0 or no target) ignore t. Dynamic jammers
        move at velocity_mps along the straight line from initial position
        toward target_position, stopping once target is reached.
        """
        if self.velocity_mps == 0.0 or self.target_position is None:
            return self.position

        x0, y0 = self.position
        x1, y1 = self.target_position
        dx, dy = x1 - x0, y1 - y0
        distance_to_target = math.hypot(dx, dy)
        if distance_to_target == 0.0:
            return self.position

        travelled = self.velocity_mps * t_sec
        if travelled >= distance_to_target:
            return self.target_position

        frac = travelled / distance_to_target
        return (x0 + frac * dx, y0 + frac * dy)

    def gain_toward(self, sensor_position: Tuple[float, float], t_sec: float = 0.0) -> float:
        """Antenna gain in dB radiated toward a sensor.

        Omnidirectional emitters return peak_gain_dB; directional emitters evaluate
        the Gaussian pattern at the bearing from their current position to the sensor.
        """
        if not self.is_directional:
            return self.peak_gain_dB

        jx, jy = self.position_at(t_sec)
        sx, sy = sensor_position
        bearing_deg = math.degrees(math.atan2(sy - jy, sx - jx))
        return directional_gain(
            bearing_deg,
            theta_main=self.theta_main_deg,
            theta_3db=self.theta_3db_deg,
            G0_dB=self.peak_gain_dB,
            front_back_ratio_dB=self.front_back_ratio_dB,
        )
