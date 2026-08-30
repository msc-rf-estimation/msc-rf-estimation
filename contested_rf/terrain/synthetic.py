"""Synthetic structured terrain loss: a controlled stand-in for diffraction.

Produces a deterministic, smoothly varying excess-loss field with the same
essential character as knife-edge diffraction — spatially structured and
persistent, and invisible to the learner's free-space forward model.

The field is a sum of Gaussian hill shadows: localised regions of extra
attenuation in dB, with a spatial lengthscale well above the shadow-fading
decorrelation length, so it is structured signal a GP can learn rather than
noise. Applied to the jammer's received power in both the ground-truth map and
the observations. Superseded by contested_rf.terrain.profiles for the reported
experiments, which use real knife-edge diffraction over a DEM.
"""
import numpy as np

# Default hill shadows: (cx, cy, peak_loss_dB, width_m). Placed to overlap the
# jammer/base-station geometry so the shadows actually bite.
_DEFAULT_HILLS = (
    (900.0, 1100.0, 14.0, 280.0),
    (1400.0, 600.0, 11.0, 220.0),
    (600.0, 1500.0, 9.0, 300.0),
)


class SyntheticTerrain:
    """Deterministic structured excess-loss field (dB), queryable anywhere."""

    def __init__(self, hills=_DEFAULT_HILLS):
        self.hills = tuple(hills)

    def loss(self, points):
        """Excess attenuation (dB, >= 0) at query points (M, 2)."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        total = np.zeros(points.shape[0])
        for cx, cy, amp, w in self.hills:
            d2 = (points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2
            total += amp * np.exp(-d2 / (2.0 * w ** 2))
        return total
