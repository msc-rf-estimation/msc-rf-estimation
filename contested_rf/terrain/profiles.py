"""Path-dependent diffraction loss from a jammer to receiver points.

Unlike shadow fading, knife-edge diffraction depends on the path rather than
the position: the terrain between emitter and receiver. This module samples the
elevation profile along each emitter-to-receiver line, finds the single most
obstructing point, and returns its diffraction loss, vectorised over many
receiver points so a whole grid or survey can be evaluated at once.
"""
import numpy as np

from contested_rf.propagation.diffraction import fresnel_parameter, knife_edge_loss_dB


class DiffractionField:
    """Diffraction excess-loss (dB) from a fixed emitter over the DEM."""

    def __init__(self, dem, freq_Hz, emitter_agl_m=2.0, rx_agl_m=50.0,
                 n_samples=48):
        """
        Args:
            dem: TerrainDEM.
            freq_Hz: carrier frequency.
            emitter_agl_m: jammer antenna height above ground level, m.
            rx_agl_m: receiver (UAV) height above ground level, m.
            n_samples: profile samples along each path (interior points used
                for the obstruction search).
        """
        self.dem = dem
        self.freq_Hz = freq_Hz
        self.emitter_agl = emitter_agl_m
        self.rx_agl = rx_agl_m
        self.n_samples = n_samples

    def loss(self, emitter_xy, points):
        """Diffraction loss (dB, >=0) from emitter to each of `points` (M,2)."""
        emitter_xy = np.asarray(emitter_xy, dtype=float)
        points = np.atleast_2d(np.asarray(points, dtype=float))
        M = points.shape[0]

        # Interior sample fractions along the path (exclude endpoints).
        fr = np.linspace(0.0, 1.0, self.n_samples + 2)[1:-1]  # (S,)
        S = fr.shape[0]

        # Sample positions: (M, S, 2).
        sample_xy = (emitter_xy[None, None, :]
                     + fr[None, :, None] * (points[:, None, :] - emitter_xy[None, None, :]))
        terr = self.dem.elevation_at(sample_xy.reshape(-1, 2)).reshape(M, S)

        # Endpoint heights above datum.
        h_tx = self.dem.elevation_at(emitter_xy[None, :])[0] + self.emitter_agl
        h_rx = self.dem.elevation_at(points) + self.rx_agl  # (M,)

        # LoS ray height at each sample fraction: linear interp of endpoints.
        los = h_tx + fr[None, :] * (h_rx[:, None] - h_tx)  # (M, S)
        h = terr - los  # obstruction height above LoS

        # Along-path distances to each sample.
        d_total = np.hypot(points[:, 0] - emitter_xy[0],
                           points[:, 1] - emitter_xy[1])  # (M,)
        d1 = fr[None, :] * d_total[:, None]
        d2 = (1.0 - fr[None, :]) * d_total[:, None]

        v = fresnel_parameter(h, d1, d2, self.freq_Hz)  # (M, S)
        # Dominant knife edge = maximum Fresnel parameter along the path.
        v_max = np.max(v, axis=1)
        return knife_edge_loss_dB(v_max)
