"""A single shareable shadow-fading realisation over the operating area.

sample_shadowing_2d draws a correlated Gaussian field at arbitrary query
points, but two separate calls give independent draws rather than one physical
field sampled in two places. The ground-truth SINR map and the survey
observations must share one realisation, so ShadowField samples the field once
on a grid finer than the decorrelation length and interpolates to any query
points. Both the observation generator and the evaluation map query the same
object.

The evaluation target therefore includes the realised shadow field, so the GP
layer is credited for reconstructing real spatial shadow structure. This is
disclosed in the dissertation as one of two asymmetries favouring the
GP-based estimators.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class ShadowField:
    """One correlated-Gaussian shadow realisation, queryable anywhere."""

    def __init__(self, x_range, y_range, sigma=4.0, L=50.0,
                 resolution_m=40.0, seed=None):
        """
        Args:
            x_range, y_range: (min, max) bounds of the area, metres.
            sigma: shadow std dev, dB.
            L: decorrelation length, metres. Grid spacing is kept below this.
            resolution_m: coarse-grid spacing (m). Must be < L to resolve the
                field. Default 40 m for L = 50 m.
            seed: RNG seed for the realisation.
        """
        self.sigma = sigma
        self.L = L
        xs = np.arange(x_range[0], x_range[1] + resolution_m, resolution_m)
        ys = np.arange(y_range[0], y_range[1] + resolution_m, resolution_m)
        self._xs, self._ys = xs, ys

        XX, YY = np.meshgrid(xs, ys, indexing="ij")
        pts = np.column_stack([XX.ravel(), YY.ravel()])

        # Joint sample of the field on the coarse grid (one realisation).
        rng = np.random.default_rng(seed)
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.sum(diff ** 2, axis=-1)
        cov = (sigma ** 2) * np.exp(-d2 / (2.0 * L ** 2))
        cov[np.diag_indices_from(cov)] += 1e-6
        Lchol = np.linalg.cholesky(cov)
        field = (Lchol @ rng.normal(size=pts.shape[0])).reshape(XX.shape)

        # Interpolator over the grid; linear, with nearest-style extrapolation
        # at the edges so query points on the boundary are always covered.
        self._interp = RegularGridInterpolator(
            (xs, ys), field, method="linear",
            bounds_error=False, fill_value=None,
        )
        self._field = field

    def evaluate(self, points):
        """Shadow values (dB) at query points (M, 2)."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        return self._interp(points)
