"""Digital elevation model over the operating area.

The diffraction machinery needs only an elevation field E(x, y), so the
interface is source-agnostic: TerrainDEM.elevation_at(points).

The default factory builds a procedural DEM with South-Downs-like statistics —
a smooth, correlated rolling-hill field with elevations roughly in the 40-160 m
band. Using a procedural field rather than a real SRTM tile keeps the terrain
condition a controlled stress test of structured model error rather than a
site-accurate prediction; this is stated as a limitation in the dissertation.
To use a real raster instead, construct TerrainDEM.from_grid(...) with an
elevation array loaded via rasterio; nothing downstream changes.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class TerrainDEM:
    """Elevation field E(x, y) queryable at arbitrary points."""

    def __init__(self, xs, ys, elevation_grid):
        self._xs, self._ys = np.asarray(xs), np.asarray(ys)
        self._interp = RegularGridInterpolator(
            (self._xs, self._ys), np.asarray(elevation_grid, dtype=float),
            method="linear", bounds_error=False, fill_value=None)
        self.grid = np.asarray(elevation_grid, dtype=float)

    @classmethod
    def from_grid(cls, xs, ys, elevation_grid):
        """Wrap a real elevation raster (e.g. resampled SRTM)."""
        return cls(xs, ys, elevation_grid)

    @classmethod
    def procedural(cls, x_range, y_range, resolution_m=40.0, seed=0,
                   base_m=90.0, relief_m=35.0, lengthscale_m=450.0):
        """South-Downs-like procedural DEM: smooth correlated rolling hills.

        Built as a low-frequency correlated Gaussian field (large lengthscale)
        offset to a positive base elevation. Deterministic given the seed.
        """
        xs = np.arange(x_range[0], x_range[1] + resolution_m, resolution_m)
        ys = np.arange(y_range[0], y_range[1] + resolution_m, resolution_m)
        XX, YY = np.meshgrid(xs, ys, indexing="ij")
        pts = np.column_stack([XX.ravel(), YY.ravel()])

        rng = np.random.default_rng(seed)
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.sum(diff ** 2, axis=-1)
        cov = np.exp(-d2 / (2.0 * lengthscale_m ** 2))
        cov[np.diag_indices_from(cov)] += 1e-6
        L = np.linalg.cholesky(cov)
        field = (L @ rng.normal(size=pts.shape[0])).reshape(XX.shape)

        # Normalise to unit std, then scale to the target relief and base.
        field = (field - field.mean()) / (field.std() + 1e-12)
        elevation = base_m + relief_m * field
        elevation = np.clip(elevation, 0.0, None)
        return cls(xs, ys, elevation)

    def elevation_at(self, points):
        """Terrain elevation (m) at query points (M, 2)."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        return self._interp(points)
