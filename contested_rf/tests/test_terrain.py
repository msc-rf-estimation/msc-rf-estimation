"""Tests for the DEM and path-diffraction field."""
import numpy as np

from contested_rf.terrain.dem import TerrainDEM
from contested_rf.terrain.profiles import DiffractionField


def test_procedural_dem_elevation_range_and_determinism():
    dem1 = TerrainDEM.procedural((0, 2000), (0, 2000), seed=0)
    dem2 = TerrainDEM.procedural((0, 2000), (0, 2000), seed=0)
    pts = np.array([[100.0, 100.0], [1000.0, 1500.0], [1900.0, 800.0]])
    e1, e2 = dem1.elevation_at(pts), dem2.elevation_at(pts)
    assert np.allclose(e1, e2)              # deterministic
    assert np.all(e1 >= 0.0)                # non-negative elevation
    assert e1.max() < 400.0                 # sane relief band


def test_from_grid_roundtrip():
    xs = np.linspace(0, 100, 6)
    ys = np.linspace(0, 100, 6)
    grid = np.tile(np.linspace(50, 100, 6)[:, None], (1, 6))  # ramp in x
    dem = TerrainDEM.from_grid(xs, ys, grid)
    assert abs(dem.elevation_at([[0.0, 50.0]])[0] - 50.0) < 1e-6
    assert dem.elevation_at([[100.0, 50.0]])[0] > dem.elevation_at([[0.0, 50.0]])[0]


def test_diffraction_loss_nonnegative_and_structured():
    dem = TerrainDEM.procedural((0, 2000), (0, 2000), seed=1, relief_m=80.0)
    df = DiffractionField(dem, freq_Hz=2.4e9)
    grid = np.array([[x, y] for x in np.linspace(0, 2000, 20)
                     for y in np.linspace(0, 2000, 20)])
    loss = df.loss(np.array([1000.0, 1000.0]), grid)
    assert loss.shape == (400,)
    assert np.all(loss >= 0.0)
    # Rolling terrain should shadow at least some receivers.
    assert loss.max() > 3.0


def test_diffraction_zero_over_flat_dem():
    # A perfectly flat DEM -> no obstruction above LoS -> no loss.
    xs = np.linspace(0, 2000, 11); ys = np.linspace(0, 2000, 11)
    flat = np.full((11, 11), 100.0)
    dem = TerrainDEM.from_grid(xs, ys, flat)
    df = DiffractionField(dem, freq_Hz=2.4e9)
    grid = np.array([[500.0, 500.0], [1500.0, 1500.0], [200.0, 1800.0]])
    loss = df.loss(np.array([1000.0, 1000.0]), grid)
    assert np.allclose(loss, 0.0)
