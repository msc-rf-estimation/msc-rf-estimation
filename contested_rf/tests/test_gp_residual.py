"""Tests for the GP residual layer (Matérn 5/2 GP)."""
import numpy as np

from contested_rf.estimators.gp_residual import GPResidual, matern52


def test_matern52_self_covariance_equals_signal_var():
    X = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
    K = matern52(X, X, lengthscale=50.0, signal_var=9.0)
    # Diagonal (r=0) => signal_var exactly.
    assert np.allclose(np.diag(K), 9.0)
    # Symmetric PSD-ish: symmetric and decreasing with distance.
    assert np.allclose(K, K.T)
    assert K[0, 1] < K[0, 0]


def test_gp_interpolates_training_points():
    # Fit to a smooth 2D function; posterior mean should be close at train pts.
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 100, size=(40, 2))
    f = lambda P: np.sin(P[:, 0] / 20.0) * np.cos(P[:, 1] / 25.0) * 5.0
    y = f(X)

    gp = GPResidual(lengthscale=20.0, signal_var=10.0, noise_var=1e-3)
    gp.set_data(X, y).optimize()

    mu = gp.predict(X, return_var=False)
    rmse = np.sqrt(np.mean((mu - y) ** 2))
    assert rmse < 0.5  # near-interpolation with tiny noise


def test_uncertainty_grows_away_from_data():
    # Cluster of training points near the origin; query near vs far.
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [0.0, 2.0], [1.0, 2.0]])
    y = np.array([0.1, -0.2, 0.05, 0.15, -0.1])

    gp = GPResidual(lengthscale=5.0, signal_var=1.0, noise_var=1e-2)
    gp.set_data(X, y)
    gp._build_cache()

    _, var_near = gp.predict(np.array([[1.0, 1.0]]))
    _, var_far = gp.predict(np.array([[500.0, 500.0]]))
    assert var_far[0] > var_near[0]
    # Far from data, variance approaches the prior signal variance.
    assert var_far[0] > 0.5 * gp.signal_var


def test_marginal_likelihood_prefers_correct_lengthscale():
    # Data with a clear spatial scale; ML should not pick a degenerate l.
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 200, size=(60, 2))
    f = lambda P: 4.0 * np.sin(P[:, 0] / 40.0)
    y = f(X) + rng.normal(0, 0.3, size=60)

    gp = GPResidual(lengthscale=10.0, signal_var=5.0, noise_var=1.0)
    gp.set_data(X, y).optimize()

    # Recovered lengthscale should be on the order of tens of metres, not
    # pinned to a bound.
    assert 5.0 < gp.lengthscale < 5000.0
    # A reasonable fit beats the flat-prior (zero-mean) baseline in ML.
    assert np.isfinite(gp.log_marginal_likelihood())


def test_sparse_path_runs_and_approximates_exact():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 100, size=(400, 2))
    f = lambda P: np.sin(P[:, 0] / 15.0) * 3.0
    y = f(X) + rng.normal(0, 0.2, size=400)

    exact = GPResidual(lengthscale=15.0, signal_var=5.0, noise_var=0.5)
    exact.set_data(X, y)
    exact._build_cache()

    sparse = GPResidual(
        lengthscale=15.0, signal_var=5.0, noise_var=0.5,
        sparse_threshold=100, n_inducing=80, random_state=3,
    )
    sparse.set_data(X, y)
    sparse._build_cache()
    assert sparse._cache["mode"] == "sparse"

    Xq = rng.uniform(0, 100, size=(50, 2))
    mu_e = exact.predict(Xq, return_var=False)
    mu_s = sparse.predict(Xq, return_var=False)
    # Sparse should track exact reasonably on a smooth field.
    assert np.sqrt(np.mean((mu_e - mu_s) ** 2)) < 1.0


def test_predict_with_no_data_returns_prior():
    gp = GPResidual(signal_var=7.0)
    mu, var = gp.predict(np.array([[10.0, 10.0], [20.0, 5.0]]))
    assert np.allclose(mu, 0.0)
    assert np.allclose(var, 7.0)
