"""Gaussian process residual layer.

The parametric layer predicts f_learner(x, theta_hat) from the SMC posterior
mean. It is deliberately impoverished — free-space exponent n = 2.0 and no
terrain term, against a truth of n = 2.5 with shadow fading and knife-edge
diffraction — so everything it cannot represent appears as a structured
residual r_k = z_k - f_learner(x_k, theta_hat). This module models that
residual, letting the combined estimator predict
f_learner(x*, theta_hat) + mu_GP(x*) with sigma^2_GP(x*) as its uncertainty.

Kernel is Matern 5/2: twice mean-square differentiable, so it admits the sharp
gradients at diffraction edges that an RBF kernel smooths away. Hyperparameters
are fitted by maximising the log marginal likelihood, warm-started and refit
periodically rather than every step. A sparse subset-of-regressors path with M
inducing points is available for large training sets; the exact path is the
default.

Implemented directly in NumPy and SciPy rather than through a GP library.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

# Numerical floor added to the kernel diagonal for Cholesky stability, on top
# of the (learned) observation noise variance.
_JITTER = 1e-8


def matern52(X1, X2, lengthscale, signal_var):
    """Matern 5/2 covariance matrix between two point sets.

        k(r) = signal_var (1 + sqrt(5) r/l + 5 r^2 / (3 l^2)) exp(-sqrt(5) r/l)

    Args:
        X1: (n1, D) input locations.
        X2: (n2, D) input locations.
        lengthscale: isotropic lengthscale l > 0, metres.
        signal_var: signal variance > 0, dB^2.

    Returns:
        (n1, n2) covariance matrix.
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    # Pairwise Euclidean distances via the (a-b)^2 = a^2 - 2ab + b^2 expansion.
    sq = (
        np.sum(X1 ** 2, axis=1)[:, None]
        - 2.0 * X1 @ X2.T
        + np.sum(X2 ** 2, axis=1)[None, :]
    )
    r = np.sqrt(np.maximum(sq, 0.0))
    s5 = np.sqrt(5.0)
    a = s5 * r / lengthscale
    return signal_var * (1.0 + a + (a ** 2) / 3.0) * np.exp(-a)


class GPResidual:
    """Matern 5/2 GP with marginal-likelihood hyperparameter fitting.

        gp = GPResidual()
        gp.set_data(X, r)
        gp.optimize()
        mu, var = gp.predict(X_query)

    Hyperparameters are held in log space so the optimiser is unconstrained.
    """

    def __init__(
        self,
        lengthscale=100.0,
        signal_var=9.0,
        noise_var=4.5 ** 2,
        sparse_threshold=None,
        n_inducing=200,
        random_state=0,
    ):
        """Args:
            lengthscale: initial isotropic lengthscale, metres.
            signal_var: initial signal variance, dB^2.
            noise_var: initial noise variance, dB^2; the default matches the combined
                shadow and measurement noise used by the SMC likelihood.
            sparse_threshold: training-set size above which the sparse path is used.
                None means always exact.
            n_inducing: number of inducing points for the sparse path.
            random_state: seed for inducing-point selection.
        """
        self.log_lengthscale = np.log(lengthscale)
        self.log_signal_var = np.log(signal_var)
        self.log_noise_var = np.log(noise_var)

        self.sparse_threshold = sparse_threshold
        self.n_inducing = n_inducing
        self._rng = np.random.default_rng(random_state)

        self.X = None
        self.y = None
        # Cached factorisation for prediction (rebuilt on set_data/optimize).
        self._cache = None

    # -- hyperparameter properties -----------------------------------------
    @property
    def lengthscale(self):
        return float(np.exp(self.log_lengthscale))

    @property
    def signal_var(self):
        return float(np.exp(self.log_signal_var))

    @property
    def noise_var(self):
        return float(np.exp(self.log_noise_var))

    def _use_sparse(self):
        return (
            self.sparse_threshold is not None
            and self.X is not None
            and self.X.shape[0] > self.sparse_threshold
        )

    # -- data ---------------------------------------------------------------
    def set_data(self, X, y):
        """Set the training inputs (locations) and targets (residuals)."""
        self.X = np.atleast_2d(np.asarray(X, dtype=float))
        self.y = np.asarray(y, dtype=float).ravel()
        self._cache = None
        return self

    # -- marginal likelihood ------------------------------------------------
    def _neg_log_marginal_likelihood(self, theta):
        """Negative log marginal likelihood as a function of log-hyperparams.

        log p(y|X) = -1/2 y^T K_y^-1 y - 1/2 log|K_y| - n/2 log(2 pi),
        with K_y = K + sigma_n^2 I. Returned negated for minimisation.
        """
        log_l, log_sf, log_sn = theta
        l = np.exp(log_l)
        sf = np.exp(log_sf)
        sn = np.exp(log_sn)

        n = self.X.shape[0]
        K = matern52(self.X, self.X, l, sf)
        K[np.diag_indices_from(K)] += sn + _JITTER

        try:
            c, low = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e25
        alpha = cho_solve((c, low), self.y)
        # log|K_y| = 2 sum(log(diag(L)))
        log_det = 2.0 * np.sum(np.log(np.diag(c)))
        nll = 0.5 * self.y @ alpha + 0.5 * log_det + 0.5 * n * np.log(2.0 * np.pi)
        return float(nll)

    def optimize(self, restarts=1):
        """Refit hyperparameters by maximising the log marginal likelihood.

        Warm-started from the current hyperparameters (the previous optimum),
        which is what makes the streaming refit-every-200-obs schedule cheap.
        Optionally add random restarts to reduce sensitivity to local optima.

        Returns self.
        """
        if self.X is None or self.X.shape[0] < 3:
            # Not enough data to fit meaningfully; keep the warm-start values.
            self._build_cache()
            return self

        x0 = np.array(
            [self.log_lengthscale, self.log_signal_var, self.log_noise_var]
        )
        candidates = [x0]
        for _ in range(max(0, restarts - 1)):
            candidates.append(x0 + self._rng.normal(0.0, 1.0, size=3))

        best = None
        for start in candidates:
            res = minimize(
                self._neg_log_marginal_likelihood,
                start,
                method="L-BFGS-B",
                # Keep hyperparameters in a sane range (log space).
                bounds=[
                    (np.log(1.0), np.log(1e4)),    # lengthscale 1 m .. 10 km
                    (np.log(1e-3), np.log(1e4)),   # signal var
                    (np.log(1e-3), np.log(1e3)),   # noise var
                ],
            )
            if best is None or res.fun < best.fun:
                best = res

        self.log_lengthscale, self.log_signal_var, self.log_noise_var = best.x
        self._build_cache()
        return self

    # -- prediction ---------------------------------------------------------
    def _build_cache(self):
        """Factorise the training kernel once so predict() is cheap."""
        if self.X is None or self.X.shape[0] == 0:
            self._cache = None
            return

        if self._use_sparse():
            self._build_sparse_cache()
            return

        l, sf, sn = self.lengthscale, self.signal_var, self.noise_var
        K = matern52(self.X, self.X, l, sf)
        K[np.diag_indices_from(K)] += sn + _JITTER
        c, low = cho_factor(K, lower=True)
        alpha = cho_solve((c, low), self.y)
        self._cache = {"mode": "exact", "chol": (c, low), "alpha": alpha}

    def _build_sparse_cache(self):
        """Subset-of-Regressors sparse approximation with M inducing points.

        Predictive mean uses the low-rank approximation
            K_*Z (K_ZZ_noise)^-1 ... form, giving O(n M^2) cost.
        Inducing points Z are a random subset of the training inputs (cheap and
        adequate for a residual field with a single dominant lengthscale).
        """
        l, sf, sn = self.lengthscale, self.signal_var, self.noise_var
        n = self.X.shape[0]
        m = min(self.n_inducing, n)
        idx = self._rng.choice(n, size=m, replace=False)
        Z = self.X[idx]

        Kmm = matern52(Z, Z, l, sf)
        Kmm[np.diag_indices_from(Kmm)] += _JITTER
        Knm = matern52(self.X, Z, l, sf)

        # SoR: Sigma = Kmm + (1/sn) Knm^T Knm ; mean weights on inducing basis.
        A = Kmm + (Knm.T @ Knm) / sn
        c, low = cho_factor(A, lower=True)
        b = cho_solve((c, low), Knm.T @ self.y) / sn
        self._cache = {
            "mode": "sparse",
            "Z": Z,
            "Kmm_chol": cho_factor(Kmm, lower=True),
            "A_chol": (c, low),
            "b": b,
        }

    def predict(self, X_star, return_var=True):
        """Posterior mean (and optionally variance) at query locations.

        Args:
            X_star: (M, D) query locations.
            return_var: if True also return the marginal predictive variance
                (including observation noise-free latent variance).

        Returns:
            mu: (M,) posterior mean.
            var: (M,) posterior variance (only if return_var).
        """
        X_star = np.atleast_2d(np.asarray(X_star, dtype=float))
        l, sf, sn = self.lengthscale, self.signal_var, self.noise_var

        if self._cache is None:
            self._build_cache()
        if self._cache is None:
            # No data at all: prior mean 0, prior variance = signal_var.
            mu = np.zeros(X_star.shape[0])
            if return_var:
                return mu, np.full(X_star.shape[0], sf)
            return mu

        if self._cache["mode"] == "sparse":
            Z = self._cache["Z"]
            Ksm = matern52(X_star, Z, l, sf)
            mu = Ksm @ self._cache["b"]
            if not return_var:
                return mu
            # SoR predictive variance: Ksm A^-1 Ksm^T (approximate).
            v = cho_solve(self._cache["A_chol"], Ksm.T)
            var = np.einsum("ij,ji->i", Ksm, v)
            return mu, np.maximum(var, 1e-9)

        # Exact GP.
        Ks = matern52(X_star, self.X, l, sf)
        mu = Ks @ self._cache["alpha"]
        if not return_var:
            return mu
        c, low = self._cache["chol"]
        v = cho_solve((c, low), Ks.T)
        kss = sf  # Matérn 5/2 self-covariance = signal_var
        var = kss - np.einsum("ij,ji->i", Ks, v)
        return mu, np.maximum(var, 1e-9)

    def log_marginal_likelihood(self):
        """Current log marginal likelihood (useful for kernel comparison)."""
        if self.X is None or self.X.shape[0] < 1:
            return float("nan")
        theta = np.array(
            [self.log_lengthscale, self.log_signal_var, self.log_noise_var]
        )
        return -self._neg_log_marginal_likelihood(theta)
