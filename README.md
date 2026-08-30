# Spatial Estimation of Contested RF Environments

Source code accompanying an MSc dissertation on parametric–residual decomposition for
radio map estimation in contested spectrum. The study asks when decomposing an estimator
into an SMC-inferred parametric jammer layer plus a Gaussian-process residual outperforms
a monolithic Gaussian process, as a joint function of survey coverage and source
identifiability.

## What this code does

A **twin simulation** holds a high-fidelity truth model apart from a deliberately
impoverished learner model, so that no estimator is ever scored on data drawn from its own
forward model. The truth uses a log-distance path-loss exponent of 2.5, correlated shadow
fading and single knife-edge diffraction over a procedural digital elevation model; the
learner assumes free-space propagation, has no terrain term, and must infer the jammer.

Four estimators reconstruct the SINR map over a 2 km x 2 km arena from a sparse four-UAS
survey: a **combined** parametric-plus-GP-residual estimator, a **pure GP** fitted directly
to the raw observations, a **pure parametric** reconstruction, and a **no-jamming** floor.

## Layout

    contested_rf/propagation   log-distance path loss, Gaussian directional antenna,
                               knife-edge diffraction, SINR composition
    contested_rf/terrain       procedural DEM, per-path Fresnel-Kirchhoff parameters
    contested_rf/simulation    scenarios, jammers, shadow field, truth model, UAS survey
    contested_rf/estimators    3D and 6D particle filters, GP residual layer, combined
                               predictors, evaluation helpers
    contested_rf/metrics       reconstruction RMSE, credible-interval coverage,
                               paired-bootstrap statistics
    contested_rf/tests         117 unit and integration tests

Each experiment has a top-level script that writes a JSON checkpoint as it runs, so a long
sweep resumes rather than restarts:

    run_factorial.py             dense-coverage factorial (2 scenarios x 2 terrains)
    run_coverage.py              partial-coverage sweeps, flat and procedural terrain
    run_standoff.py              emitter placed outside the surveyed band
    compute_frontier.py          accuracy against wall-clock cost
    scenario1_calibration_sparse.py   credible-interval calibration
    build_paired_diff_figs.py    paired difference figures
    build_fig51_posterior.py     Scenario 2 posterior over beam bearing

Logs and checkpoints from the reported runs are in `results/`.

## Purpose-built components

Four parts are written for this study rather than taken off the shelf.

1. **The twin-simulation harness**, which keeps truth and learner in separate code paths so
   that model error, rather than familiarity with the generating process, is what is measured.
2. **The Gaussian process** (`estimators/gp_residual.py`), implemented directly against
   NumPy and SciPy rather than through a GP library so the linear algebra stays inspectable.
   Hyperparameters are fitted by maximising the log marginal likelihood with random restarts,
   warm-started from the previous optimum and refitted periodically to amortise the cubic
   cost, with a sparse subset-of-regressors approximation as a fallback.
3. **The six-dimensional Scenario 2 particle filter**
   (`estimators/particle_filter_s2.py`), which composes the known omnidirectional emitter and
   the unknown directional emitter in linear power before evaluating the likelihood, and
   clamps the antenna pattern at the stated front-to-back ratio so the front/back ambiguity
   is physically realistic rather than an artefact of an unbounded back lobe.
4. **Mode-collapse diagnostics** for that filter: the circular variance of the weighted
   posterior over beam bearing, and an antipodal hedge score measuring how much posterior
   weight sits in two opposing wedges. Together these separate a genuinely multi-modal
   posterior from a filter that has committed early to one mode.

## Running it

    pip install -r requirements.txt
    pytest                 # 117 tests
    python run_factorial.py

Python 3.10 or later. NumPy, SciPy and Matplotlib only; no GP or simulation frameworks.
