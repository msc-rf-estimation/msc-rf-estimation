"""Directional antenna gain patterns."""
import numpy as np


def directional_gain(theta, theta_main, theta_3db, G0_dB=0.0,
                     front_back_ratio_dB=None):
    """Gaussian directional antenna pattern, in dB.

        G_dB(theta) = G0_dB - 12 (delta / theta_3db)^2

    where delta is the offset from boresight wrapped into (-180, 180]. The
    coefficient 12 ~= 10 log10(exp(4 ln 2)) puts the half-power points at
    +/- theta_3db / 2.

    An unclamped Gaussian falls to roughly -108 dB at 180 degrees for a 60 degree
    beam, which is not physical. Supplying front_back_ratio_dB clamps the falloff
    at a realistic back-lobe depth:

        G_dB(theta) = G0_dB - min(12 (delta / theta_3db)^2, F_b)

    The clamp matters for the Scenario 2 filter: with unbounded nulls a back-lobe
    explanation costs so much likelihood that the front/back ambiguity disappears
    from the posterior.

    Args:
        theta: angle(s) in degrees, scalar or array.
        theta_main: boresight direction, degrees.
        theta_3db: full half-power beamwidth, degrees.
        G0_dB: peak gain at boresight, dB. Default 0 returns deviation from peak.
        front_back_ratio_dB: optional clamp depth in dB; None leaves the bare
            Gaussian. Broadcasts over particle arrays.

    Returns:
        Gain in dB, broadcast over the inputs.
    """
    theta = np.asarray(theta, dtype=float)

    # Compute the signed angular offset from the main beam, wrapped into the
    # range (-180, 180]. Without this wrap, a query at theta=350 with a main
    # beam at theta_main=10 would be reported as 340 degrees away, not 20.
    delta = (theta - theta_main + 180.0) % 360.0 - 180.0

    # Quadratic-in-angle falloff. At delta = +/- theta_3db / 2 this gives -3 dB;
    # at delta = +/- theta_3db it gives -12 dB; etc.
    falloff_dB = 12.0 * (delta / theta_3db) ** 2

    if front_back_ratio_dB is not None:
        # Clamp the falloff at the supplied back-lobe depth. Broadcasts over
        # particle arrays when F_b is per-particle.
        falloff_dB = np.minimum(falloff_dB, np.asarray(front_back_ratio_dB,
                                                       dtype=float))

    return G0_dB - falloff_dB
