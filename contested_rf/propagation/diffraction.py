"""Single knife-edge diffraction loss (ITU-R P.526).

The dominant ridge on a path is idealised as a knife edge and the excess loss
computed from the Fresnel parameter

    v = h sqrt(2 (d1 + d2) / (lambda d1 d2))

with h the obstruction height above the direct line of sight and d1, d2 the
along-path distances to it. The ITU-R P.526 approximation is

    J(v) = 0                                              for v <= -0.78
    J(v) = 6.9 + 20 log10(sqrt((v - 0.1)^2 + 1) + v - 0.1) otherwise

This excess loss is added on top of the log-distance path loss. A single knife
edge under-models multi-ridge terrain; that residual model error is deliberate
and is what the GP residual layer must absorb.
"""
import numpy as np

_LIGHT_C = 3e8


def fresnel_parameter(h, d1, d2, freq_Hz):
    """Fresnel-Kirchhoff diffraction parameter v.

    Args:
        h: obstruction height above the line of sight, metres; positive when
            the obstruction protrudes above it.
        d1: transmitter-to-obstruction distance, metres.
        d2: obstruction-to-receiver distance, metres.
        freq_Hz: carrier frequency.

    Returns:
        v, broadcast over the inputs.
    """
    wavelength = _LIGHT_C / freq_Hz
    d1 = np.maximum(np.asarray(d1, dtype=float), 1e-6)
    d2 = np.maximum(np.asarray(d2, dtype=float), 1e-6)
    return h * np.sqrt(2.0 * (d1 + d2) / (wavelength * d1 * d2))


def knife_edge_loss_dB(v):
    """ITU-R P.526 single knife-edge diffraction loss (dB), given v.

    Vectorised. Loss is 0 dB in deep line-of-sight (v <= -0.78) and rises
    smoothly through ~6 dB at grazing (v = 0) to heavy loss for large v.
    """
    v = np.asarray(v, dtype=float)
    loss = 6.9 + 20.0 * np.log10(np.sqrt((v - 0.1) ** 2 + 1.0) + v - 0.1)
    return np.where(v <= -0.78, 0.0, loss)


def diffraction_loss_from_profile(tx_xy, rx_xy, obstruction_xy,
                                  obstruction_height, tx_height, rx_height,
                                  freq_Hz):
    """Excess diffraction loss in dB for a single obstruction on a path.

    h is measured from the obstruction top to the line of sight interpolated at
    the obstruction's horizontal position.

    Args:
        tx_xy, rx_xy, obstruction_xy: (x, y) positions, metres.
        obstruction_height: terrain top height at the obstruction, metres.
        tx_height, rx_height: antenna heights above datum, metres.
        freq_Hz: carrier frequency.

    Returns:
        Loss in dB, >= 0.
    """
    tx_xy = np.asarray(tx_xy, dtype=float)
    rx_xy = np.asarray(rx_xy, dtype=float)
    obstruction_xy = np.asarray(obstruction_xy, dtype=float)

    d_total = np.hypot(*(rx_xy - tx_xy))
    d1 = np.hypot(*(obstruction_xy - tx_xy))
    if d_total < 1e-6:
        return 0.0
    frac = np.clip(d1 / d_total, 0.0, 1.0)
    d2 = d_total - d1

    # Height of the direct LoS ray at the obstruction's along-path fraction.
    los_height = tx_height + frac * (rx_height - tx_height)
    h = obstruction_height - los_height

    v = fresnel_parameter(h, d1, d2, freq_Hz)
    return float(knife_edge_loss_dB(v))
