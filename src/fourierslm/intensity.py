"""Intensity of the beam on the SLM [W/m²].

For a pulse, give the peak power, `fourierslm.peak_power`, to get the peak intensity, on the SLM and in every plane
after it. Coordinates X and Y are 1D, and the arrays have the shape (len(Y), len(X)).
"""

import numpy as np


def _r2(X, Y, center):
    """Squared distance to center, shape (len(Y), len(X))."""
    return (np.asarray(X)[None, :] - center[0]) ** 2 + (np.asarray(Y)[:, None] - center[1]) ** 2


def gaussian(
    X: np.ndarray, Y: np.ndarray, w: float, power: float = 1.0, center: tuple[float, float] = (0.0, 0.0)
) -> np.ndarray:
    """Gaussian beam, 2P/(πw²)·exp(−2r²/w²).

    Parameters
    ----------
    X, Y : ndarray
        1D coordinates [m].
    w : float
        Radius at 1/e² of the intensity [m].
    power : float, default 1
        Power P [W], the peak power of a pulse for its peak intensity.
    center : tuple of float, default (0, 0)
        Center of the beam [m].

    Returns
    -------
    ndarray
        Intensity [W/m²].
    """
    return 2 * power / (np.pi * w**2) * np.exp(-2 * _r2(X, Y, center) / w**2)


def flattop(
    X: np.ndarray, Y: np.ndarray, radius: float, power: float = 1.0, center: tuple[float, float] = (0.0, 0.0)
) -> np.ndarray:
    """Uniform beam cut by a circular aperture, P/(π radius²) inside the disk and 0 outside.

    Parameters
    ----------
    X, Y : ndarray
        1D coordinates [m].
    radius : float
        Radius of the aperture [m].
    power : float, default 1
        Power P [W] of the continuous disk, the peak power of a pulse for its peak intensity. The pixelated edge
        changes it slightly.
    center : tuple of float, default (0, 0)
        Center of the beam [m].

    Returns
    -------
    ndarray
        Intensity [W/m²].
    """
    return power / (np.pi * radius**2) * (_r2(X, Y, center) <= radius**2)
