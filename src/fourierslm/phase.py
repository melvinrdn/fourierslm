"""Phase patterns for the SLM [rad].

The patterns are continuous, not wrapped, so they add. Wrap or quantize the sum only for display. Coordinates X and Y
are 1D, the arrays have the shape (len(Y), len(X)), and angles are in radians. The sign convention is the plane wave
exp[i(kz − ωt)], so a pattern that grows along +X steers the beam toward +x.
"""

import numpy as np


def _xy(X, Y, center):
    """Coordinates relative to center, broadcast to (len(Y), len(X))."""
    return np.asarray(X)[None, :] - center[0], np.asarray(Y)[:, None] - center[1]


def lens(
    X: np.ndarray, Y: np.ndarray, wavelength: float, f: float, center: tuple[float, float] = (0.0, 0.0)
) -> np.ndarray:
    """Thin lens, −k(X² + Y²)/(2f).

    Parameters
    ----------
    X, Y : ndarray
        1D coordinates [m].
    wavelength : float
        Wavelength [m].
    f : float
        Focal length [m], positive for a converging lens.
    center : tuple of float, default (0, 0)
        Center of the lens [m].

    Returns
    -------
    ndarray
        Phase [rad].
    """
    x, y = _xy(X, Y, center)
    return -np.pi / (wavelength * f) * (x**2 + y**2)


def axicon(
    X: np.ndarray, Y: np.ndarray, wavelength: float, angle: float, center: tuple[float, float] = (0.0, 0.0)
) -> np.ndarray:
    """Axicon, −kR sin(angle), which focuses to a Bessel beam.

    Parameters
    ----------
    X, Y : ndarray
        1D coordinates [m].
    wavelength : float
        Wavelength [m].
    angle : float
        Half angle of the cone of rays [rad].
    center : tuple of float, default (0, 0)
        Apex [m].

    Returns
    -------
    ndarray
        Phase [rad].
    """
    x, y = _xy(X, Y, center)
    return -2 * np.pi / wavelength * np.sin(angle) * np.hypot(x, y)


def grating(X: np.ndarray, Y: np.ndarray, wavelength: float, angle: float, azimuth: float = 0.0) -> np.ndarray:
    """Blazed grating, k sin(angle)·(X cos ψ + Y sin ψ).

    Behind a lens of focal length f, it moves the focus to f sin(angle) along the azimuth ψ.

    Parameters
    ----------
    X, Y : ndarray
        1D coordinates [m].
    wavelength : float
        Wavelength [m].
    angle : float
        Deflection angle [rad].
    azimuth : float, default 0
        Direction ψ of the deflection [rad], 0 along +x and π/2 along +y.

    Returns
    -------
    ndarray
        Phase [rad].
    """
    x, y = _xy(X, Y, (0.0, 0.0))
    return 2 * np.pi / wavelength * np.sin(angle) * (x * np.cos(azimuth) + y * np.sin(azimuth))


def vortex(
    X: np.ndarray, Y: np.ndarray, charge: float, center: tuple[float, float] = (0.0, 0.0), rotation: float = 0.0
) -> np.ndarray:
    """Vortex, ℓψ.

    Parameters
    ----------
    X, Y : ndarray
        1D coordinates [m].
    charge : float
        Topological charge ℓ, integer or fractional.
    center : tuple of float, default (0, 0)
        Center of the vortex [m].
    rotation : float, default 0
        Azimuth of the 2πℓ step is π + rotation [rad]. It only shows for a fractional charge.

    Returns
    -------
    ndarray
        Phase [rad], with the azimuth ψ − rotation taken in (−π, π].
    """
    x, y = _xy(X, Y, center)
    psi = np.arctan2(y, x)
    if rotation:
        psi = np.angle(np.exp(1j * (psi - rotation)))
    return charge * psi


def binary(X: np.ndarray, Y: np.ndarray, period: float, depth: float = np.pi, azimuth: float = 0.0) -> np.ndarray:
    """Binary grating, stripes of 0 and depth.

    Parameters
    ----------
    X, Y : ndarray
        1D coordinates [m].
    period : float
        Period, two stripes [m].
    depth : float, default π
        Phase of every other stripe [rad].
    azimuth : float, default 0
        Direction ψ across the stripes [rad], 0 for stripes along y.

    Returns
    -------
    ndarray
        Phase [rad], 0 on the stripe that starts at the origin.
    """
    x, y = _xy(X, Y, (0.0, 0.0))
    u = x * np.cos(azimuth) + y * np.sin(azimuth)
    return depth * (np.floor(2 * u / period + 1e-9) % 2)  # pixels on an edge go to the next stripe, not by rounding


def radial_step(
    X: np.ndarray,
    Y: np.ndarray,
    radius: float,
    inner: float = 0.0,
    outer: float = np.pi,
    center: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Phase step at a radius.

    Parameters
    ----------
    X, Y : ndarray
        1D coordinates [m].
    radius : float
        Radius of the step [m].
    inner, outer : float, default 0 and π
        Phase inside and outside the radius [rad].
    center : tuple of float, default (0, 0)
        Center of the step [m].

    Returns
    -------
    ndarray
        Phase [rad].
    """
    x, y = _xy(X, Y, center)
    return np.where(x**2 + y**2 <= radius**2, inner, outer)


def wrap(phase: np.ndarray) -> np.ndarray:
    """Wrap a phase into [0, 2π).

    Parameters
    ----------
    phase : ndarray
        Phase [rad].

    Returns
    -------
    ndarray
        Phase modulo 2π [rad].
    """
    return np.mod(phase, 2 * np.pi)


def quantize(phase: np.ndarray, bits: int) -> np.ndarray:
    """Round a phase to the gray levels of an SLM.

    Parameters
    ----------
    phase : ndarray
        Phase [rad].
    bits : int
        Bit depth of the SLM, 2^bits levels over 2π.

    Returns
    -------
    ndarray
        Phase on the levels 2πn/2^bits, n = 0 … 2^bits − 1 [rad]. 2π rounds to 0.
    """
    levels = 2**bits
    return 2 * np.pi / levels * (np.round(wrap(phase) * levels / (2 * np.pi)) % levels)
