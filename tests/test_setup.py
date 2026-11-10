import numpy as np
import pytest

from fourierslm import focus_grid, intensity, peak_power, phase, slm_field, slm_grid

LAM, F, PITCH = 1030e-9, 0.2, 8e-6
X, Y = slm_grid(n=512, pitch=PITCH)


def test_slm_grid():
    X6, Y6 = slm_grid(12.0, n=6)
    assert np.array_equal(X6, [-6, -4, -2, 0, 2, 4]) and np.array_equal(X6, Y6)
    assert slm_grid(n=5, pitch=1.0)[0][2] == 0
    assert np.allclose(slm_grid(16e-3)[0], slm_grid(pitch=16e-3 / 2048)[0], rtol=1e-12)
    for bad in ({}, {"size": 1.0, "pitch": 1e-3}):
        with pytest.raises(ValueError):
            slm_grid(**bad)


@pytest.mark.parametrize("pad", [2, 1])
def test_focus_grid_detail_and_window(pad):
    """The pitch at the focus is λf/(pad·size), so the window n times it is λf/(pad·pitch)."""
    size, n = 10e-3, 512
    Xs, Ys = slm_grid(size, n)
    x, y = focus_grid(Xs, Ys, LAM, F, pad)
    assert len(x) == n and np.array_equal(x, y) and x[n // 2] == 0
    assert np.isclose(x[1] - x[0], LAM * F / (pad * size))
    with pytest.raises(ValueError):
        focus_grid(Xs, Ys[:-1], LAM, F)


def test_gaussian():
    I = intensity.gaussian(X, Y, 0.3e-3, power=2.0, center=(40e-6, -80e-6))
    assert I.shape == (512, 512)
    assert np.isclose(I.sum() * PITCH**2, 2.0, rtol=1e-9)
    assert np.isclose(I[256 - 10, 256 + 5], 2 * 2.0 / (np.pi * 0.3e-3**2))


def test_flattop():
    I = intensity.flattop(X, Y, 1e-3, power=0.5)
    assert np.isclose(I.max(), 0.5 / (np.pi * 1e-3**2))
    assert np.isclose(I.sum() * PITCH**2, 0.5, rtol=1e-2)
    assert I[256, 256 + 126] == 0 and I[256, 256 + 125] > 0


def test_slm_field():
    I, p = intensity.gaussian(X, Y, 0.3e-3), phase.vortex(X, Y, 1)
    E = slm_field(I, p)
    assert np.allclose(np.abs(E) ** 2, I, rtol=1e-12)
    assert np.allclose(np.angle(E * np.exp(-1j * p))[I > 0], 0, atol=1e-12)
    assert np.allclose(slm_field(I, 0), np.sqrt(I), rtol=1e-12)


def test_peak_power():
    """A Gaussian pulse of FWHM τ with this peak holds the energy."""
    tau, t = 200e-15, np.linspace(-1e-12, 1e-12, 20001)
    P = peak_power(1e-3, tau) * np.exp(-4 * np.log(2) * t**2 / tau**2)
    assert np.isclose(P.sum() * (t[1] - t[0]), 1e-3, rtol=1e-9)
