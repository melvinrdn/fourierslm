import numpy as np
import pytest

import fourierslm as fs
from fourierslm import beam, phase, propagate

LAM, F, PITCH, W, P = 1030e-9, 0.2, 8e-6, 0.4e-3, 2.0
X, Y = fs.grid(512, PITCH), fs.grid(384, PITCH)
W0 = LAM * F / (np.pi * W)  # waist at the focus
ZR = np.pi * W0**2 / LAM
I0 = 2 * P / (np.pi * W0**2)
x = np.linspace(-3 * W0, 3 * W0, 121)


@pytest.fixture(scope="module")
def E():
    return beam.gaussian(X, Y, W, P).astype(complex)


def test_gaussian_focus(E):
    I = beam.intensity(propagate.focus(E, X, Y, x, x, LAM, F))
    assert np.allclose(I, I0 * np.exp(-2 * (x[None, :] ** 2 + x[:, None] ** 2) / W0**2), rtol=0, atol=1e-6 * I0)


def test_z_scan_intensity_and_gouy_phase(E):
    z = np.linspace(-3 * ZR, 3 * ZR, 13)
    axis = propagate.focus(E, X, Y, np.zeros(1), np.zeros(1), LAM, F, z)[:, 0, 0]
    assert np.allclose(beam.intensity(axis), I0 / (1 + (z / ZR) ** 2), rtol=1e-6)
    assert np.allclose(np.angle(axis * np.exp(1j * (np.pi / 2 + np.arctan(z / ZR)))), 0, atol=1e-6)


def test_scalar_z_and_scan_agree(E):
    z = np.array([-ZR, 0.5 * ZR])
    scan = propagate.focus(E, X, Y, x, x[:50], LAM, F, z, slm_distance=0.1)
    for i, zi in enumerate(z):
        assert np.array_equal(scan[i], propagate.focus(E, X, Y, x, x[:50], LAM, F, zi, slm_distance=0.1))


def test_lens_on_the_slm_moves_the_focus(E):
    """A lens of focal length f_slm on the SLM moves the focus to z = −f²/f_slm."""
    f_slm = 20.0
    z = np.linspace(-6, 4, 201) * ZR
    axis = propagate.focus(E * np.exp(1j * phase.lens(X, Y, LAM, f_slm)), X, Y, np.zeros(1), np.zeros(1), LAM, F, z)
    assert abs(z[np.argmax(np.abs(axis[:, 0, 0]))] + F**2 / f_slm) <= z[1] - z[0]


@pytest.mark.parametrize(("z", "slm_distance"), [(0.0, None), (0.0, 0.12), (1.5 * ZR, 0.12)])
def test_fft_matches_mft(E, z, slm_distance):
    G, xf, yf = propagate.focus_fft(E, PITCH, LAM, F, z, slm_distance)
    sx, sy = slice(len(xf) // 2 - 30, len(xf) // 2 + 31), slice(len(yf) // 2 - 20, len(yf) // 2 + 21)
    M = propagate.focus(E, X, Y, xf[sx], yf[sy], LAM, F, z, slm_distance)
    assert np.allclose(G[sy, sx], M, rtol=0, atol=1e-10 * np.abs(M).max())


def test_fft_odd_shape():
    Xo, Yo = fs.grid(301, PITCH), fs.grid(255, PITCH)
    Eo = beam.gaussian(Xo, Yo, 0.2e-3, center=(30e-6, -20e-6)) * np.exp(1j * phase.vortex(Xo, Yo, 1))
    G, xf, yf = propagate.focus_fft(Eo, PITCH, LAM, F, 0.0, 0.12, pad=1.5)
    assert G.shape[0] % 2 == 0 and G.shape[1] % 2 == 0
    assert np.allclose(G, propagate.focus(Eo, Xo, Yo, xf, yf, LAM, F, 0.0, 0.12), rtol=0, atol=1e-9 * np.abs(G).max())


def test_fft_conserves_power(E):
    G, xf, yf = propagate.focus_fft(E, PITCH, LAM, F, 0.3 * ZR, 0.12)
    p_slm = beam.intensity(E).sum() * PITCH**2
    assert np.isclose(beam.intensity(G).sum() * (xf[1] - xf[0]) * (yf[1] - yf[0]), p_slm, rtol=1e-12)
    assert np.isclose(p_slm, P, rtol=1e-6)


def test_focus_matches_free_space_and_lens():
    """SLM at L ≠ f, plane z ≠ 0: Collins integral against angular spectrum, thin lens, angular spectrum."""
    lam, f, L, n = 1e-6, 0.05, 0.03, 2048
    Xs = fs.grid(n, PITCH)
    w0 = lam * f / (np.pi * 0.25e-3)
    E = beam.gaussian(Xs, Xs, 0.25e-3) * np.exp(1j * phase.vortex(Xs, Xs, 1))
    z = 1.5 * np.pi * w0**2 / lam
    G = propagate.free(E, PITCH, lam, L) * np.exp(1j * phase.lens(Xs, Xs, lam, f))
    G = propagate.free(G, PITCH, lam, f + z)
    s = slice(n // 2 - 40, n // 2 + 41)
    M = propagate.focus(E, Xs, Xs, Xs[s], Xs[s], lam, f, z, slm_distance=L)
    assert np.abs(G[s, s] - M).max() < 1e-3 * np.abs(M).max()


def test_free_is_unitary_and_reversible(E):
    G = propagate.free(E, PITCH, LAM, 0.05, bandlimit=False)
    assert np.isclose(np.sum(np.abs(G) ** 2), np.sum(np.abs(E) ** 2), rtol=1e-12)
    assert np.allclose(propagate.free(G, PITCH, LAM, -0.05, bandlimit=False), E, rtol=0, atol=1e-12 * np.abs(E).max())


def test_free_scan_and_single_planes_agree(E):
    z = np.array([-0.02, 0.0, 0.05])
    scan = propagate.free(E, PITCH, LAM, z)
    for i, zi in enumerate(z):
        assert np.array_equal(scan[i], propagate.free(E, PITCH, LAM, zi))


def test_free_scan_continues_focus(E):
    """A z scan by free from the focal plane, against the exact focus in each plane."""
    u = fs.grid(256, 12e-6)
    z = np.linspace(-2, 2, 5) * ZR
    scan = propagate.free(propagate.focus(E, X, Y, u, u, LAM, F, slm_distance=0.12), 12e-6, LAM, z)
    exact = propagate.focus(E, X, Y, u, u, LAM, F, z, slm_distance=0.12)
    assert np.abs(scan - exact).max() < 1e-6 * np.abs(exact).max()


def test_free_zero_distance(E):
    assert np.allclose(propagate.free(E, PITCH, LAM, 0.0), E, rtol=0, atol=1e-12 * np.abs(E).max())


def test_complex64(E):
    E32 = E.astype(np.complex64)
    M = propagate.focus(E32, X, Y, x, x, LAM, F, slm_distance=0.12)
    assert M.dtype == np.complex64
    ref = propagate.focus(E, X, Y, x, x, LAM, F, slm_distance=0.12)
    assert np.abs(M - ref).max() < 1e-5 * np.abs(ref).max()
    assert propagate.focus_fft(E32, PITCH, LAM, F)[0].dtype == np.complex64
    assert propagate.free(E32, PITCH, LAM, 0.01).dtype == np.complex64
