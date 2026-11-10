import numpy as np
import pytest
from scipy.constants import c, epsilon_0

from fourierslm import beam, grid, phase, propagate

LAM, F, PITCH, W, P = 1030e-9, 0.2, 8e-6, 0.4e-3, 2.0
X, Y = grid(512, PITCH), grid(384, PITCH)
W0 = LAM * F / (np.pi * W)  # waist at the focus
ZR = np.pi * W0**2 / LAM
I0 = 2 * P / (np.pi * W0**2)


@pytest.fixture(scope="module")
def E():
    return beam.gaussian(X, Y, W, P).astype(complex)


def analytic(x, y, z, L):
    """Gaussian beam, radius W0 and curvature (1 − L/F)/F in the back focal plane, then q0 → q0 + z."""
    k = 2 * np.pi / LAM
    q0 = 1 / ((1 - L / F) / F + 2j / (k * W0**2))
    E_f = np.sqrt(2 * I0 / (epsilon_0 * c)) / 1j  # on-axis field in the back focal plane
    return E_f * q0 / (q0 + z) * np.exp(1j * k * (x[None, :] ** 2 + y[:, None] ** 2) / (2 * (q0 + z)))


def scan(E, z, half_width=1.6e-3, **kw):
    """Focal plane by focus, cropped to ±half_width, then free to the planes z, with the crop coordinates."""
    G, x, _ = propagate.focus(E, PITCH, LAM, F, **kw)
    keep = np.abs(x) < half_width
    u = x[keep]
    return propagate.free(G[np.ix_(keep, keep)], u[1] - u[0], LAM, z), u


@pytest.mark.parametrize("slm_distance", [None, 0.12])
def test_gaussian_focus(E, slm_distance):
    G, x, y = propagate.focus(E, PITCH, LAM, F, slm_distance)
    sx, sy = np.abs(x) <= 3 * W0, np.abs(y) <= 3 * W0
    ref = analytic(x[sx], y[sy], 0.0, F if slm_distance is None else slm_distance)
    assert np.allclose(G[np.ix_(sy, sx)], ref, rtol=0, atol=1e-7 * np.abs(ref).max())  # SLM edge at 3.8w


def test_grid_is_square():
    for pad in (2, 1.5):
        G, x, y = propagate.focus(beam.gaussian(X, Y, W), PITCH, LAM, F, pad=pad)
        assert G.shape == (len(y), len(x)) and len(x) == len(y) >= pad * len(X) and np.array_equal(x, y)
        assert np.isclose(x[1] - x[0], LAM * F / (len(x) * PITCH))


def test_odd_shape():
    Xo, Yo = grid(301, PITCH), grid(255, PITCH)
    w, w0 = 0.2e-3, LAM * F / (np.pi * 0.2e-3)
    for pad in (2, 1.5):
        G, x, y = propagate.focus(beam.gaussian(Xo, Yo, w), PITCH, LAM, F, pad=pad)
        assert len(x) % 2 == 0
        I = beam.intensity(G)
        assert np.allclose(I, I.max() * np.exp(-2 * (x[None, :] ** 2 + y[:, None] ** 2) / w0**2), atol=1e-6 * I.max())


def test_focus_conserves_power(E):
    for pad in (2, 1.5):
        G, x, y = propagate.focus(E, PITCH, LAM, F, 0.12, pad)
        p_slm = beam.intensity(E).sum() * PITCH**2
        assert np.isclose(beam.intensity(G).sum() * (x[1] - x[0]) * (y[1] - y[0]), p_slm, rtol=1e-12)
        assert np.isclose(p_slm, P, rtol=1e-6)


def test_focus_matches_free_space_and_lens():
    """SLM at L ≠ f: FT with curvature against angular spectrum, thin lens, angular spectrum."""
    lam, L, n, N = 1e-6, 0.03, 2048, 900
    f = N * PITCH**2 / lam  # the pitch λf/(N·pitch) at the focus is then the SLM pitch
    Xs = grid(n, PITCH)
    E = beam.gaussian(Xs, Xs, 0.25e-3) * np.exp(1j * phase.vortex(Xs, Xs, 1))
    G = propagate.free(E, PITCH, lam, L) * np.exp(1j * phase.lens(Xs, Xs, lam, f))
    G = propagate.free(G, PITCH, lam, f)
    c = slice(n // 2 - 128, n // 2 + 128)  # the beam is zero outside
    M, x, _ = propagate.focus(E[c, c], PITCH, lam, f, slm_distance=L, pad=N / 256)
    assert np.isclose(x[1] - x[0], PITCH)
    s, sm = slice(n // 2 - 40, n // 2 + 41), slice(N // 2 - 40, N // 2 + 41)
    assert np.abs(G[s, s] - M[sm, sm]).max() < 1e-3 * np.abs(M).max()


@pytest.mark.parametrize("slm_distance", [None, 0.12])
def test_free_scan_continues_focus(E, slm_distance):
    """A z scan by free from the cropped focal plane, against the analytic beam."""
    z = np.linspace(-2, 2, 5) * ZR
    fields, u = scan(E, z, slm_distance=slm_distance)
    s = np.abs(u) <= 3 * W0
    ref = np.array([analytic(u[s], u[s], zi, F if slm_distance is None else slm_distance) for zi in z])
    assert np.abs(fields[:, s][:, :, s] - ref).max() < 2e-6 * np.abs(ref).max()


def test_z_scan_intensity_and_gouy_phase(E):
    z = np.linspace(-3 * ZR, 3 * ZR, 13)
    fields, u = scan(E, z, half_width=2e-3)
    on_axis = fields[:, len(u) // 2, len(u) // 2]
    assert np.allclose(beam.intensity(on_axis), I0 / (1 + (z / ZR) ** 2), rtol=5e-6)  # free is not paraxial
    assert np.allclose(np.angle(on_axis * np.exp(1j * (np.pi / 2 + np.arctan(z / ZR)))), 0, atol=2e-6)


def test_lens_on_the_slm_moves_the_focus(E):
    """A lens of focal length f_slm on the SLM moves the focus to z = −f²/f_slm."""
    f_slm = 20.0
    z = np.linspace(-6, 4, 201) * ZR
    fields, u = scan(E * np.exp(1j * phase.lens(X, Y, LAM, f_slm)), z, half_width=4e-3)
    on_axis = fields[:, len(u) // 2, len(u) // 2]
    assert abs(z[np.argmax(np.abs(on_axis))] + F**2 / f_slm) <= z[1] - z[0]


def test_free_is_unitary_and_reversible(E):
    G = propagate.free(E, PITCH, LAM, 0.05, bandlimit=False)
    assert np.isclose(np.sum(np.abs(G) ** 2), np.sum(np.abs(E) ** 2), rtol=1e-12)
    assert np.allclose(propagate.free(G, PITCH, LAM, -0.05, bandlimit=False), E, rtol=0, atol=1e-12 * np.abs(E).max())


def test_free_scan_and_single_planes_agree(E):
    z = np.array([-0.02, 0.0, 0.05])
    fields = propagate.free(E, PITCH, LAM, z)
    for i, zi in enumerate(z):
        assert np.array_equal(fields[i], propagate.free(E, PITCH, LAM, zi))


def test_free_zero_distance(E):
    assert np.allclose(propagate.free(E, PITCH, LAM, 0.0), E, rtol=0, atol=1e-12 * np.abs(E).max())


def test_complex64(E):
    E32 = E.astype(np.complex64)
    M = propagate.focus(E32, PITCH, LAM, F, slm_distance=0.12)[0]
    assert M.dtype == np.complex64
    ref = propagate.focus(E, PITCH, LAM, F, slm_distance=0.12)[0]
    assert np.abs(M - ref).max() < 1e-5 * np.abs(ref).max()
    assert propagate.free(E32, PITCH, LAM, 0.01).dtype == np.complex64
