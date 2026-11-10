import numpy as np
import pytest

from fourierslm import focus_grid, intensity, phase, propagate, slm_field, slm_grid

LAM, F, PITCH, W, P = 1030e-9, 0.2, 8e-6, 0.4e-3, 2.0
X, Y = slm_grid(n=512, pitch=PITCH)  # SLM edge at 5.1w
x, y = focus_grid(X, Y, LAM, F)
W0 = LAM * F / (np.pi * W)  # waist at the focus
ZR = np.pi * W0**2 / LAM
I0 = 2 * P / (np.pi * W0**2)


@pytest.fixture(scope="module")
def E():
    return slm_field(intensity.gaussian(X, Y, W, P), 0)


def analytic(x, y, z, L):
    """Gaussian beam, radius W0 and curvature (1 − L/F)/F in the back focal plane, then q0 → q0 + z."""
    k = 2 * np.pi / LAM
    q0 = 1 / ((1 - L / F) / F + 2j / (k * W0**2))
    return np.sqrt(I0) / 1j * q0 / (q0 + z) * np.exp(1j * k * (x[None, :] ** 2 + y[:, None] ** 2) / (2 * (q0 + z)))


def scan(E, z, half_width=1.6e-3, **kw):
    """Focal plane by focus on ±half_width, then free to the planes z, with its coordinates."""
    u, _ = focus_grid(X, Y, LAM, F, window=2 * half_width)
    return propagate.free(propagate.focus(E, X, Y, u, u, LAM, F, **kw), u[1] - u[0], LAM, z), u


@pytest.mark.parametrize("slm_distance", [None, 0.12])
def test_gaussian_focus(E, slm_distance):
    u, v = focus_grid(X, Y, LAM, F, window=6 * W0)
    G = propagate.focus(E, X, Y, u, v, LAM, F, slm_distance)
    ref = analytic(u, u, 0.0, F if slm_distance is None else slm_distance)
    assert np.allclose(G, ref, rtol=0, atol=1e-11 * np.abs(ref).max())


def test_window_is_the_crop_of_the_plane(E):
    G = propagate.focus(E, X, Y, x, y, LAM, F, 0.12)
    assert G.shape == (len(y), len(x)) == E.shape
    u, v = focus_grid(X, Y, LAM, F, window=2e-3)
    keep = np.abs(x) <= 1e-3
    assert np.array_equal(u, x[keep]) and u[len(u) // 2] == 0
    assert np.array_equal(propagate.focus(E, X, Y, u, v, LAM, F, 0.12), G[np.ix_(keep, keep)])


def test_pad_changes_the_sampling_not_the_field(E):
    """pad 1 has twice the pitch of pad 2, so every other sample of pad 2 is a sample of pad 1."""
    x1, y1 = focus_grid(X, Y, LAM, F, pad=1)
    G1, G2 = propagate.focus(E, X, Y, x1, y1, LAM, F), propagate.focus(E, X, Y, x, y, LAM, F)
    n = len(X)
    s1, s2 = slice(n // 4, 3 * n // 4), slice(0, n, 2)
    assert np.allclose(x1[s1], x[s2], rtol=1e-12, atol=0)
    assert np.allclose(G1[s1, s1], G2[s2, s2], rtol=0, atol=1e-12 * np.abs(G2).max())


def test_focus_rejects_a_wrong_grid(E):
    x_other, _ = focus_grid(X, Y, 800e-9, F)
    for xs, ys in ((x_other, x_other), (x, -y), (x[:200], y[:200])):
        with pytest.raises(ValueError):
            propagate.focus(E, X, Y, xs, ys, LAM, F)


def test_odd_n():
    Xo, Yo = slm_grid(n=301, pitch=PITCH)
    w, w0 = 0.2e-3, LAM * F / (np.pi * 0.2e-3)
    for pad in (2, 1.5):
        xo, yo = focus_grid(Xo, Yo, LAM, F, pad)
        I = np.abs(propagate.focus(slm_field(intensity.gaussian(Xo, Yo, w), 0), Xo, Yo, xo, yo, LAM, F)) ** 2
        assert I.shape == (301, 301)
        assert np.allclose(I, I.max() * np.exp(-2 * (xo[None, :] ** 2 + yo[:, None] ** 2) / w0**2), atol=1e-6 * I.max())


@pytest.mark.parametrize("pad", [1, 2])
def test_focus_conserves_power(E, pad):
    xp, yp = focus_grid(X, Y, LAM, F, pad)
    G = propagate.focus(E, X, Y, xp, yp, LAM, F, 0.12)
    p_slm = np.sum(np.abs(E) ** 2) * PITCH**2
    assert np.isclose(np.sum(np.abs(G) ** 2) * (xp[1] - xp[0]) ** 2, p_slm, rtol=1e-12)
    assert np.isclose(p_slm, P, rtol=1e-6)


def test_focus_matches_free_space_and_lens():
    """SLM at L ≠ f: FT with curvature against angular spectrum, thin lens, angular spectrum."""
    lam, L, n, N = 1e-6, 0.03, 2048, 900
    f = N * PITCH**2 / lam  # the pitch λf/(N·pitch) at the focus is then the SLM pitch
    Xs, Ys = slm_grid(n=n, pitch=PITCH)
    E = slm_field(intensity.gaussian(Xs, Ys, 0.25e-3), phase.vortex(Xs, Ys, 1))
    G = propagate.free(E, PITCH, lam, L) * np.exp(1j * phase.lens(Xs, Ys, lam, f))
    G = propagate.free(G, PITCH, lam, f)
    c = slice(n // 2 - 128, n // 2 + 128)  # the beam is zero outside
    s = slice(n // 2 - 40, n // 2 + 41)
    M = propagate.focus(E[c, c], Xs[c], Ys[c], Xs[s], Ys[s], lam, f, slm_distance=L)
    assert np.abs(G[s, s] - M).max() < 1e-3 * np.abs(M).max()


@pytest.mark.parametrize("slm_distance", [None, 0.12])
def test_free_scan_continues_focus(E, slm_distance):
    """A z scan by free from the focal window, against the analytic beam."""
    z = np.linspace(-2, 2, 5) * ZR
    fields, u = scan(E, z, slm_distance=slm_distance)
    s = np.abs(u) <= 3 * W0
    ref = np.array([analytic(u[s], u[s], zi, F if slm_distance is None else slm_distance) for zi in z])
    assert np.abs(fields[:, s][:, :, s] - ref).max() < 2e-6 * np.abs(ref).max()


def test_z_scan_intensity_and_gouy_phase(E):
    z = np.linspace(-3 * ZR, 3 * ZR, 13)
    fields, u = scan(E, z, half_width=2e-3)
    on_axis = fields[:, len(u) // 2, len(u) // 2]
    assert np.allclose(np.abs(on_axis) ** 2, I0 / (1 + (z / ZR) ** 2), rtol=5e-6)  # free is not paraxial
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
    M = propagate.focus(E32, X, Y, x, y, LAM, F, slm_distance=0.12)
    assert M.dtype == np.complex64
    ref = propagate.focus(E, X, Y, x, y, LAM, F, slm_distance=0.12)
    assert np.abs(M - ref).max() < 1e-5 * np.abs(ref).max()
    assert propagate.free(E32, PITCH, LAM, 0.01).dtype == np.complex64
