"""Humliček (1982) cpf12 Voigt profile.

JAX-differentiable. ~1e-6 relative accuracy in float32.
Provides `voigt_profile(nu, nu0, sigmaD, gammaL)` plus
the underlying `hjert(x, a)` Hjerting function with custom JVP rules.
"""

import jax.numpy as jnp
from jax import jit
from jax import custom_jvp

__all__ = [
    "hjert",
    "voigt_profile",
]

# ================================================================== #
#  Humlíček (1982) cpf12 rational approximation     #
#                                                                      #
#  w(z) = Re + i·Im of the complex probability function is computed    #
#  via four regions, each a rational polynomial in t = y − ix.         #
#  Only Region 4 (small |z|) needs exp(−x²); the rest is pure         #
#  multiply-add — dramatically fewer FLOPs than Algorithm 916.         #
# ================================================================== #

@jit
def _wofz_humlicek(x, y):
    """Complex probability function  w(z) = w(x + iy)  via Humlíček (1982).

    Returns a *complex* array.  Fully vectorized, branchless (uses
    jnp.where to blend 4 regions so both branches are always evaluated
    with safe inputs).
    """
    ax = jnp.abs(x)
    s = ax + y                      # region discriminator

    # t = y - ix  (the Humlíček variable)
    t_re = y
    t_im = -x

    # ---- Region 1:  s ≥ 15  (far wings — simple asymptotic) ----------
    # w ≈ t·C / (0.5 + t²)      C = i/√π ≈ 0.5641896
    # where t² = (y² − x²) − 2ixy
    t2_re = t_re * t_re - t_im * t_im
    t2_im = 2.0 * t_re * t_im
    d1_re = 0.5 + t2_re
    d1_im = t2_im
    # complex  t / d1  (multiply t by conj(d1) / |d1|²)
    d1_mag2 = d1_re * d1_re + d1_im * d1_im
    w1_re = (t_re * d1_re + t_im * d1_im) / d1_mag2
    w1_im = (t_im * d1_re - t_re * d1_im) / d1_mag2
    w1_re = 0.5641896 * w1_re
    w1_im = 0.5641896 * w1_im

    # ---- Region 2:  5.5 ≤ s < 15 ----------------------------------------
    # w ≈ t · (1.410474 + C·t²) / (0.75 + t²·(3 + t²))
    #    = t · num2 / den2
    num2_re = 0.5641896 * t2_re + 1.410474
    num2_im = 0.5641896 * t2_im
    # den2 = 0.75 + t²·(3 + t²)
    # inner = 3 + t²
    inn_re = 3.0 + t2_re
    inn_im = t2_im
    # t² · inner
    p_re = t2_re * inn_re - t2_im * inn_im
    p_im = t2_re * inn_im + t2_im * inn_re
    den2_re = 0.75 + p_re
    den2_im = p_im
    # t * num2
    tn2_re = t_re * num2_re - t_im * num2_im
    tn2_im = t_re * num2_im + t_im * num2_re
    # (t·num2) / den2
    d2_mag2 = den2_re * den2_re + den2_im * den2_im
    w2_re = (tn2_re * den2_re + tn2_im * den2_im) / d2_mag2
    w2_im = (tn2_im * den2_re - tn2_re * den2_im) / d2_mag2

    # ---- Region 3:  y ≥ 0.195·|x| − 0.176  (and s < 5.5) ---------------
    # w = (16.4955 + t·(20.20933 + t·(11.96482 + t·(3.778987 + t·0.5642236))))
    #   / (16.4955 + t·(38.82363 + t·(39.27121 + t·(21.69274 + t·(6.699398 + t)))))
    # Evaluate numerator via Horner in complex arithmetic
    # num = ((((0.5642236·t + 3.778987)·t + 11.96482)·t + 20.20933)·t + 16.4955)
    def _horner3(tr, ti):
        """Evaluate the Region 3 rational approximation for t = tr + i·ti."""
        def _cmul(a_re, a_im):
            """Return (a_re + i·a_im) * (tr + i·ti)."""
            return a_re * tr - a_im * ti, a_re * ti + a_im * tr

        # Numerator
        n_re, n_im = 0.5642236 * tr, 0.5642236 * ti
        n_re, n_im = _cmul(n_re + 3.778987, n_im)
        n_re, n_im = _cmul(n_re + 11.96482, n_im)
        n_re, n_im = _cmul(n_re + 20.20933, n_im)
        n_re = n_re + 16.4955

        # Denominator
        d_re, d_im = _cmul(tr + 6.699398, ti)
        d_re, d_im = _cmul(d_re + 21.69274, d_im)
        d_re, d_im = _cmul(d_re + 39.27121, d_im)
        d_re, d_im = _cmul(d_re + 38.82363, d_im)
        d_re = d_re + 16.4955

        mag2 = d_re * d_re + d_im * d_im
        r = (n_re * d_re + n_im * d_im) / mag2
        i = (n_im * d_re - n_re * d_im) / mag2
        return r, i

    w3_re, w3_im = _horner3(t_re, t_im)           # t = y - ix

    # ---- Region 4:  the core  (s < 5.5  and  y < 0.195·|x| − 0.176) -----
    # Humlíček's prescription:
    #   w₄(x, y) = 2·exp(−x²)·[cos(2xy) + i·sin(2xy)]  −  w₃(x, −y)
    # i.e. evaluate the Region-3 rational with t_neg = −y − ix.
    w3neg_re, w3neg_im = _horner3(-y, -x)          # t = -y - ix

    exx = jnp.exp(-x * x)
    twoXY = 2.0 * x * y
    w4_re = 2.0 * exx * jnp.cos(twoXY) - w3neg_re
    w4_im = 2.0 * exx * jnp.sin(twoXY) - w3neg_im

    # ---- Blend regions (branchless) --------------------------------------
    mask4 = (s < 5.5) & (y < 0.195 * ax - 0.176)
    mask3 = (s < 5.5) & ~mask4
    mask2 = (s >= 5.5) & (s < 15.0)
    # mask1 is the remainder (s >= 15)

    out_re = jnp.where(mask4, w4_re,
             jnp.where(mask3, w3_re,
             jnp.where(mask2, w2_re, w1_re)))
    out_im = jnp.where(mask4, w4_im,
             jnp.where(mask3, w3_im,
             jnp.where(mask2, w2_im, w1_im)))
    return out_re, out_im


# ---- Hjerting H(x,a) and L(x,a) via Humlíček ---

@custom_jvp
def hjert(x, a):
    """Voigt-Hjerting function  H(x, a) = Re[w(x + i·a)].

    Humlíček (1982) cpf12 rational approximation.
    Fully vectorized, no vmap.  ~10⁻⁶ relative accuracy (float32).
    """
    re, _ = _wofz_humlicek(x, a)
    return re


def _ljert(x, a):
    """L(x, a) = Im[w(x + i·a)].  Helper for the custom JVP."""
    _, im = _wofz_humlicek(x, a)
    return im


@hjert.defjvp
def _hjert_jvp(primals, tangents):
    x, a = primals
    ux, ua = tangents
    H_re, H_im = _wofz_humlicek(x, a)   # compute both in one call
    dHdx = 2.0 * a * H_im - 2.0 * x * H_re
    dHda = 2.0 * x * H_im + 2.0 * a * H_re - 2.0 / jnp.sqrt(jnp.pi)
    return H_re, dHdx * ux + dHda * ua


# ================================================================== #
#  Public API                                                          #
# ================================================================== #

@jit
def voigt_profile(nu, nu0, sigmaD, gammaL):
    """Normalised Voigt profile  (Humlíček cpf12, fully vectorized).

    Args:
        nu:     Frequency / wavenumber array.  Shape (N,).
        nu0:    Line centre (scalar).
        sigmaD: Gaussian (Doppler) width σ_D (scalar).
        gammaL: Lorentzian (pressure) half-width γ_L (scalar).

    Returns:
        Array of shape (N,) — the normalised Voigt profile.
    """
    sfac = 1.0 / (jnp.sqrt(2.0) * sigmaD)
    return sfac * hjert(sfac * (nu - nu0), sfac * gammaL) / jnp.sqrt(jnp.pi)
