"""Learned stellar template: a Gaussian-process spectrum, integrated out exactly.

The stellar spectrum is a list of node values ``β`` in the stellar rest frame,
joined by straight lines (:class:`StellarGrid`). With the telluric and instrument
parameters fixed, the forward model is affine in ``β``::

    model_n(β) = model_n(β⁰) + A_n (β − β⁰)

A Gaussian prior ``β ~ N(β̄, Q⁻¹)`` — a Matérn-3/2 field written as a banded
precision ``Q`` around an anchor ``β̄`` (:class:`StellarPrior`) — makes the
template's posterior Gaussian. Its mean ``m`` solves one banded linear system::

    (G + Q)(m − β⁰) = b + Q(β̄ − β⁰),   G = Σ_n A_nᵀ W_n A_n,   b = Σ_n A_nᵀ W_n r_n

with ``W_n`` the inverse noise variances and ``r_n`` the residuals at ``β⁰``. The
evidence, the probability of the data with the template integrated out, is::

    −2 ln Z = χ²(m) + (m − β̄)ᵀ Q (m − β̄) + ln det(G + Q) − ln det Q + Σ ln 2πσ²

and sets the prior's hyperparameters (:func:`fit_hyperparameters`).

``A_n`` is read off the forward model exactly with colored JVPs: every pixel sees
fewer template nodes than there are colors, so one JVP per color recovers every
row (:func:`normal_equations`). Everything after the one-off geometry setup is
JAX — jit-compiled, differentiable, and runs on GPU. The equations are derived in
``docs/stellar.rst``.
"""

import logging
import math
from functools import partial
from typing import List, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from enuma.config import FitConfig
from enuma.constants import C_KMS
from enuma.inference.banded import (
    block_cholesky,
    block_inverse_diag,
    block_logdet,
    block_matvec,
    block_solve,
    jacobi_scale,
)
from enuma.model.forward import (
    _exposure_inputs,
    _exposure_slice,
    centre_night_terms,
    forward_model,
    forward_model_batched,
    wave_shift_pix,
)
from enuma.model.stellar import apply_stellar, fetch_phoenix_spectrum, normalize_template
from enuma.state import Layout, ModelConfig, ModelContext, ModelParameters

__all__ = [
    "StellarGrid",
    "NightGeometry",
    "ProbeSpec",
    "StellarPrior",
    "Hyper",
    "Posterior",
    "StellarTemplate",
    "build_stellar_grid",
    "night_geometry",
    "probe_spec",
    "nodes_to_grid",
    "reference_on_nodes",
    "make_prior",
    "initial_hyper",
    "hyper_values",
    "anchor",
    "normal_equations",
    "refit_continuum",
    "posterior",
    "fit_hyperparameters",
]

logger = logging.getLogger("enuma.inference.stellar_gp")

# Pixels per chunk when accumulating G (each chunk is one small dense product).
_CHUNK_PIXELS = 32
# Default margin for the fitted wavelength-solution shift (pixels) when sizing node ranges.
_WAVE_SHIFT_MARGIN_PIX = 16
# Quadrature points for the Matérn normalisation integral.
_N_QUAD = 4096
# Bounds that keep the hyperparameters in a sane range during optimisation.
_SIGMA_BOUNDS = (1e-4, 2.0)
_ALPHA_BOUNDS = (1e-4, 10.0)
# ℓ below ~1 km/s (the thermal width of stellar lines) stops being line-like; above
# ~10 km/s the template would start to take over continuum-scale structure.
_ELL_BOUNDS_KMS = (1.0, 10.0)
_SCALE_BOUNDS = (-1.0, 3.0)


# ---------------------------------------------------------------------------
# Geometry: rest-frame node grid, node → model-grid map, pixel → node windows
# ---------------------------------------------------------------------------

class StellarGrid(NamedTuple):
    """Rest-frame template nodes, one padded row of blocks per order.

    Node ``j`` of order ``o`` sits at the global model-lattice index
    ``stride·(node_lo[o] + j)``, i.e. at rest-frame
    ``ln λ = lnlam0 + stride·(node_lo[o] + j)·dlnlam``. Every order's row is padded
    to ``n_pad = n_blocks·block`` nodes; padded nodes see no data and carry a unit
    prior, so they drop out of every result.
    """
    lnlam0: float          # ln λ (nm) of global lattice index 0
    dlnlam: float          # model-grid step in ln λ
    stride: int            # model-grid samples per node
    node_lo: np.ndarray    # (O,) global node index of each order's first node
    n_nodes: np.ndarray    # (O,) real nodes per order
    block: int             # block size = number of probe colors
    n_blocks: int          # blocks per order row

    @property
    def n_pad(self) -> int:
        return self.block * self.n_blocks

    @property
    def h_kms(self) -> float:
        """Node spacing in km/s."""
        return self.stride * self.dlnlam * C_KMS

    def wave_nm(self) -> np.ndarray:
        """``(O, n_pad)`` rest-frame node wavelengths (nm), NaN on padded nodes."""
        j = np.arange(self.n_pad)[None, :]
        g = self.stride * (self.node_lo[:, None] + j)
        wave = np.exp(self.lnlam0 + g * self.dlnlam)
        return np.where(j < self.n_nodes[:, None], wave, np.nan)

    def real(self) -> np.ndarray:
        """``(O, n_pad)`` mask of real (unpadded) nodes."""
        return np.arange(self.n_pad)[None, :] < self.n_nodes[:, None]


def _round_up(x: float, multiple: int) -> int:
    return int(multiple * math.ceil(x / multiple))


def build_stellar_grid(model_waves: Sequence[np.ndarray], obs_waves: Sequence[np.ndarray],
                       velocities: Sequence[np.ndarray], *, node_kms: float,
                       lsf_kernel_width: int,
                       wave_shift_pix: float = _WAVE_SHIFT_MARGIN_PIX) -> StellarGrid:
    """Size the rest-frame node grid for one or several nights of the same setting.

    Args:
        model_waves: per night, the float64 model wavelength grid (nm). All nights
            must be slices of the same opacity lattice (uniform in ln λ).
        obs_waves: per night, the observed ``(O, P)`` wavelengths (nm).
        velocities: per night, the ``(N,)`` stellar velocities relative to the
            observatory, systemic RV + barycentric term (km/s).
        node_kms: requested node spacing; rounded to whole model samples.
        lsf_kernel_width: LSF kernel extent in model samples (sets the colors).
        wave_shift_pix: largest wavelength-solution shift (pixels) the node ranges
            must absorb on top of the observed span.
    """
    mw0 = np.asarray(model_waves[0], np.float64)
    lnlam0 = float(np.log(mw0[0]))
    dlnlam = float((np.log(mw0[-1]) - np.log(mw0[0])) / (mw0.size - 1))
    for mw in model_waves:
        g = (np.log(np.asarray(mw, np.float64)[0]) - lnlam0) / dlnlam
        if abs(g - round(g)) > 1e-3:
            raise ValueError("The nights' model grids are not slices of one opacity lattice.")

    stride = max(1, int(round(node_kms / (dlnlam * C_KMS))))
    h_ln = stride * dlnlam

    shapes = {np.shape(w) for w in obs_waves}
    if len(shapes) != 1:
        raise NotImplementedError("All nights must share the instrument setting (same "
                                  f"order/pixel layout); got shapes {sorted(shapes)}.")

    # Widest pixel, in model samples; sets the colors and the node margins.
    pix_samples = max(float(np.max(np.diff(np.log(np.asarray(w, np.float64)), axis=1)))
                      for w in obs_waves) / dlnlam
    support = lsf_kernel_width + math.ceil(pix_samples) + 4     # samples a pixel sees
    row_nodes = math.ceil(support / stride) + 2
    block = _round_up(row_nodes + 4, 8)
    margin = block // 2 + math.ceil(wave_shift_pix * pix_samples / stride) + 2

    # Rest-frame pixel positions over every night and exposure, in node units.
    t_min, t_max = None, None
    for w, v in zip(obs_waves, velocities):
        lnw = np.log(np.asarray(w, np.float64))                    # (O, P)
        u = np.log1p(np.asarray(v, np.float64) / C_KMS)            # (N,)
        lo = (lnw.min(axis=1) - u.max() - lnlam0) / h_ln
        hi = (lnw.max(axis=1) - u.min() - lnlam0) / h_ln
        t_min = lo if t_min is None else np.minimum(t_min, lo)
        t_max = hi if t_max is None else np.maximum(t_max, hi)
    node_lo = (np.floor(t_min) - margin).astype(np.int64)
    node_hi = (np.ceil(t_max) + margin).astype(np.int64)

    # Neighbouring orders (e.g. adjacent detectors) whose margins overlap split the
    # gap at the midpoint between their pixels: no pixel sees that far.
    order = np.argsort(t_min)
    for a, b in zip(order[:-1], order[1:]):
        if node_hi[a] >= node_lo[b]:
            mid = int(np.floor(0.5 * (t_max[a] + t_min[b])))
            if mid - t_max[a] < block // 2 + 2 or t_min[b] - mid < block // 2 + 2:
                raise NotImplementedError("Orders overlapping in wavelength are not "
                                          "supported by the learned template yet.")
            node_hi[a], node_lo[b] = mid - 1, mid
    n_nodes = (node_hi - node_lo + 1).astype(np.int64)
    for mw in model_waves:
        g0 = round((np.log(np.asarray(mw, np.float64)[0]) - lnlam0) / dlnlam)
        if (np.any(stride * node_lo < g0)
                or np.any(stride * node_hi > g0 + np.asarray(mw).size - 1)):
            raise ValueError("Template nodes fall outside the model grid; the orders are "
                             "too close to the edge of the loaded opacity range.")

    n_blocks = int(math.ceil(n_nodes.max() / block))
    grid = StellarGrid(lnlam0, dlnlam, stride, node_lo, n_nodes, block, n_blocks)
    logger.info("Stellar template: %d orders × ≤%d nodes at %.3f km/s, blocks of %d.",
                n_nodes.size, int(n_nodes.max()), grid.h_kms, block)
    return grid


class NightGeometry(NamedTuple):
    """Per-night arrays (JAX) linking template nodes, the model grid and pixels."""
    phi_q: jnp.ndarray            # (M,) model-grid samples covered by template nodes
    phi_o: jnp.ndarray            # (M,) their order
    phi_j: jnp.ndarray            # (M,) left node (row index)
    phi_w: jnp.ndarray            # (M,) weight of the right node
    lam_node0: jnp.ndarray        # (O,) λ (nm) of each order's first node
    ref_minus_node0: jnp.ndarray  # (O,) order reference wavelength − lam_node0 (nm)


class ProbeSpec(NamedTuple):
    """Static sizes for :func:`normal_equations` (hashable, a jit static argument)."""
    colors: int     # probe colors = block size
    chunk: int      # pixels per chunk
    window: int     # nodes spanned by one chunk's rows
    n_pad: int      # nodes per order row
    n_model: int    # model-grid samples of the night
    h_ln: float     # node spacing in ln λ


def night_geometry(grid: StellarGrid, model_wave: np.ndarray,
                   order_start_idx: np.ndarray, order_size: int) -> NightGeometry:
    """Node → model-grid interpolation map and pixel-window anchors for one night.

    ``model_wave`` is the night's float64 model grid; ``order_start_idx`` /
    ``order_size`` its per-order chunks (from the context / config).
    """
    mw = np.asarray(model_wave, np.float64)
    g0 = int(round((np.log(mw[0]) - grid.lnlam0) / grid.dlnlam))
    qs, os_, js, ws = [], [], [], []
    for o in range(grid.node_lo.size):
        g = np.arange(grid.stride * grid.node_lo[o],
                      grid.stride * (grid.node_lo[o] + grid.n_nodes[o] - 1) + 1)
        t = (g - grid.stride * grid.node_lo[o]) / grid.stride
        j = np.minimum(np.floor(t).astype(np.int64), grid.n_nodes[o] - 2)
        qs.append(g - g0)
        os_.append(np.full(g.size, o))
        js.append(j)
        ws.append(t - j)
    starts = np.asarray(order_start_idx, np.int64)
    order_ref = 0.5 * (mw[starts] + mw[starts + int(order_size) - 1])   # as in setup
    lam_node0 = np.exp(grid.lnlam0 + grid.stride * grid.node_lo * grid.dlnlam)
    return NightGeometry(
        phi_q=jnp.asarray(np.concatenate(qs), jnp.int32),
        phi_o=jnp.asarray(np.concatenate(os_), jnp.int32),
        phi_j=jnp.asarray(np.concatenate(js), jnp.int32),
        phi_w=jnp.asarray(np.concatenate(ws), jnp.float32),
        lam_node0=jnp.asarray(lam_node0, jnp.float32),
        ref_minus_node0=jnp.asarray(order_ref - lam_node0, jnp.float32),
    )


def probe_spec(grid: StellarGrid, obs_wave: np.ndarray, n_model: int) -> ProbeSpec:
    """Static sizes for one night's :func:`normal_equations`."""
    h_ln = grid.stride * grid.dlnlam
    step = float(np.max(np.diff(np.log(np.asarray(obs_wave, np.float64)), axis=1))) / h_ln
    window = _round_up(grid.block + (_CHUNK_PIXELS - 1) * step * 1.25 + 8, 8)
    return ProbeSpec(colors=grid.block, chunk=_CHUNK_PIXELS, window=window,
                     n_pad=grid.n_pad, n_model=int(n_model), h_ln=float(h_ln))


def nodes_to_grid(beta: jnp.ndarray, geom: NightGeometry, n_model: int) -> jnp.ndarray:
    """Template on the night's model grid (rest frame) from the ``(O, n_pad)`` nodes;
    samples outside every order's node range are 1."""
    vals = ((1.0 - geom.phi_w) * beta[geom.phi_o, geom.phi_j]
            + geom.phi_w * beta[geom.phi_o, geom.phi_j + 1])
    return jnp.ones(n_model, beta.dtype).at[geom.phi_q].set(vals)


# ---------------------------------------------------------------------------
# Reference spectrum and prior
# ---------------------------------------------------------------------------

def _smooth(x: np.ndarray, sigma_nodes: float) -> np.ndarray:
    """Gaussian smoothing along the last axis (numpy, setup only)."""
    half = max(1, int(math.ceil(4 * sigma_nodes)))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma_nodes) ** 2)
    k /= k.sum()
    return np.stack([np.convolve(row, k, mode="same") for row in np.atleast_2d(x)])


def reference_on_nodes(grid: StellarGrid, fit_config: FitConfig, *,
                       vsini_kms: float = 0.0):
    """Reference spectrum ``β_ref`` and its line-depth envelope on the nodes.

    ``fit_config.stellar_reference`` selects ``"flat"`` (1 everywhere), ``"newera"``
    (the PHOENIX-NewEra template of ``stellar_teff`` / ``stellar_logg`` / …,
    normalised like the fixed-template mode and rotated by ``vsini_kms``), or a
    two-column file (rest-frame wavelength nm, normalised flux). Returns numpy
    ``(beta_ref, envelope)``, both ``(O, n_pad)``; padded nodes get 1 and 0.
    """
    fc = fit_config
    real = grid.real()
    ref = fc.stellar_reference
    if ref == "flat":
        return np.ones(real.shape), np.zeros(real.shape)

    beta_ref = np.ones(real.shape)
    if ref == "newera":
        if fc.stellar_teff is None or fc.stellar_logg is None:
            raise ValueError("stellar_reference='newera' needs stellar_teff and stellar_logg "
                             "(or use stellar_reference='flat').")
        wave_nodes = grid.wave_nm()
        pad = 0.01 * (np.nanmax(wave_nodes) - np.nanmin(wave_nodes))
        s_wave, s_flux = fetch_phoenix_spectrum(
            fc.stellar_teff, fc.stellar_logg, fc.stellar_feh, fc.stellar_alpha,
            grid=fc.stellar_grid, wave_min_nm=float(np.nanmin(wave_nodes)) - pad,
            wave_max_nm=float(np.nanmax(wave_nodes)) + pad, cache_dir=fc.cache_dir)
        s_norm = normalize_template(s_wave, s_flux)
        dv = grid.dlnlam * C_KMS
        half = int(fc.stellar_rot_half_width)
        for o in range(grid.node_lo.size):
            # Fine lattice around the order's nodes, rotated like the fixed template.
            g = np.arange(grid.stride * grid.node_lo[o] - half - 2,
                          grid.stride * (grid.node_lo[o] + grid.n_nodes[o]) + half + 3)
            fine = np.interp(np.exp(grid.lnlam0 + g * grid.dlnlam), s_wave, s_norm,
                             left=1.0, right=1.0)
            fine = np.asarray(apply_stellar(jnp.asarray(fine, jnp.float32), dv, 0.0,
                                            max(float(vsini_kms), 1e-3), half_width=half,
                                            epsilon=fc.stellar_epsilon))
            idx = grid.stride * (grid.node_lo[o] + np.arange(grid.n_nodes[o])) - g[0]
            beta_ref[o, :grid.n_nodes[o]] = fine[idx]
    else:
        table = np.loadtxt(ref, comments="#", dtype=np.float64)
        wave_nodes = grid.wave_nm()
        for o in range(grid.node_lo.size):
            n = grid.n_nodes[o]
            beta_ref[o, :n] = np.interp(wave_nodes[o, :n], table[:, 0], table[:, 1],
                                        left=1.0, right=1.0)

    sigma_nodes = fc.stellar_gp_length_kms / grid.h_kms
    envelope = np.where(real, _smooth(np.abs(1.0 - beta_ref) * real, sigma_nodes), 0.0)
    return np.where(real, beta_ref, 1.0), envelope


class StellarPrior(NamedTuple):
    """Fixed inputs of the template prior (JAX arrays)."""
    beta_ref: jnp.ndarray   # (O, n_pad) reference spectrum (1 on padded nodes)
    envelope: jnp.ndarray   # (O, n_pad) reference line-depth envelope (0 on padded nodes)
    real: jnp.ndarray       # (O, n_pad) real (unpadded) nodes
    h_kms: jnp.ndarray      # () node spacing (km/s)


class Hyper(NamedTuple):
    """Unconstrained hyperparameters of the template prior (an optax pytree)."""
    log_sigma: jnp.ndarray  # (O,) log amplitude per order
    scale: jnp.ndarray      # (O,) reference scale s_o per order
    log_alpha: jnp.ndarray  # (O,) log weight of the reference envelope per order
    log_ell: jnp.ndarray    # (O,) log correlation length per order (km/s)


def make_prior(grid: StellarGrid, beta_ref: np.ndarray, envelope: np.ndarray) -> StellarPrior:
    """:class:`StellarPrior` from the output of :func:`reference_on_nodes`."""
    return StellarPrior(jnp.asarray(beta_ref, jnp.float32), jnp.asarray(envelope, jnp.float32),
                        jnp.asarray(grid.real()), jnp.asarray(grid.h_kms, jnp.float32))


def initial_hyper(n_orders: int, fit_config: FitConfig) -> Hyper:
    """Starting hyperparameters from the ``stellar_*`` knobs: the reference at full
    depth (``s_o = 1``; irrelevant for a flat reference) and envelope weight 1."""
    return Hyper(
        log_sigma=jnp.full(n_orders, math.log(fit_config.stellar_gp_sigma), jnp.float32),
        scale=jnp.full(n_orders, 0.0 if fit_config.stellar_reference == "flat" else 1.0,
                       jnp.float32),
        log_alpha=jnp.zeros(n_orders, jnp.float32),
        log_ell=jnp.full(n_orders, math.log(fit_config.stellar_gp_length_kms), jnp.float32),
    )


def hyper_values(hyper: Hyper) -> dict:
    """Physical (bounded) values, each ``(O,)``: ``sigma``, ``scale``, ``alpha``, ``ell_kms``."""
    clip_log = lambda v, b: jnp.exp(jnp.clip(v, math.log(b[0]), math.log(b[1])))
    return dict(sigma=clip_log(hyper.log_sigma, _SIGMA_BOUNDS),
                scale=jnp.clip(hyper.scale, *_SCALE_BOUNDS),
                alpha=clip_log(hyper.log_alpha, _ALPHA_BOUNDS),
                ell_kms=clip_log(hyper.log_ell, _ELL_BOUNDS_KMS))


def anchor(prior: StellarPrior, hyper: Hyper) -> jnp.ndarray:
    """Prior anchor ``β̄ = 1 + s_o (β_ref − 1)``, ``(O, n_pad)``."""
    s = hyper_values(hyper)["scale"]
    return 1.0 + s[:, None] * (prior.beta_ref - 1.0)


def _amplitude(prior: StellarPrior, hv: dict) -> jnp.ndarray:
    """Prior standard deviation per node, ``d² = σ_o² + α² ϱ²`` (1 on padded nodes)."""
    d = jnp.sqrt(hv["sigma"][:, None] ** 2 + (hv["alpha"][:, None] * prior.envelope) ** 2)
    return jnp.where(prior.real, d, 1.0)


def _matern_tau2(kappa: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
    """Normalisation ``τ²`` of the lattice Matérn-3/2 precision
    ``Q = τ² h (κ² − Δ_h)²`` that gives unit marginal variance."""
    theta = (jnp.arange(_N_QUAD) + 0.5) * (2.0 * jnp.pi / _N_QUAD) - jnp.pi
    sym = kappa ** 2 + (4.0 / h ** 2) * jnp.sin(theta / 2.0) ** 2
    return jnp.sum(1.0 / sym ** 2) / (h * _N_QUAD)


def _matern_band(real: jnp.ndarray, h: jnp.ndarray, ell: jnp.ndarray, n_colors: int):
    """Unit-variance Matérn-3/2 precision of one order row in upper-band storage
    ``(n_pad, n_colors)``: ``Q_ℓ = τ² h LᵀL`` with ``L = κ² − Δ_h`` truncated at the
    row's real nodes; padded nodes get an identity row."""
    kappa = jnp.sqrt(3.0) / ell
    c = _matern_tau2(kappa, h) * h
    a = kappa ** 2 + 2.0 / h ** 2
    r = real.astype(jnp.float32)
    r1 = jnp.concatenate([r[1:], jnp.zeros(1)])           # node j + 1 real
    r2 = jnp.concatenate([r[2:], jnp.zeros(2)])           # node j + 2 real
    rm1 = jnp.concatenate([jnp.zeros(1), r[:-1]])         # node j − 1 real
    main = jnp.where(real, c * (a ** 2 + (r1 + rm1) / h ** 4), 1.0)
    off1 = r * r1 * c * (-2.0 * a / h ** 2)
    off2 = r * r2 * c / h ** 4
    band = jnp.zeros((real.shape[0], n_colors))
    return band.at[:, 0].set(main).at[:, 1].set(off1).at[:, 2].set(off2)


def _band_to_blocks(band: jnp.ndarray):
    """Upper-band storage ``(n_pad, B)`` (``band[i, d] = M[i, i + d]``) → the
    block-tridiagonal ``(diag (K, B, B), sub (K − 1, B, B))`` of :mod:`banded`."""
    n_pad, B = band.shape
    K = n_pad // B
    i = jnp.arange(B)[:, None]
    j = jnp.arange(B)[None, :]
    k = jnp.arange(K)[:, None, None]
    diag = band[k * B + jnp.minimum(i, j), jnp.abs(i - j)]
    # M[(k+1)B + i, kB + j] = band[kB + j, B + i − j], nonzero only when i < j.
    ks = jnp.arange(K - 1)[:, None, None]
    sub = jnp.where(i < j, band[ks * B + j, jnp.minimum(B + i - j, B - 1)], 0.0)
    return diag, sub


# ---------------------------------------------------------------------------
# Normal equations: A_n read off the forward model with colored JVPs
# ---------------------------------------------------------------------------

def _window_start(params_n: ModelParameters, ctx_n: ModelContext, config: ModelConfig,
                  geom: NightGeometry, spec: ProbeSpec) -> jnp.ndarray:
    """``(O, P)`` first node of each pixel's probe window: the node nearest the
    pixel's rest-frame centre minus half the colors."""
    shift = wave_shift_pix(params_n, ctx_n, config) * ctx_n.obs_d_wave
    lam_off = ctx_n.obs_wave_off + shift + geom.ref_minus_node0[:, None]
    x = (jnp.log1p(lam_off / geom.lam_node0[:, None])
         - jnp.log1p((params_n.rv_systemic + ctx_n.baryrv) / C_KMS))
    return jnp.round(x / spec.h_ln).astype(jnp.int32) - spec.colors // 2


@partial(jax.jit, static_argnames=("config", "layout", "spec"))
def normal_equations(params: ModelParameters, ctx: ModelContext, config: ModelConfig,
                     layout: Layout, obs_flux: jnp.ndarray, obs_w: jnp.ndarray,
                     beta0: jnp.ndarray, geom: NightGeometry, spec: ProbeSpec):
    """Accumulate one night's normal equations at the expansion point ``beta0``.

    For every exposure, ``jax.linearize`` the forward model in the template nodes
    and push one tangent per color through it: color ``c`` switches on every node
    whose row index is ``c`` modulo the color count, and each pixel sees at most
    one of them, so the JVPs hold every row of ``A_n`` exactly. Pixels are then
    grouped in chunks whose rows span a short node window, and each chunk adds its
    dense ``Aᵀ W A`` into banded storage.

    Args:
        obs_flux, obs_w: ``(N, O, P)`` data and inverse-variance weights (0 = masked).
        beta0: ``(O, n_pad)`` expansion point (the current template).

    Returns:
        ``(G_band (O, n_pad, colors), b (O, n_pad), chi2, overflow)`` with
        ``G_band[o, i, d] = G[i, i + d]``, ``b = Σ Aᵀ W r`` and ``chi2 = Σ rᵀ W r``
        at ``beta0``. ``overflow`` flags a pixel window outside its chunk or row,
        which would invalidate the geometry (callers raise on it).
    """
    O, P = obs_flux.shape[1:]
    Pc, C, Wn = spec.colors, spec.chunk, spec.window
    n_chunk = -(-P // C)
    pad = n_chunk * C - P
    u = jnp.arange(Pc)
    tangents = jnp.broadcast_to(
        (jnp.arange(spec.n_pad)[None, :] % Pc == u[:, None]).astype(beta0.dtype)[:, None, :],
        (Pc, O, spec.n_pad))
    a_idx = jnp.arange(Wn)[:, None]
    d_idx = jnp.arange(Pc)[None, :]
    in_window = (a_idx + d_idx) < Wn                               # (Wn, Pc)
    take_b = jnp.minimum(a_idx + d_idx, Wn - 1)
    oi4 = jnp.arange(O)[:, None, None, None]
    ci4 = jnp.arange(n_chunk)[None, :, None, None]
    pi4 = jnp.arange(C)[None, None, :, None]

    mapped = _exposure_inputs(params, ctx, config, layout)
    mapped["obs"] = obs_flux
    mapped["w"] = obs_w

    def body(carry, ex):
        band, b, chi2, overflow = carry
        params_n, ctx_n = _exposure_slice(params, ctx, config, ex)

        def model_of(beta):
            template = nodes_to_grid(beta, geom, spec.n_model)
            return forward_model(params_n, ctx_n._replace(stellar_template=template), config)

        m0, f_jvp = jax.linearize(model_of, beta0)
        Y = jax.vmap(f_jvp)(tangents)                               # (Pc, O, P)
        w = ex["w"]
        r0 = jnp.where(w > 0, ex["obs"] - m0, 0.0)
        chi2 = chi2 + jnp.sum(w * r0 * r0)

        s = _window_start(params_n, ctx_n, config, geom, spec)      # (O, P)
        R = jnp.take_along_axis(Y, (s[None] + u[:, None, None]) % Pc, axis=0)
        R = jnp.moveaxis(R, 0, -1)                                  # (O, P, Pc): A[o, p, s + u]

        R = jnp.pad(R, ((0, 0), (0, pad), (0, 0))).reshape(O, n_chunk, C, Pc)
        wc = jnp.pad(w, ((0, 0), (0, pad))).reshape(O, n_chunk, C)
        rc = jnp.pad(r0, ((0, 0), (0, pad))).reshape(O, n_chunk, C)
        sc = jnp.pad(s, ((0, 0), (0, pad)), mode="edge").reshape(O, n_chunk, C)
        s0 = sc[..., 0]                                              # (O, n_chunk)
        off = sc - s0[..., None]
        overflow = (overflow | jnp.any(off < 0) | jnp.any(off > Wn - Pc)
                    | jnp.any(s0 < 0) | jnp.any(s0 + Wn > spec.n_pad))

        D = jnp.zeros((O, n_chunk, C, Wn), R.dtype).at[
            oi4, ci4, pi4, jnp.clip(off[..., None] + u, 0, Wn - 1)].set(R)
        Gc = jnp.einsum("ocpa,ocp,ocpb->ocab", D, wc, D)              # (O, n_chunk, Wn, Wn)
        bc = jnp.einsum("ocpa,ocp->oca", D, wc * rc)                  # (O, n_chunk, Wn)

        vals = jnp.take_along_axis(Gc, jnp.broadcast_to(take_b, Gc.shape[:2] + take_b.shape),
                                   axis=3)
        vals = jnp.where(in_window, vals, 0.0)                       # Gc[a, a + d]
        rows = s0[..., None, None] + a_idx                           # (O, n_chunk, Wn, 1)
        band = band.at[oi4, rows, d_idx].add(vals, mode="drop")
        b = b.at[oi4[..., 0], s0[..., None] + jnp.arange(Wn)].add(bc, mode="drop")
        return (band, b, chi2, overflow), None

    init = (jnp.zeros((O, spec.n_pad, Pc), beta0.dtype), jnp.zeros((O, spec.n_pad), beta0.dtype),
            jnp.zeros((), beta0.dtype), jnp.array(False))
    (band, b, chi2, overflow), _ = jax.lax.scan(body, init, mapped)
    return band, b, chi2, overflow


@partial(jax.jit, static_argnames=("config", "layout", "n_iter"))
def refit_continuum(params: ModelParameters, ctx: ModelContext, config: ModelConfig,
                    layout: Layout, obs_flux: jnp.ndarray, obs_w: jnp.ndarray,
                    prior_sigma: float = 0.1, n_iter: int = 2):
    """MAP continuum (static shape and per-exposure scales) with everything else fixed.

    The model is ``s_{n,o} · (B·w_o) · g`` with ``g`` the model at unit continuum. Given
    the scales, each order's shape ``w_o`` is a small linear least-squares problem
    over all exposures, with the fit's ``N(1, prior_sigma)`` prior on ``w``. Given the
    shape, each scale has a closed form; the scales are then re-centred (zero mean
    ln s per order), their mean moving into ``w``. The two solves alternate
    ``n_iter`` times. Run before each star step, this hands smooth mismatch to the
    continuum instead of the template.

    Returns:
        ``(continuum_coeffs (O, n_weights), continuum_log_scale)``; the log scales
        are returned unchanged when they are not fitted per exposure.
    """
    unit = params._replace(continuum_coeffs=jnp.ones_like(params.continuum_coeffs),
                           continuum_log_scale=jnp.zeros_like(params.continuum_log_scale))
    g = forward_model_batched(unit, ctx, config, layout)             # (N, O, P)
    Bm = ctx.cont_bspline_matrix                                     # (P, k)
    lam = 1.0 / prior_sigma ** 2
    wg2, wgf = obs_w * g * g, obs_w * g * obs_flux
    fit_scale = "continuum_log_scale" in layout.per_exposure
    O = g.shape[1]
    ls0 = (centre_night_terms(params, layout).continuum_log_scale if fit_scale
           else jnp.zeros((g.shape[0], O), g.dtype))

    def step(_, carry):
        w, ls = carry
        s = jnp.exp(jnp.broadcast_to(ls, (g.shape[0], O)))           # (N, O)
        M = (jnp.einsum("pk,no,nop,pl->okl", Bm, s * s, wg2, Bm)
             + lam * jnp.eye(Bm.shape[1]))
        rhs = jnp.einsum("pk,no,nop->ok", Bm, s, wgf) + lam
        w = jnp.linalg.solve(M, rhs[..., None])[..., 0]              # (O, k)
        if not fit_scale:
            return w, ls
        c = jnp.einsum("pk,ok->op", Bm, w)                           # (O, P)
        den = jnp.einsum("nop,op->no", wg2, c * c)
        num = jnp.einsum("nop,op->no", wgf, c)
        ok = (den > 0) & (num > 0)
        log_s = jnp.where(ok, jnp.log(jnp.where(ok, num / jnp.where(ok, den, 1.0), 1.0)), 0.0)
        mean = jnp.mean(log_s, axis=0)                               # (O,)
        return w * jnp.exp(mean)[:, None], log_s - mean

    w, ls = jax.lax.fori_loop(0, n_iter, step, (params.continuum_coeffs, ls0))
    return w, (ls if fit_scale else params.continuum_log_scale)


# ---------------------------------------------------------------------------
# Posterior, evidence, hyperparameters
# ---------------------------------------------------------------------------

class Posterior(NamedTuple):
    """Template posterior at fixed tellurics (JAX arrays)."""
    mean: jnp.ndarray            # (O, n_pad) posterior mean m
    sigma: jnp.ndarray           # (O, n_pad) posterior std (NaN when not computed)
    neg2_lnz: jnp.ndarray        # () −2 ln Z, without the data-only Σ ln 2πσ²
    neg2_lnz_order: jnp.ndarray  # (O,) per-order −2 ln Z without the χ² at β⁰
    chi2: jnp.ndarray            # () χ² at m
    penalty: jnp.ndarray         # () (m − β̄)ᵀ Q (m − β̄)
    logdet_post: jnp.ndarray     # () ln det(G + Q)
    logdet_prior: jnp.ndarray    # () ln det Q


def _order_terms(G_band, b, beta0, beta_ref, envelope, real, h, sigma, scale, alpha, ell,
                 with_var: bool):
    """Posterior mean and evidence terms of one order row (vmapped over orders)."""
    n_pad, B = G_band.shape
    K = n_pad // B
    anc = 1.0 + scale * (beta_ref - 1.0)
    d = jnp.where(real, jnp.sqrt(sigma ** 2 + (alpha * envelope) ** 2), 1.0)

    # Q = D⁻¹ Q_ℓ D⁻¹, band by band.
    q_ell = _matern_band(real, h, ell, B)
    d1 = jnp.concatenate([d[1:], jnp.ones(1)])
    d2 = jnp.concatenate([d[2:], jnp.ones(2)])
    q_band = q_ell.at[:, 0].divide(d * d).at[:, 1].divide(d * d1).at[:, 2].divide(d * d2)
    Qd, Qs = _band_to_blocks(q_band)
    Gd, Gs = _band_to_blocks(G_band)

    # ln det Q = ln det Q_ℓ − 2 Σ ln d (Q_ℓ factored after Jacobi scaling).
    Ed, Es, es = jacobi_scale(*_band_to_blocks(q_ell))
    logdet_prior = (block_logdet(block_cholesky(Ed, Es)[0]) - 2.0 * jnp.sum(jnp.log(es))
                    - 2.0 * jnp.sum(jnp.log(d)))

    Pd, Ps, ps = jacobi_scale(Gd + Qd, Gs + Qs)
    Ld, Ls = block_cholesky(Pd, Ps)
    blk = lambda x: x.reshape(K, B)
    rhs = blk(b) + block_matvec(Qd, Qs, blk(anc - beta0))
    delta = ps * block_solve(Ld, Ls, ps * rhs)                      # (K, B)
    chi2_shift = -2.0 * jnp.sum(delta * blk(b)) + jnp.sum(delta * block_matvec(Gd, Gs, delta))
    mean = beta0 + delta.reshape(-1)
    dm = blk(mean - anc)
    penalty = jnp.sum(dm * block_matvec(Qd, Qs, dm))
    logdet_post = block_logdet(Ld) - 2.0 * jnp.sum(jnp.log(ps))
    if with_var:
        var = (ps ** 2 * block_inverse_diag(Ld, Ls)).reshape(-1)
        sigma_post = jnp.where(real, jnp.sqrt(var), jnp.nan)
    else:
        sigma_post = jnp.full(n_pad, jnp.nan)
    return mean, sigma_post, chi2_shift, penalty, logdet_post, logdet_prior


def _all_orders(G_band, b, beta0, prior: StellarPrior, hyper: Hyper, with_var: bool):
    hv = hyper_values(hyper)
    terms = partial(_order_terms, with_var=with_var)
    return jax.vmap(terms, in_axes=(0, 0, 0, 0, 0, 0, None, 0, 0, 0, 0))(
        G_band, b, beta0, prior.beta_ref, prior.envelope, prior.real, prior.h_kms,
        hv["sigma"], hv["scale"], hv["alpha"], hv["ell_kms"])


@partial(jax.jit, static_argnames=("with_variance",))
def posterior(G_band: jnp.ndarray, b: jnp.ndarray, chi2_0: jnp.ndarray, beta0: jnp.ndarray,
              prior: StellarPrior, hyper: Hyper, with_variance: bool = False) -> Posterior:
    """Template posterior and evidence from the accumulated normal equations.

    ``G_band``, ``b`` and ``chi2_0`` come from :func:`normal_equations` (summed over
    nights) at the expansion point ``beta0``.
    """
    mean, sig, chi2_shift, pen, ld_post, ld_prior = _all_orders(
        G_band, b, beta0, prior, hyper, with_variance)
    per_order = chi2_shift + pen + ld_post - ld_prior
    return Posterior(mean=mean, sigma=sig, neg2_lnz=chi2_0 + jnp.sum(per_order),
                     neg2_lnz_order=per_order, chi2=chi2_0 + jnp.sum(chi2_shift),
                     penalty=jnp.sum(pen), logdet_post=jnp.sum(ld_post),
                     logdet_prior=jnp.sum(ld_prior))


@partial(jax.jit, static_argnames=("n_iter",))
def fit_hyperparameters(G_band: jnp.ndarray, b: jnp.ndarray, beta0: jnp.ndarray,
                        prior: StellarPrior, hyper0: Hyper, n_iter: int = 20):
    """Maximise the evidence over the prior's hyperparameters with L-BFGS.

    The objective is −2 ln Z minus its data-only constant, so it never re-runs the
    forward model. A backtracking line search keeps each iteration to a few
    objective values and one gradient. Returns ``(hyper, objective trace (n_iter,))``.
    """
    def loss(h):
        _, _, chi2_shift, pen, ld_post, ld_prior = _all_orders(G_band, b, beta0, prior, h, False)
        return jnp.sum(chi2_shift + pen + ld_post - ld_prior)

    # atol absorbs float32 rounding of the ~10⁵–10⁶ objective near the optimum.
    solver = optax.lbfgs(linesearch=optax.scale_by_backtracking_linesearch(
        max_backtracking_steps=15, atol=0.1, store_grad=True))
    value_and_grad = optax.value_and_grad_from_state(loss)

    def step(carry, _):
        h, state = carry
        value, grad = value_and_grad(h, state=state)
        updates, state = solver.update(grad, state, h, value=value, grad=grad, value_fn=loss)
        return (optax.apply_updates(h, updates), state), value

    (hyper, _), trace = jax.lax.scan(step, (hyper0, solver.init(hyper0)), None, length=n_iter)
    return hyper, trace


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class StellarTemplate(NamedTuple):
    """The learned stellar spectrum on its rest-frame nodes (one row per order).

    Padded nodes carry NaN wavelengths; mask them with ``np.isfinite(wave)``.
    """
    wave: np.ndarray           # (O, n_pad) rest-frame wavelength (nm)
    flux: np.ndarray           # (O, n_pad) posterior mean
    sigma: np.ndarray          # (O, n_pad) posterior standard deviation
    anchor: np.ndarray         # (O, n_pad) prior anchor β̄
    prior_sigma: np.ndarray    # (O, n_pad) prior standard deviation
    reference: np.ndarray      # (O, n_pad) reference spectrum
    hyper: dict                # sigma, scale, alpha, ell_kms, each (O,)
    neg2_lnz: float            # −2 ln Z, without the data-only Σ ln 2πσ²
    neg2_lnz_order: np.ndarray  # (O,) per-order part (without χ² at the expansion point)
    rv_kms: float              # systemic RV defining the rest frame

    def save(self, path: str) -> None:
        """Write every field to an ``.npz`` file."""
        np.savez(path, wave=self.wave, flux=self.flux, sigma=self.sigma, anchor=self.anchor,
                 prior_sigma=self.prior_sigma, reference=self.reference,
                 neg2_lnz=self.neg2_lnz, neg2_lnz_order=self.neg2_lnz_order,
                 rv_kms=self.rv_kms, **{f"hyper_{k}": np.asarray(v) for k, v in self.hyper.items()})


def stellar_template(grid: StellarGrid, prior: StellarPrior, hyper: Hyper, post: Posterior,
                     rv_kms: float) -> StellarTemplate:
    """Numpy :class:`StellarTemplate` from a posterior."""
    real = grid.real()
    hv = {k: np.asarray(v) for k, v in hyper_values(hyper).items()}
    prior_sigma = np.asarray(_amplitude(prior, hyper_values(hyper)))
    nan = lambda x: np.where(real, np.asarray(x), np.nan)
    return StellarTemplate(
        wave=grid.wave_nm(), flux=nan(post.mean), sigma=nan(post.sigma),
        anchor=nan(anchor(prior, hyper)), prior_sigma=nan(prior_sigma),
        reference=nan(prior.beta_ref), hyper=hv, neg2_lnz=float(post.neg2_lnz),
        neg2_lnz_order=np.asarray(post.neg2_lnz_order), rv_kms=float(rv_kms))
