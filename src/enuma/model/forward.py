from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from enuma.constants import G_EARTH, LSF_FWHM_TO_SIGMA, MMW_DRY_AIR, MMW_H2O
from enuma.model.stellar import apply_stellar
from enuma.model.voigt import voigt_profile
from enuma.state import Layout, ModelConfig, ModelContext, ModelParameters

__all__ = [
    "LSF_PROFILE_CHOICES",
    "OPACITY_GRID_R",
    "apply_variable_lsf",
    "apply_custom_lsf",
    "expand_tp_profiles",
    "net_stellar_rv",
    "get_layer_properties",
    "forward_model",
    "telluric_only",
    "forward_model_batched",
]

# ---------------------------------------------------------------------------
# Instrument LSF: variable-width chunked convolution (Gaussian / Voigt kernel),
# or a fixed user-supplied empirical kernel ("custom"; see ``apply_custom_lsf``).
# ---------------------------------------------------------------------------
LSF_PROFILE_CHOICES = ("gaussian", "voigt", "custom")
OPACITY_GRID_R = 1e6

@partial(jax.jit, static_argnums=(2, 3, 4),
         static_argnames=("n_chunks", "kernel_width", "profile"))
def apply_variable_lsf(model_flux: jnp.ndarray,
                       sigma_pix: jnp.ndarray,
                       n_chunks: int = 20,
                       kernel_width: int = 101,
                       profile: str = "gaussian",
                       voigt_gamma_ratio: float = 0.1) -> jnp.ndarray:
    """Variable-width instrument LSF via chunk-wise constant-kernel convolution.

    Splits the input into `n_chunks` contiguous segments and convolves each
    with a constant-width kernel sampled at the chunk centre. 

    Args:
        model_flux: (n,) flux on the high-res model grid.
        sigma_pix:  (n,) Gaussian sigma in pixel units at each sample. For
            Voigt, this is the Gaussian component of the Voigt profile.
        n_chunks: number of constant-width chunks per order (static). Increase for
            steeper width gradients; 20 is a sensible default for ≤10 % width
            variation across the input.
        kernel_width: odd integer, total kernel extent in pixels.
        profile: "gaussian" or "voigt" (static).
        voigt_gamma_ratio: Lorentzian HWHM / Gaussian sigma for the Voigt
            profile. Ignored for Gaussian.

    Returns:
        Convolved flux, same shape as `model_flux`.
    """
    if profile not in LSF_PROFILE_CHOICES:
        raise ValueError(f"Unknown LSF profile '{profile}'. Choices: {LSF_PROFILE_CHOICES}")

    n = model_flux.shape[0]
    chunk = (n + n_chunks - 1) // n_chunks       # ceil — n is static under JIT
    pad_extra = chunk * n_chunks - n             # tail padding to align chunks
    pad_k = kernel_width // 2

    # Pad flux on both sides for kernel reach + tail padding for chunk alignment.
    flux_p = jnp.pad(model_flux, (pad_k, pad_k + pad_extra), mode='edge')
    sigma_p = jnp.pad(sigma_pix, (0, pad_extra), mode='edge')

    starts = jnp.arange(n_chunks) * chunk        # chunk start indices in flux_p
    chunk_sigmas = sigma_p[starts + chunk // 2]  # σ at each chunk's centre

    # Build all n_chunks kernels in one broadcasted vectorised call.
    x = jnp.arange(kernel_width) - pad_k
    safe_s = jnp.maximum(chunk_sigmas[:, None], 1e-5)
    if profile == "gaussian":
        kernels = jnp.exp(-0.5 * (x[None, :] / safe_s) ** 2)
    elif profile == "voigt":
        gamma = jnp.maximum(voigt_gamma_ratio * safe_s, 1e-8)
        kernels = voigt_profile(x[None, :], 0.0, safe_s, gamma)
    kernels = kernels / jnp.sum(kernels, axis=1, keepdims=True)

    def conv(start, kernel):
        cp = jax.lax.dynamic_slice(flux_p, (start,), (chunk + 2 * pad_k,))
        return jnp.convolve(cp, kernel, mode='valid')   # output size = chunk

    out = jax.vmap(conv)(starts, kernels)        # (n_chunks, chunk)
    return out.flatten()[:n]


@jax.jit
def apply_custom_lsf(model_flux: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    """Convolve flux with a fixed, user-supplied empirical LSF kernel.

    Unlike :func:`apply_variable_lsf`, the kernel is constant across the order — an
    empirical LSF has a measured width and shape, so it is applied as-is rather than
    re-scaled by a fitted resolution. The kernel must already be normalised and
    sampled on the model grid (odd length, centred at zero offset); this is done once
    at setup (``enuma.model.setup``) so the per-call cost is a single convolution.

    Args:
        model_flux: (n,) flux on the high-res model grid.
        kernel: (k,) normalised kernel on the model grid; ``k`` odd, centred.

    Returns:
        Convolved flux, same shape as ``model_flux``.
    """
    pad = kernel.shape[0] // 2
    flux_p = jnp.pad(model_flux, (pad, pad), mode='edge')
    return jnp.convolve(flux_p, kernel, mode='valid')


def expand_tp_profiles(ctx: ModelContext, params: ModelParameters
                       ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Expand the non-centered GP latents into per-layer T and H2O corrections.
    """
    t_dev = ctx.temp_chol_L @ params.t_latent
    t_centers = ctx.t_ref_centers * (1.0 + t_dev)
    h2o_dex = ctx.h2o_chol_L @ params.h2o_latent
    return t_dev, t_centers, h2o_dex


def net_stellar_rv(ctx: ModelContext, params: ModelParameters) -> jnp.ndarray:
    return params.rv_systemic + ctx.baryrv


def get_layer_properties(ctx: ModelContext, params: ModelParameters
                         ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Per-layer (pressure, temperature, air column, gas mixing ratios).

    The height-pressure relation is fixed to the reference standard atmosphere
    (no per-call hydrostatic re-integration); the moist mean molecular weight
    uses the perturbed H2O VMR so the air column reflects the water burden.
    """
    _, t_centers, h2o_dex_layers = expand_tp_profiles(ctx, params)

    gas_mixing = {}
    for s, vmr in ctx.vmr_ref_centers.items():
        dex = h2o_dex_layers if s == "h2o" else params.dry_vmr_scalars.get(s, jnp.array([0.0]))
        gas_mixing[s] = vmr * (10.0 ** dex)

    vmr_h2o = gas_mixing.get("h2o", jnp.zeros_like(t_centers))
    m_ratio = 1.0 - vmr_h2o * (1.0 - MMW_H2O / MMW_DRY_AIR)
    dry_column = jnp.abs(jnp.diff(ctx.p_ref_boundaries)) / (MMW_DRY_AIR * G_EARTH)
    total_air_column = dry_column / m_ratio

    return ctx.p_ref_centers, t_centers, total_air_column, gas_mixing


def _optical_depth(ctx: ModelContext, params: ModelParameters,
                   config: ModelConfig) -> jnp.ndarray:
    """Telluric transmission on the model grid: bilinear (logP, T) interpolation
    of each species' opacity into the optical depth, then ``exp(-tau·airmass)``.

    When ``config.wind_enabled`` the per-layer GDAS line-of-sight wind Doppler-
    shifts each layer's opacity before the column is summed, applied to first
    order as ``tau ≈ A − (1/Δv)·dB/ds`` (see ``enuma.model.profile`` /
    ``setup.wind_los_kms`` for the wind profile and projection).
    """
    _, t_centers, n_col_air, gas_mixing = get_layer_properties(ctx, params)
    p_centers = ctx.p_ref_centers

    first_species = next(iter(ctx.opacity_grids))
    grid0, grid_info = ctx.opacity_grids[first_species]
    n_P, n_T = grid0.shape[:2]
    n_PT = n_P * n_T

    coord_P = jnp.clip((jnp.log10(p_centers) - grid_info.logP_min) / grid_info.logP_step, 0.0, n_P - 1.0)
    coord_T = jnp.clip((t_centers - grid_info.T_min) / grid_info.T_step, 0.0, n_T - 1.0)
    idx_P = jnp.minimum(jnp.floor(coord_P).astype(jnp.int32), n_P - 2)
    idx_T = jnp.minimum(jnp.floor(coord_T).astype(jnp.int32), n_T - 2)
    frac_P, frac_T = coord_P - idx_P, coord_T - idx_T

    corner_idx = jnp.array([
        idx_P * n_T + idx_T, idx_P * n_T + idx_T + 1,
        (idx_P + 1) * n_T + idx_T, (idx_P + 1) * n_T + idx_T + 1,
    ])
    corner_weights = jnp.array([
        (1 - frac_P) * (1 - frac_T), (1 - frac_P) * frac_T,
        frac_P * (1 - frac_T), frac_P * frac_T,
    ])  # (4, n_layers)

    total_tau = jnp.zeros_like(ctx.model_wave)
    total_tau_wind = jnp.zeros_like(ctx.model_wave)            # velocity-weighted column (B)
    for species, (grid_values, _) in ctx.opacity_grids.items():
        col_density = gas_mixing[species] * n_col_air                       # (n_layers,)
        grid_flat = grid_values.reshape(n_PT, -1)
        weighted_col = jnp.zeros(n_PT).at[corner_idx].add(corner_weights * col_density)
        total_tau += weighted_col @ grid_flat
        if config.wind_enabled:
            weighted_col_wind = jnp.zeros(n_PT).at[corner_idx].add(
                corner_weights * (col_density * ctx.wind_v_los))
            total_tau_wind += weighted_col_wind @ grid_flat

    if config.wind_enabled:
        # First-order per-layer wind Doppler shift on the velocity-uniform grid:
        # tau(s) ≈ A(s) − (1/Δv)·dB/ds, with B the velocity-weighted optical depth.
        total_tau = total_tau - jnp.gradient(total_tau_wind) / ctx.model_dv_kms

    return jnp.exp(-total_tau * ctx.airmass)


def _process_order(wave_coeffs, res_coeffs, order_dwave, pixel_dwave,
                    start_idx, obs_wave_norm, m_off, obs_off, order_ref,
                    *, y_trans, x_pix, config, lsf_kernel):
    """Resample one order's slice of the model transmission onto the observed grid.

    LSF-convolution, wave-solution shift, and flux-conserving resample. The continuum
    is applied separately (see ``forward_model``). ``forward_model`` vmaps this over
    orders.
    """
    m_flux = jax.lax.dynamic_slice(y_trans, (start_idx,), (config.order_size,))
    m_wave_abs = m_off + order_ref

    if config.lsf_profile == "custom":
        # Fixed empirical kernel (shared across orders); the fitted R(λ) is unused.
        conv_flux = apply_custom_lsf(m_flux, lsf_kernel)
    else:
        R = 1e5 * jnp.exp(jnp.polyval(res_coeffs, x_pix))
        sigma_pix = (
            m_wave_abs / (LSF_FWHM_TO_SIGMA * order_dwave) / R
            # * jnp.sqrt(1.0 / R ** 2 - 1.0 / OPACITY_GRID_R ** 2)
        )
        conv_flux = apply_variable_lsf(
            m_flux, sigma_pix,
            n_chunks=config.lsf_n_chunks,
            kernel_width=config.lsf_kernel_width,
            profile=config.lsf_profile,
            voigt_gamma_ratio=config.lsf_voigt_gamma_ratio,
        )

    # Wave solution: shift the observed grid (in offset space) by the pixel polynomial.
    obs_off_shifted = obs_off + jnp.polyval(wave_coeffs, obs_wave_norm) * pixel_dwave

    # Resample onto the observed grid conserving cumulative flux.
    edges = jnp.concatenate([
        obs_off_shifted[:1] - 0.5 * (obs_off_shifted[1] - obs_off_shifted[0]),
        0.5 * (obs_off_shifted[1:] + obs_off_shifted[:-1]),
        obs_off_shifted[-1:] + 0.5 * (obs_off_shifted[-1] - obs_off_shifted[-2]),
    ])

    dm = jnp.diff(m_off)
    resid = conv_flux - 1.0
    cum = jnp.concatenate(
        [jnp.zeros(1), jnp.cumsum(0.5 * (resid[1:] + resid[:-1]) * dm)]
    )

    def cumulative_at(e):
        k = jnp.clip(jnp.searchsorted(m_off, e) - 1, 0, m_off.shape[0] - 2)
        t = e - m_off[k]
        slope = (resid[k + 1] - resid[k]) / dm[k]
        return cum[k] + resid[k] * t + 0.5 * slope * t * t

    cum_edges = cumulative_at(edges)          # one searchsorted over shared edges
    res_flux = 1.0 + jnp.diff(cum_edges) / jnp.diff(edges)
    return res_flux


def _transmission(params: ModelParameters, ctx: ModelContext,
                  config: ModelConfig) -> jnp.ndarray:
    """Telluric (x stellar when enabled) transmission on the observed grid, before
    continuum: optical depth → LSF convolution → wave-solution shift → flux-conserving
    resample, per order. ``forward_model`` multiplies this by the continuum;
    ``telluric_only`` exposes it with the stellar branch disabled.
    """
    y_trans = _optical_depth(ctx, params, config)

    if config.stellar_enabled:
        stellar = apply_stellar(
            ctx.stellar_template, ctx.stellar_dv_kms,
            net_stellar_rv(ctx, params), params.vsini,
            half_width=config.stellar_rot_half_width,
            epsilon=config.stellar_epsilon,
        )
        y_trans = y_trans * stellar

    x_pix = jnp.linspace(-1.0, 1.0, config.order_size)
    process = partial(_process_order, y_trans=y_trans, x_pix=x_pix, config=config,
                      lsf_kernel=ctx.lsf_kernel)
    return jax.vmap(process)(
        params.wave_coeffs, params.resolution_coeffs,
        ctx.order_d_wave, ctx.obs_d_wave, ctx.order_start_idx, ctx.obs_wave_norm,
        ctx.model_wave_off, ctx.obs_wave_off, ctx.order_ref_wave,
    )


def _continuum(params: ModelParameters, ctx: ModelContext) -> jnp.ndarray:
    """Per-order continuum on the observed grid: the B-spline basis times the fitted
    per-order weights, ``(n_orders, n_pixels)``."""
    return jax.vmap(lambda w: ctx.cont_bspline_matrix @ w)(params.continuum_coeffs)


@partial(jax.jit, static_argnums=(2,))
def forward_model(params: ModelParameters, ctx: ModelContext,
                  config: ModelConfig) -> jnp.ndarray:
    """Full modelled flux on the observed grid: the telluric (x stellar) transmission
    times the per-order continuum."""
    return _transmission(params, ctx, config) * _continuum(params, ctx)


@partial(jax.jit, static_argnums=(2,))
def telluric_only(params: ModelParameters, ctx: ModelContext,
                  config: ModelConfig) -> jnp.ndarray:
    """Telluric transmission alone on the observed grid — LSF-convolved, wave-shifted
    and resampled, with NO continuum and the stellar template disabled.
    """
    return _transmission(params, ctx, config._replace(stellar_enabled=False))


def forward_model_batched(params: ModelParameters, ctx: ModelContext,
                          config: ModelConfig, layout: Layout) -> jnp.ndarray:
    """Modelled flux for a timeseries, ``(N, n_orders, n_pixels)``.

    Reuses the single-exposure :func:`forward_model` unchanged, mapped over the N
    exposures with :func:`jax.lax.map` so peak memory stays at one exposure's worth.
    """
    N = layout.n_exp

    def stack(value, site_name):
        """Pass a per-exposure leaf through; broadcast a shared leaf to ``(N, …)``."""
        v = jnp.asarray(value)
        return v if site_name in layout.per_exposure else jnp.broadcast_to(v, (N,) + v.shape)

    mapped = dict(
        t_latent=stack(params.t_latent, "t_latent"),
        h2o_latent=stack(params.h2o_latent, "h2o_latent"),
        continuum_coeffs=stack(params.continuum_coeffs, "continuum_coeffs"),
        wave_coeffs=stack(params.wave_coeffs, "wave_coeffs"),
        resolution_coeffs=stack(params.resolution_coeffs, "resolution_coeffs"),
        dry={s: stack(v, f"dry_vmr_{s}") for s, v in params.dry_vmr_scalars.items()},
        airmass=ctx.airmass,
        baryrv=ctx.baryrv,
    )
    if config.wind_enabled:
        mapped["wind_v_los"] = ctx.wind_v_los                  # (N, n_layers)

    def one(ex):
        params_i = params._replace(
            t_latent=ex["t_latent"], h2o_latent=ex["h2o_latent"],
            continuum_coeffs=ex["continuum_coeffs"], wave_coeffs=ex["wave_coeffs"],
            resolution_coeffs=ex["resolution_coeffs"], dry_vmr_scalars=ex["dry"],
        )  # rv_systemic / vsini stay the shared scalars closed over in ``params``
        ctx_i = ctx._replace(airmass=ex["airmass"], baryrv=ex["baryrv"])
        if config.wind_enabled:
            ctx_i = ctx_i._replace(wind_v_los=ex["wind_v_los"])
        return forward_model(params_i, ctx_i, config)

    return jax.lax.map(one, mapped)
