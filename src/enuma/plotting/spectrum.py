"""Single-exposure diagnostic plots + shared SVI diagnostics.

`plot_fit_results` writes the spectrum/residual panels plus a VMR + temperature
profile figure. `plot_residuals_diagnostic` bins residuals by model line depth
to disentangle LSF/wing vs saturation/RT error modes. `plot_instrument_diagnostics`,
`plot_loss_history`, and `plot_param_convergence` are reused by the time-series
diagnostics (see :mod:`enuma.plotting.timeseries`).
"""

import logging
import math
from typing import Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

from enuma.model.forward import expand_tp_profiles, forward_model, telluric_only
from enuma.state import ModelConfig, ModelContext, ModelParameters

__all__ = [
    "plot_fit_results",
    "plot_residuals_diagnostic",
    "plot_instrument_diagnostics",
    "plot_param_convergence",
    "plot_loss_history",
]

logger = logging.getLogger("enuma.plotting.spectrum")

def _set_plot_style(size=13):
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({
        'font.size': size,
        # 'font.family': 'sans-serif',
        # 'font.serif': 'Helvetica',
        # 'pdf.fonttype': 42,
        "xtick.labelsize": size,   
        "ytick.labelsize": size,   
        "xtick.direction": 'in', 
        "ytick.direction": 'in', 
        'ytick.right': True,
        'xtick.top': True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        # "xtick.major.size": 7,
        # "xtick.minor.size": 3.5,
        # "xtick.major.width": 2,
        # "xtick.minor.width": 2,
        # "ytick.major.size": 7,
        # "ytick.minor.size": 3.5,
        # "ytick.major.width": 2,
        # "ytick.minor.width": 2,
        "lines.linewidth": 1.,   
        'image.origin': 'lower',
        'image.cmap': 'magma',
        "savefig.dpi": 300,   
        })
        
def _load_comparison_model(comparison_fits: Optional[str]):
    """Read a 2-column wave/flux comparison spectrum, or return None."""
    if comparison_fits is None:
        return None
    try:
        data = fits.getdata(comparison_fits)
    except (FileNotFoundError, OSError) as exc:
        logger.warning("Could not read comparison FITS '%s': %s", comparison_fits, exc)
        return None
    return interp1d(data[:, 0], data[:, 1], kind='cubic', fill_value='extrapolate')


def plot_fit_results(
    optimized_params: ModelParameters,
    context: ModelContext,
    config: ModelConfig,
    obs_flux: jnp.ndarray,
    obs_mask: jnp.ndarray = None,
    output_path: str = "fit_results.png",
    fit_species=None,
    profile_output_path: Optional[str] = None,
    benchmark: bool = True,
    comparison_fits: Optional[str] = None,
):
    """Visualise the telluric fit: a spectrum/residual figure (``output_path``)
    and a VMR + temperature profile figure (``profile_output_path``).

    Args:
        comparison_fits: optional 2-column reference spectrum (e.g. molecfit)
            overplotted on the residual panels when ``benchmark=True``.
    """
    flux_opt = jax.device_get(forward_model(optimized_params, context, config))
    # Telluric transmission (stellar disabled); obs / telluric ≈ telluric-corrected spectrum.
    flux_tell = jax.device_get(telluric_only(optimized_params, context, config))
    wave_nm = jax.device_get(context.obs_wave)
    obs_flux_np = jax.device_get(obs_flux)
    obs_mask_np = jax.device_get(obs_mask).astype(bool) if obs_mask is not None else None
    model_m = _load_comparison_model(comparison_fits) if benchmark else None

    _set_plot_style()

    _plot_spectrum_figure(optimized_params, context, wave_nm, obs_flux_np, flux_opt,
                          flux_tell, obs_mask_np, model_m, output_path)

    if profile_output_path is None:
        profile_output_path = output_path.replace('.png', '_profiles.png')
    _plot_profile_figure(optimized_params, context, fit_species, profile_output_path)


def _plot_spectrum_figure(optimized_params, context, wave_nm, obs_flux_np, flux_opt,
                          flux_telluric, obs_mask_np, model_m, output_path):
    """Multi-row spectrum + residual panels (3 orders per row).

    Each flux panel overlays the observation, the full model, the fitted
    continuum, the telluric transmission (``flux_telluric``), and the
    telluric-divided observation (``obs / telluric``), which approximates the
    stellar spectrum once the atmosphere is removed.
    """
    n_orders = obs_flux_np.shape[0]
    n_cols = 3
    n_rows = math.ceil(n_orders / n_cols)
    fig, axes = plt.subplots(
        nrows=n_rows * 2, ncols=1, figsize=(14, max(3.0, 3.0 * n_rows)),
        gridspec_kw={'height_ratios': [1, 1] * n_rows},
        sharex=False, constrained_layout=True,
    )

    for row in range(n_rows):
        fx, rx = axes[row * 2], axes[row * 2 + 1]
        rx.sharex(fx)
        x_min = x_max = None
        for i in range(row * n_cols, min(n_orders, row * n_cols + n_cols)):
            # Labels only on the very first order, so the legend isn't repeated.
            lab_data, lab_opt, lab_cont, lab_tell, lab_star = (
                ('Observed Data', 'Optimized Model', 'Fitted Continuum',
                 'Telluric Transmission', 'Obs / Telluric (≈ Stellar)')
                if i == 0 else (None, None, None, None, None)
            )
            fx.plot(wave_nm[i], obs_flux_np[i], 'k-', label=lab_data, alpha=0.9, linewidth=1)
            fx.plot(wave_nm[i], flux_opt[i], 'C1-', label=lab_opt, alpha=0.9, linewidth=1)

            # Disentangle the two components: the telluric transmission, and the
            # telluric-divided observation that approximates the stellar spectrum.
            # The division is unreliable in saturated telluric cores, so mask pixels
            # where the telluric transmission drops below 0.1.
            fx.plot(wave_nm[i], flux_telluric[i], 'C0-', label=lab_tell, alpha=0.7, linewidth=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                stellar_approx = obs_flux_np[i] / flux_telluric[i]
            stellar_approx = np.where(flux_telluric[i] > 0.1, stellar_approx, np.nan)
            if obs_mask_np is not None:
                stellar_approx = np.where(obs_mask_np[i], stellar_approx, np.nan)
            fx.plot(wave_nm[i], stellar_approx, 'C3-', label=lab_star, alpha=0.6, linewidth=1)

            residuals = obs_flux_np[i] - flux_opt[i]
            if obs_mask_np is not None:
                residuals = np.where(obs_mask_np[i], residuals, np.nan)
            rx.plot(wave_nm[i], residuals, 'C1-', alpha=0.8, linewidth=1)

            continuum = jnp.dot(context.cont_bspline_matrix, optimized_params.continuum_coeffs[i])
            fx.plot(wave_nm[i], continuum, 'C2:', label=lab_cont, alpha=0.8, linewidth=1)

            if model_m is not None:
                res_cmp = obs_flux_np[i] / continuum - model_m(wave_nm[i])
                res_cmp -= gaussian_filter(res_cmp, sigma=30)
                if obs_mask_np is not None:
                    res_cmp = np.where(obs_mask_np[i], res_cmp, np.nan)
                rx.plot(wave_nm[i], res_cmp, 'k-', alpha=0.8, linewidth=1, zorder=0)

            x_min = wave_nm[i][0] if x_min is None else min(x_min, wave_nm[i][0])
            x_max = wave_nm[i][-1] if x_max is None else max(x_max, wave_nm[i][-1])

        if x_min is not None:
            rx.set_xlim(float(x_min), float(x_max))
        fx.set_ylabel("Flux")
        fx.tick_params(labelbottom=False)
        rx.set_ylabel("Obs. - Mod.")
        rx.set_ylim(-0.03, 0.03)
        for ax in (fx, rx):
            ax.grid(True, alpha=0.25)
            ax.minorticks_on()
        if row == 0:
            fx.legend(fontsize='small')

    axes[-1].set_xlabel("Wavelength (nm)")
    fig.savefig(output_path)
    # plt.show()
    plt.close(fig)


def _plot_profile_figure(optimized_params, context, fit_species, output_path):
    """VMR profiles (left) + temperature profile (right): initial vs converged.

    The initial profile is the reference atmosphere (the GP latents / dex offsets
    all start at zero); the converged profile expands the fitted latents with the
    same formula the forward model uses (``enuma.model.forward.expand_tp_profiles``),
    so the plotted T and H2O can never drift from the fitted model. Initial profiles
    are dashed, converged solid.
    """
    z_km = jax.device_get(context.z_centers) / 1e5
    colors = [f"C{i}" for i in range(10)]
    _, t_opt, h2o_dex = expand_tp_profiles(context, optimized_params)

    fig = plt.figure(figsize=(8, 4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

    # --- VMR profiles: reference (initial, dashed) vs fitted (converged, solid).
    # H2O uses the per-layer dex from the GP expansion; dry species use their
    # scalar dex offset. Restrict to `fit_species` if given.
    ax_vmr = fig.add_subplot(gs[0, 0])
    plot_species = []
    if 'h2o' in context.vmr_ref_centers:
        plot_species.append(('h2o', h2o_dex))
    plot_species += list((optimized_params.dry_vmr_scalars or {}).items())
    if fit_species is not None:
        keep = set(fit_species)
        plot_species = [(s, v) for s, v in plot_species if s in keep]

    for i, (species, opt_dex) in enumerate(plot_species):
        c = colors[i % len(colors)]
        vmr_ref = context.vmr_ref_centers[species]
        ax_vmr.plot(jax.device_get(vmr_ref * (10 ** opt_dex)), z_km,
                    color=c, linestyle='-', linewidth=1, label=species)
        ax_vmr.plot(jax.device_get(vmr_ref), z_km, color=c, linestyle='--', alpha=0.6)
    ax_vmr.plot([], [], color='0.4', linestyle='--', label='initial')

    ax_vmr.set(xscale='log', yscale='log', xlim=(8e-9, 1e0), ylim=(2, 70),
               xlabel="Volume Mixing Ratio", ylabel="Altitude (km)", title="Vertical VMR Profiles")
    ax_vmr.legend(loc='upper right', fontsize='small')
    ax_vmr.grid(True, alpha=0.3, which='both')
    ax_vmr.minorticks_on()

    # --- Temperature: reference (initial, dashed) vs fitted (converged, solid),
    # T_ref * (1 + temp_chol_L @ t_latent), matching the forward model.
    ax_t = fig.add_subplot(gs[0, 1], sharey=ax_vmr)
    ax_t.plot(jax.device_get(t_opt), z_km, color='k', linestyle='-', linewidth=1, label='Fitted')
    ax_t.plot(jax.device_get(context.t_ref_centers), z_km, color='k', linestyle='--',
              linewidth=1, alpha=0.6, label='Initial')
    ax_t.set(xlim=(190, 300), yscale='log', xlabel="Temperature (K)", title="Temperature Profile")
    ax_t.legend(loc='upper right', fontsize='small')
    ax_t.grid(True, alpha=0.3, which='both')
    ax_t.minorticks_on()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_residuals_diagnostic(
    optimized_params: ModelParameters,
    context: ModelContext,
    config: ModelConfig,
    obs_flux: jnp.ndarray,
    obs_mask: jnp.ndarray = None,
    n_bins: int = 20,
    output_path: str = "residuals_diagnostic.png",
    comparison_fits: Optional[str] = None,
):
    """Bin residuals by model line depth to diagnose dominant error mode.

    Expected residual-RMS scaling with line depth d = 1 - model_flux:
      - d^0: noise / continuum bias / wavelength solution
      - d^1: LSF wing/shape mismatch
      - d^2+: saturation, RT nonlinearity

    Args:
        comparison_fits: optional path to a 2-column reference spectrum
            (e.g. molecfit) — overlaid for benchmarking. None disables.
    """
    model_m = _load_comparison_model(comparison_fits)
    flux_cmp = model_m(jax.device_get(context.obs_wave)) if model_m is not None else None

    model_flux = jax.device_get(forward_model(optimized_params, context, config))
    obs_np = jax.device_get(obs_flux)
    # Zero-model pixels divide to inf/nan; the np.isfinite keep-filter below drops them.
    with np.errstate(divide="ignore", invalid="ignore"):
        residuals = obs_np / model_flux - 1.0
    if flux_cmp is not None:
        continuum = jnp.stack([
            jnp.dot(context.cont_bspline_matrix, optimized_params.continuum_coeffs[i])
            for i in range(obs_np.shape[0])
        ])
        residuals_cmp = obs_np / flux_cmp / continuum - 1.0
        residuals_cmp -= np.nanmedian(residuals_cmp, axis=1, keepdims=True)
    else:
        residuals_cmp = None

    if obs_mask is not None:
        mask = jax.device_get(obs_mask).astype(bool)
    else:
        mask = np.ones_like(model_flux, dtype=bool)

    flat_model = model_flux.flatten()
    flat_resid = residuals.flatten()
    flat_resid_cmp = residuals_cmp.flatten() if residuals_cmp is not None else None
    keep = mask.flatten() & np.isfinite(flat_model) & np.isfinite(flat_resid)
    flat_model = flat_model[keep]
    flat_resid = flat_resid[keep]
    if flat_resid_cmp is not None:
        flat_resid_cmp = flat_resid_cmp[keep]

    # Bin by model_flux. Restrict to [0, 1.05] to avoid wild points.
    lo, hi = 0.0, min(float(np.nanmax(flat_model)) * 1.05, 1.2)
    bin_edges = np.linspace(lo, hi, n_bins + 1)
    bin_idx = np.clip(np.digitize(flat_model, bin_edges) - 1, 0, n_bins - 1)

    rms = np.full(n_bins, np.nan)
    rms_cmp = np.full(n_bins, np.nan) if flat_resid_cmp is not None else None
    for b in range(n_bins):
        sel = bin_idx == b
        if sel.sum() > 5:
            rms[b] = np.sqrt(np.mean(flat_resid[sel] ** 2))
            if flat_resid_cmp is not None:
                rms_cmp[b] = np.sqrt(np.mean(flat_resid_cmp[sel] ** 2))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    valid = np.isfinite(rms) & (bin_centers < 1)
    bin_centers, rms = bin_centers[valid], rms[valid]
    if rms_cmp is not None:
        rms_cmp = rms_cmp[valid]
    ref_d1 = 1.0 / np.sqrt(bin_centers) * rms[-1]

    # RMS vs depth, log-log, with a reference power law.
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(bin_centers, rms, 'ro-', markersize=5, label='RMS')
    if rms_cmp is not None:
        ax.loglog(bin_centers, rms_cmp, 'ko-', markersize=5, label='RMS (comparison)')
    ax.loglog(bin_centers, ref_d1, 'C0--', alpha=0.7, label=r'$\propto F^{-1/2}$')
    ax.set_xlabel("Model flux")
    ax.set_ylabel("Residual RMS")
    ax.set_title("Residual RMS vs Line Depth")
    ax.legend(fontsize='small')
    ax.grid(True, alpha=0.3, which='both')
    ax.minorticks_on()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved residual decomposition → %s", output_path)


def plot_instrument_diagnostics(
    optimized_params: ModelParameters,
    context: ModelContext,
    config: ModelConfig,
    output_path: str = "instrument_diagnostics.png",
):
    """Plot the fitted spectral resolution R(λ) and wavelength shift per order."""
    n_orders = context.obs_wave.shape[0]
    n_pixels_obs = context.obs_wave.shape[1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    pixels = np.arange(n_pixels_obs)
    x_pix_model = np.linspace(-1.0, 1.0, config.order_size)
    model_wave_np = jax.device_get(context.model_wave)
    obs_wave_np = jax.device_get(context.obs_wave)
    order_start_idx = jax.device_get(context.order_start_idx)

    for i in range(n_orders):
        # Wavelength shift: w_coeffs are already in pixel units (see
        # model.process_order), so the polynomial on obs_wave_norm is the shift.
        obs_wn = jax.device_get(context.obs_wave_norm[i])
        w_coeffs = jax.device_get(optimized_params.wave_coeffs[i])
        pixel_shift = np.polyval(w_coeffs, obs_wn)

        # Spectral resolution R, mapped from the model grid onto observed pixels.
        r_coeffs = jax.device_get(optimized_params.resolution_coeffs[i])
        R_model = 1e5 * np.exp(np.polyval(r_coeffs, x_pix_model))   # log-space R(λ)
        start_idx = int(order_start_idx[i])
        m_wave = model_wave_np[start_idx: start_idx + config.order_size]
        R_obs = np.interp(obs_wave_np[i], m_wave, R_model)

        color = plt.cm.viridis(i / max(1, n_orders - 1))
        ax1.plot(pixels, R_obs, color=color, alpha=0.8, label=f"Order {i}")
        ax2.plot(pixels, pixel_shift, color=color, alpha=0.8)

    ax1.set_ylabel("Spectral Resolution ($R$)")
    ax1.set_title("Fitted Spectral Resolution per Order")
    ax1.grid(True, alpha=0.3)
    ax1.minorticks_on()
    if n_orders <= 10:
        ax1.legend(fontsize="small", ncol=2, loc="best")
    else:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=n_orders - 1))
        sm.set_array([])
        fig.colorbar(sm, ax=[ax1, ax2], pad=0.02).set_label("Order index")

    ax2.set_xlabel("Pixel Number")
    ax2.set_ylabel("Wavelength Shift (pixels)")
    ax2.set_title("Wavelength Solution Shift")
    ax2.grid(True, alpha=0.3)
    ax2.minorticks_on()

    fig.savefig(output_path)
    plt.close(fig)


def plot_param_convergence(param_history, output_path: str = "fit_param_convergence.png") -> None:
    """Write the per-parameter SVI convergence-trace PDF.

    One panel per fitted site; each thin line is one scalar component's value vs
    SVI step. A converged fit shows flat tails; a component that keeps drifting
    (e.g. a wave zero-point in a line-poor order) shows a trace that never
    settles. ``param_history`` is the ``(recorded_steps, {name: trace})`` tuple
    returned by :func:`enuma.inference.optimizer.run_svi` (``None`` => skipped).
    """
    if param_history is None:
        logger.info("No parameter history — skipping %s.", output_path)
        return
    rec_steps, traces = param_history
    rec_steps = np.asarray(rec_steps)
    if not traces or rec_steps.size == 0:
        logger.info("No parameter history to plot — skipping %s.", output_path)
        return

    names = list(traces.keys())
    ncols = min(3, len(names))
    nrows = math.ceil(len(names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 2.6 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for k, name in enumerate(names):
        ax = axes.flat[k]
        ax.set_visible(True)
        flat = np.asarray(traces[name]).reshape(rec_steps.size, -1)
        for j in range(flat.shape[1]):
            ax.plot(rec_steps, flat[:, j], lw=0.7, alpha=0.7)
        ax.set_title(f"{name.replace('_auto_loc', '')}  (n={flat.shape[1]})", fontsize=9)
        ax.set_xlabel("SVI step")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Parameter convergence traces", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_loss_history(losses, output_path: str = "fit_loss_history.png") -> None:
    """Write the SVI loss-history PDF.

    Y-axis is log-scaled when every loss is positive, else linear. Only the
    second half of the trajectory is plotted so the late descent isn't squashed
    by the early-iteration tail.
    """
    arr = np.asarray(losses, dtype=float)
    if arr.size == 0:
        logger.info("No loss history to plot — skipping %s.", output_path)
        return

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.plot(np.arange(len(arr))[len(arr) // 2:], arr[len(arr) // 2:], color="tab:blue", lw=1.2)
    if np.all(arr > 0):
        ax.set_yscale("log")
    ax.set_xlabel("SVI step")
    ax.set_ylabel("ELBO loss")
    ax.set_title(f"final = {arr[-1]:.3g}  (n = {len(arr)})")
    ax.grid(True, alpha=0.3, which="both")
    ax.minorticks_on()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved loss history → %s", output_path)
