"""Diagnostic plots for multi-exposure (time-series) fits.

Residual disentangle image, per-exposure atmospheric profiles, and fitted
parameter timelines, plus :func:`plot_timeseries_diagnostics` which writes the
full set (reusing the single-exposure instrument/loss/convergence plots).
"""

import logging
import pathlib
import warnings
from typing import Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from enuma.model.forward import (
    expand_tp_profiles,
    forward_model_batched,
)
from enuma.plotting.spectrum import (
    plot_instrument_diagnostics,
    plot_loss_history,
    plot_param_convergence,
)

__all__ = [
    "plot_timeseries_residuals",
    "plot_timeseries_profiles",
    "plot_timeseries_parameters",
    "plot_timeseries_diagnostics",
]

logger = logging.getLogger("enuma.plotting.timeseries")



def plot_timeseries_residuals(result, output_path="timeseries_residuals.pdf",
                                ):
    """Residual diagnostic for a multi-exposure (time-series) fit.

    The plot stacks the residual image ``obs - model`` one spectral order per
    row, with exposures along the vertical axis and wavelength along the
    horizontal axis. The exposure-mean-subtracted residual ``resid_n`` is shown
    so that order-to-order structure is easier to compare; masked pixels are
    drawn blank.

    ``result`` is a :class:`enuma.inference.fit.FitResult` from a time-series fit.

    The ``order`` argument is retained for compatibility but ignored; all orders
    are plotted.
    """
    wave = np.asarray(result.data["wave"])               # (O, P)
    obs = np.asarray(result.data["flux"])                # (N, O, P)
    msk = np.asarray(jax.device_get(result.obs_mask)).astype(bool)  # (N, O, P)
    full = np.asarray(jax.device_get(forward_model_batched(
        result.params, result.context, result.config, result.layout)))
    N, O, P = obs.shape
    # Masked / zero-model pixels are nan; columns masked in every exposure give an
    # empty nanmean slice. Both are expected and handled by the nan-aware plotting.
    with np.errstate(divide="ignore", invalid="ignore"):
        resid = np.where(msk & (full != 0), obs / full, np.nan)  # (N, O, P)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        resid_n = resid / np.nanmean(resid, axis=0, keepdims=True) - 1. # (N, O, P)

    # Symmetric, robust colour scale centred on zero across all orders.
    vmax = np.nanpercentile(np.abs(resid_n), 95)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = float(np.nanmax(np.abs(resid_n)) or 1e-3)

    fig, axes = plt.subplots(
        nrows=O, ncols=1, figsize=(13, max(3.0, 2.6 * O)),
        sharex=False, constrained_layout=True)
    if O == 1:
        axes = [axes]

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("lightgray")
    last_im = None
    for order_idx, ax in enumerate(axes):
        w = wave[order_idx]
        last_im = ax.imshow(
            resid_n[:, order_idx], aspect="auto", origin="lower", cmap=cmap,
            vmin=-vmax, vmax=vmax, interpolation="nearest",
            extent=[float(w[0]), float(w[-1]), -0.5, N - 0.5],
        )
        ax.set_ylabel("Exposure")
        ax.set_title(f"Order {order_idx}")
        ax.grid(False)
    axes[-1].set_xlabel("Wavelength (nm)")
    fig.colorbar(last_im, ax=axes, pad=0.01).set_label("resid_n")

    fig.savefig(output_path)
    plt.close(fig)

    fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=(13, 4),
            constrained_layout=True)
    ax.plot(wave.ravel(), np.std(resid_n, axis=0).ravel(), color="k", lw=1.2)
    # ax.set_ylim(0.001, 0.05)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Residual stddev")
    fig.savefig(output_path.replace(".pdf", "_std.pdf"))
    plt.close(fig)

    logger.info("Saved time-series residual diagnostic → %s", output_path)


# ---------------------------------------------------------------------------
# Time-series: atmospheric profiles + parameter timelines
# ---------------------------------------------------------------------------

def _timeseries_profiles(result) -> dict:
    """Batched per-exposure T and H2O profiles for a time-series :class:`FitResult`.

    Expands each exposure's GP latents with the canonical forward-model formula
    (:func:`enuma.model.forward.expand_tp_profiles`) vmapped over the N exposures,
    broadcasting shared (1-D) latents so every exposure shares one profile.
    Returns numpy arrays keyed by name; the per-exposure 2D arrays are
    ``(n_layers, N)`` so a column is one exposure (line family) and a row is one
    altitude across the night (heatmap).
    """
    ctx, p = result.context, result.params
    n_exp = int(np.asarray(jax.device_get(ctx.airmass)).reshape(-1).shape[0])
    z_km = np.asarray(jax.device_get(ctx.z_centers)) / 1e5            # (n_layers,)
    t_ref = np.asarray(jax.device_get(ctx.t_ref_centers))            # (n_layers,)
    vmr_ref_h2o = np.asarray(jax.device_get(ctx.vmr_ref_centers["h2o"]))

    # Latents are (N, n*) per-exposure or (n*,) shared — broadcast the shared case
    # so every exposure column expands to the same profile.
    t_lat, h2o_lat = p.t_latent, p.h2o_latent
    if t_lat.ndim == 1:
        t_lat = jnp.broadcast_to(t_lat, (n_exp,) + t_lat.shape)
    if h2o_lat.ndim == 1:
        h2o_lat = jnp.broadcast_to(h2o_lat, (n_exp,) + h2o_lat.shape)

    def _expand(tl, hl):
        return expand_tp_profiles(ctx, p._replace(t_latent=tl, h2o_latent=hl))

    t_dev_b, t_prof_b, h2o_dex_b = jax.vmap(_expand)(t_lat, h2o_lat)  # (N, n_layers)
    t_dev = np.asarray(jax.device_get(t_dev_b)).T                     # (n_layers, N)
    t_profile = np.asarray(jax.device_get(t_prof_b)).T
    h2o_dex = np.asarray(jax.device_get(h2o_dex_b)).T
    h2o_vmr = vmr_ref_h2o[:, None] * (10.0 ** h2o_dex)               # (n_layers, N)

    # Layers carrying a free GP latent (the rest are pinned at climatology, so
    # their deviation is identically zero and dilutes column-mean proxies).
    temp_L = np.asarray(jax.device_get(ctx.temp_chol_L))
    h2o_L = np.asarray(jax.device_get(ctx.h2o_chol_L))
    active = np.any(temp_L != 0, axis=1) | np.any(h2o_L != 0, axis=1)
    return dict(z_km=z_km, t_ref=t_ref, vmr_ref_h2o=vmr_ref_h2o, t_dev=t_dev,
                t_profile=t_profile, h2o_dex=h2o_dex, h2o_vmr=h2o_vmr, active=active)


def _timeseries_time_axis(result):
    """``(x (N,), x_label)`` — hours since the first exposure (MJD), else index."""
    # Derive N from airmass (always per-exposure); t_latent may be shared (1-D).
    n_exp = int(np.asarray(jax.device_get(result.context.airmass)).reshape(-1).shape[0])
    mjd = result.data.get("mjd")
    if mjd is not None and np.asarray(mjd).size == n_exp:
        mjd = np.asarray(mjd, dtype=float)
        return (mjd - mjd[0]) * 24.0, "Time since first exp. [hr]"
    return np.arange(n_exp, dtype=float), "Exposure index"


def _cell_edges(values: np.ndarray) -> np.ndarray:
    """Monotonic cell edges around ``values`` (for ``pcolormesh``)."""
    v = np.asarray(values, dtype=float)
    if v.size == 1:
        return np.array([v[0] - 0.5, v[0] + 0.5])
    mid = 0.5 * (v[:-1] + v[1:])
    return np.concatenate([[v[0] - (mid[0] - v[0])], mid, [v[-1] + (v[-1] - mid[-1])]])


def plot_timeseries_profiles(result, output_path: str = "timeseries_profiles.pdf",
                             altitude_band_km=(2.0, 40.0)):
    """Per-exposure atmospheric profiles for a multi-exposure (time-series) fit.

    Top row: the family of fitted temperature and H2O-VMR profiles, one curve per
    exposure coloured by time, with the reference climatology (dashed). Bottom row: the
    per-layer *deviation* from the reference (ΔT/T_ref in %, and H2O log10 VMR
    offset) as altitude×time heatmaps, restricted to the sensitivity band
    ``altitude_band_km`` (above it the layers are pinned at climatology).

    ``result`` is a :class:`enuma.inference.fit.FitResult` from a time-series fit.
    """
    prof = _timeseries_profiles(result)
    x, xlabel = _timeseries_time_axis(result)
    z_km = prof["z_km"]
    n_exp = prof["t_profile"].shape[1]

    norm = plt.Normalize(vmin=float(np.min(x)), vmax=float(np.max(x)))
    cmap = plt.get_cmap("viridis")
    colors = cmap(norm(x))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), constrained_layout=True)

    # --- Row 1: line families (one curve per exposure, coloured by time). ---
    ax_t = axes[0, 0]
    for i in range(n_exp):
        ax_t.plot(prof["t_profile"][:, i], z_km, color=colors[i], alpha=0.7, lw=1)
    ax_t.plot(prof["t_ref"], z_km, "k--", lw=1.6, alpha=0.8, label="Reference", zorder=6)
    ax_t.set(xlim=(190, 300), yscale="log", ylim=(2, 70),
             xlabel="Temperature (K)", ylabel="Altitude (km)", title="Temperature profiles")
    ax_t.legend(fontsize="small", loc="upper right")
    ax_t.grid(True, alpha=0.3, which="both"); ax_t.minorticks_on()

    ax_h = axes[0, 1]
    for i in range(n_exp):
        ax_h.plot(prof["h2o_vmr"][:, i], z_km, color=colors[i], alpha=0.7, lw=1)
    ax_h.plot(prof["vmr_ref_h2o"], z_km, "k--", lw=1.6, alpha=0.8, label="Reference", zorder=6)
    ax_h.set(xscale="log", yscale="log", xlim=(8e-9, 1e0), ylim=(2, 70),
             xlabel="H2O Volume Mixing Ratio", title="H2O VMR profiles")
    ax_h.legend(fontsize="small", loc="upper right")
    ax_h.grid(True, alpha=0.3, which="both"); ax_h.minorticks_on()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=[axes[0, 0], axes[0, 1]], pad=0.01, location="right").set_label(xlabel)

    # --- Row 2: deviation heatmaps (altitude × time). ---
    band = (z_km >= altitude_band_km[0]) & (z_km <= altitude_band_km[1])
    x_edges, z_edges = _cell_edges(x), _cell_edges(z_km[band])

    tdev_pct = 100.0 * prof["t_dev"][band, :]
    vmax_t = float(np.nanpercentile(np.abs(tdev_pct), 95)) or 1e-3
    im_t = axes[1, 0].pcolormesh(x_edges, z_edges, tdev_pct, cmap="RdBu_r",
                                 vmin=-vmax_t, vmax=vmax_t, shading="flat")
    axes[1, 0].set(yscale="log", xlabel=xlabel, ylabel="Altitude (km)",
                   title="Temperature deviation [%]")
    fig.colorbar(im_t, ax=axes[1, 0], pad=0.01).set_label("ΔT / T_ref [%]")

    hdex = prof["h2o_dex"][band, :]
    vmax_h = float(np.nanpercentile(np.abs(hdex), 95)) or 1e-3
    im_h = axes[1, 1].pcolormesh(x_edges, z_edges, hdex, cmap="RdBu_r",
                                 vmin=-vmax_h, vmax=vmax_h, shading="flat")
    axes[1, 1].set(yscale="log", xlabel=xlabel, title="H2O log10 VMR offset")
    fig.colorbar(im_h, ax=axes[1, 1], pad=0.01).set_label("H2O dex")

    fig.suptitle(f"Time-series atmospheric profiles ({n_exp} exposures)", fontsize=12)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved time-series profiles → %s", output_path)


def plot_timeseries_parameters(result, output_path: str = "timeseries_parameters.pdf"):
    """Key fitted parameters versus time for a multi-exposure (time-series) fit.

    Timeline panels (x = time since first exposure, or exposure index): the
    per-exposure dry-species relative columns, an H2O-column proxy (mean H2O dex
    over the sensitive layers), and a temperature proxy (mean ΔT/T over those
    layers). Airmass is overlaid on the dry-species and H2O panels (twin axis) so
    transparency correlations are visible. Shared scalars (vsini, systemic RV,
    mean R) are annotated in the title.

    ``result`` is a :class:`enuma.inference.fit.FitResult` from a time-series fit.
    """
    ctx, p = result.context, result.params
    x, xlabel = _timeseries_time_axis(result)
    n_exp = x.size
    prof = _timeseries_profiles(result)
    active = prof["active"]

    airmass = np.asarray(jax.device_get(ctx.airmass)).reshape(-1)
    h2o_col = np.nanmean(prof["h2o_dex"][active, :], axis=0)          # (N,)
    t_col = 100.0 * np.nanmean(prof["t_dev"][active, :], axis=0)      # (N,) percent

    # Dry-species relative columns (10**dex). Restrict to the fitted species when
    # given so fixed-at-reference columns (flat at 1.0) don't clutter the panel.
    fit_species = result.fit_config.fit_species
    keep = set(fit_species) if fit_species is not None else None
    dry_traces = []  # (species, values (N,), is_per_exposure)
    for s in result.dry_species:
        if keep is not None and s not in keep:
            continue
        val = np.asarray(jax.device_get(result.sites[f"dry_vmr_{s}"])).reshape(-1)
        per_exp = val.size == n_exp
        rel = 10.0 ** (val if per_exp else np.full(n_exp, val[0]))
        dry_traces.append((s, rel, per_exp))

    def _airmass_twin(ax):
        axt = ax.twinx()
        axt.plot(x, airmass, color="0.5", ls=":", lw=1.2, alpha=0.8)
        axt.set_ylabel("airmass", color="0.5", fontsize="small")
        axt.tick_params(axis="y", labelcolor="0.5", labelsize="small")
        return axt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, constrained_layout=True)

    # Panel 0: dry-species relative columns + airmass twin.
    for i, (s, rel, per_exp) in enumerate(dry_traces):
        c = f"C{i % 10}"
        if per_exp:
            axes[0].plot(x, rel, ".-", color=c, lw=1.2, label=s)
        else:
            axes[0].axhline(float(rel[0]), color=c, ls="--", lw=1.2, label=f"{s} (shared)")
    axes[0].set(ylabel=r"Dry column $10^{\rm dex}$", title="Dry-species relative columns")
    axes[0].legend(fontsize="x-small", ncol=2)
    _airmass_twin(axes[0])

    # Panel 1: H2O column proxy + airmass twin.
    axes[1].plot(x, h2o_col, "C0.-", lw=1.2)
    axes[1].set(ylabel="Mean H2O dex", title="H2O column proxy")
    _airmass_twin(axes[1])

    # Panel 2: temperature proxy.
    axes[2].plot(x, t_col, "C3.-", lw=1.2)
    # axes[2].axhline(0.0, color="k", ls=":", lw=0.8, alpha=0.6)
    axes[2].set(ylabel=r"Mean $\Delta T/T_{\rm ref}$ [%]", title="Temperature proxy")

    for ax in axes:
        ax.grid(True, alpha=0.3); ax.minorticks_on()
        ax.set_xlabel(xlabel)

    vsini = float(np.asarray(jax.device_get(p.vsini)))
    rv_sys = float(np.asarray(jax.device_get(result.sites["rv_systemic"])).reshape(-1)[0])
    mean_R = 1e5 * float(np.exp(np.mean(np.asarray(jax.device_get(p.resolution_coeffs))[..., -1])))
    fig.suptitle(f"Time-series parameters — shared: vsini={vsini:.2f} km/s, "
                 f"systemic RV={rv_sys:.2f} km/s, mean R≈{mean_R:,.0f}", fontsize=12)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved time-series parameter timelines → %s", output_path)


def plot_timeseries_diagnostics(result, output_dir: str) -> None:
    """Write the full time-series diagnostic set to ``output_dir``.

    Mirrors the single-exposure :meth:`enuma.inference.fit.FitResult.plot`: residual
    disentangle image, atmospheric profiles, parameter timelines, instrument
    R(λ)/wave-shift, loss history, and SVI parameter-convergence traces.

    ``result`` is a :class:`enuma.inference.fit.FitResult` from a time-series fit.
    """
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_timeseries_residuals(result, output_path=str(out / "residuals.pdf"))
    plot_timeseries_profiles(result, output_path=str(out / "profiles.pdf"))
    plot_timeseries_parameters(result, output_path=str(out / "parameters.pdf"))
    # The instrument diagnostic is a single-exposure plot (R(λ)/wave shift per
    # order). The LSF/wave coeffs are usually shared (O, ...); when fit per exposure
    # (resolution_per_exposure / wave_per_exposure) they are (N, O, ...), so collapse
    # the exposure axis to the night-mean for display.
    p = result.params
    inst_params = p._replace(
        resolution_coeffs=(p.resolution_coeffs.mean(axis=0)
                           if p.resolution_coeffs.ndim == 3 else p.resolution_coeffs),
        wave_coeffs=(p.wave_coeffs.mean(axis=0)
                     if p.wave_coeffs.ndim == 3 else p.wave_coeffs),
    )
    plot_instrument_diagnostics(inst_params, result.context, result.config,
                                output_path=str(out / "instrument.pdf"))
    plot_loss_history(result.losses, output_path=str(out / "loss_history.pdf"))
    plot_param_convergence(result.param_history, output_path=str(out / "param_convergence.pdf"))
    logger.info("Saved time-series diagnostics → %s", out)
