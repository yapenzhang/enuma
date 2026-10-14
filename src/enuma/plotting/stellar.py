"""Diagnostic plots for the learned stellar template (``stellar_model="gp"``).

The template itself with its error bars, the night-mean (static) residual per order
— the part of the residual the time-series residual image removes — and the
round-by-round history of the template fit, plus
:func:`plot_stellar_diagnostics`, which writes them all and saves the template.
"""

import logging
import pathlib
import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

from enuma.model.forward import forward_model_batched

__all__ = [
    "plot_stellar_template",
    "plot_static_residuals",
    "plot_stellar_history",
    "plot_stellar_diagnostics",
]

logger = logging.getLogger("enuma.plotting.stellar")


def plot_stellar_template(result, output_path: str = "stellar_template.pdf"):
    """The learned stellar template per order, in the stellar rest frame.

    Shows the posterior mean with its ±1σ band, the prior anchor
    ``1 + s_o (reference − 1)`` and the reference spectrum. Grey bands mark nodes the
    data barely constrain (posterior σ above half the prior σ), e.g. under
    saturated telluric lines, where the template falls back to the anchor.

    ``result`` is a :class:`~enuma.inference.fit.FitResult` with ``stellar`` set.
    """
    t = result.stellar
    n_orders = t.wave.shape[0]
    fig, axes = plt.subplots(n_orders, 1, figsize=(13, max(3.0, 2.4 * n_orders)),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for o, ax in enumerate(axes):
        ok = np.isfinite(t.wave[o])
        w, f, s = t.wave[o][ok], t.flux[o][ok], t.sigma[o][ok]
        ax.plot(w, t.reference[o][ok], color="0.6", lw=0.7, label="reference")
        ax.plot(w, t.anchor[o][ok], "--", color="C1", lw=0.8, label="prior anchor")
        ax.fill_between(w, f - s, f + s, color="C0", alpha=0.35, lw=0)
        ax.plot(w, f, color="C0", lw=0.9, label="learned template")
        weak = np.isfinite(s) & (s > 0.5 * t.prior_sigma[o][ok])
        ax.fill_between(w, 0, 1, where=weak, transform=ax.get_xaxis_transform(),
                        color="0.85", alpha=0.6, lw=0)
        ax.set_xlim(w[0], w[-1])
        ax.set_ylabel("Normalised flux")
        ax.set_title(f"Order {o}: reference scale s = {t.hyper['scale'][o]:.2f}, "
                     f"prior σ = {t.hyper['sigma'][o]:.3f}, "
                     f"ℓ = {t.hyper['ell_kms'][o]:.2f} km/s", fontsize="medium")
    axes[0].legend(fontsize="small", ncol=3, loc="lower left")
    axes[-1].set_xlabel("Rest-frame wavelength (nm)")
    fig.suptitle(f"Learned stellar template (RV = {t.rv_kms:.2f} km/s)", fontsize=12)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved stellar template → %s", output_path)


def plot_static_residuals(result, output_path: str = "static_residuals.pdf"):
    """Night-mean residual ``obs / model − 1`` per order, with its noise level.

    A model that fits to the noise leaves a night-mean residual at the noise of the
    mean (``median σ / √N``); a mismatched stellar template leaves line-shaped
    structure here. The time-series residual image subtracts exactly this mean, so
    this plot is its complement.

    ``result`` is a :class:`~enuma.inference.fit.FitResult` from a time-series fit.
    """
    wave = np.asarray(result.data["wave"])                         # (O, P)
    obs = np.asarray(result.data["flux"])                          # (N, O, P)
    err = np.asarray(result.data["err"])
    msk = np.asarray(jax.device_get(result.obs_mask)).astype(bool)
    full = np.asarray(jax.device_get(forward_model_batched(
        result.params, result.context, result.config, result.layout)))
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        resid = np.where(msk & (full != 0), obs / full - 1.0, np.nan)
        mean = np.nanmean(resid, axis=0)                           # (O, P)
        n_ok = np.sum(np.isfinite(resid), axis=0)
        noise = np.nanmedian(np.where(msk, err / np.abs(full), np.nan), axis=0) / np.sqrt(
            np.maximum(n_ok, 1))

    n_orders = wave.shape[0]
    fig, axes = plt.subplots(n_orders, 1, figsize=(13, max(3.0, 2.0 * n_orders)),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for o, ax in enumerate(axes):
        rms = float(np.sqrt(np.nanmean(mean[o] ** 2)))
        ax.plot(wave[o], mean[o], color="k", lw=0.7)
        ax.fill_between(wave[o], -noise[o], noise[o], color="C0", alpha=0.3, lw=0)
        ax.axhline(0.0, color="0.5", lw=0.6)
        ax.set_ylabel("⟨obs/model⟩ − 1")
        ax.set_title(f"Order {o}: static rms {100 * rms:.2f} %, noise of the mean "
                     f"{100 * float(np.nanmedian(noise[o])):.2f} %", fontsize="medium")
    axes[-1].set_xlabel("Wavelength (nm)")
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved static residuals → %s", output_path)


def plot_stellar_history(history, output_path: str = "stellar_history.pdf"):
    """Round-by-round history of the template fit: the evidence, the prior's
    hyperparameters and the dry-species columns of every night."""
    rounds = np.array([h["round"] for h in history])
    neg2 = np.array([h["neg2_lnz"] for h in history])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    axes[0].plot(rounds, neg2 - neg2.min(), "o-", color="C0")
    axes[0].set(xlabel="Star step", ylabel="−2 ln Z − min", title="Evidence")

    axes[1].plot(rounds, [np.median(h["ell_kms"]) for h in history], "o-", color="C1")
    axes[1].set(xlabel="Star step", ylabel="median ℓ (km/s)", title="Prior hyperparameters")
    ax2 = axes[1].twinx()
    ax2.plot(rounds, [np.median(h["sigma"]) for h in history], "s--", color="C2")
    ax2.set_ylabel("median prior σ", color="C2")

    for i, _ in enumerate(history[0]["dry"]):
        for j, species in enumerate(history[0]["dry"][i]):
            vals = [10.0 ** np.mean(h["dry"][i][species]) for h in history]
            axes[2].plot(rounds, vals, "o-", color=f"C{j % 10}",
                         ls="-" if i == 0 else ":", label=f"{species} (night {i})")
    axes[2].set(xlabel="Star step", ylabel=r"Dry column $10^{\rm dex}$",
                title="Dry-species columns")
    axes[2].legend(fontsize="x-small", ncol=2)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved stellar-template history → %s", output_path)


def plot_stellar_diagnostics(result, output_dir: str) -> None:
    """Write the learned-template diagnostics to ``output_dir`` and save the
    template itself as ``stellar_template.npz``.

    ``result`` is a :class:`~enuma.inference.fit.FitResult` with ``stellar`` set.
    """
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_stellar_template(result, output_path=str(out / "stellar_template.pdf"))
    plot_static_residuals(result, output_path=str(out / "static_residuals.pdf"))
    if result.stellar_history:
        plot_stellar_history(result.stellar_history, output_path=str(out / "stellar_history.pdf"))
    result.stellar.save(str(out / "stellar_template.npz"))
    logger.info("Saved learned-template diagnostics → %s", out)
