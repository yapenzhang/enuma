import logging
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.handlers as handlers

from enuma.model.setup import build_context, load_opacities
from enuma.model.forward import forward_model, forward_model_batched
from enuma.config import FitConfig
from enuma.io.results import save_best_fit_params_json, save_best_fit_spectrum_txt
from enuma.inference.sites import (
    draw_sites,
    dry_species_of,
    iter_sites,
    partition_sites,
    per_exposure_sites,
    resolve_fit_species,
)
from enuma.inference.optimizer import run_svi
from enuma.state import (
    Layout,
    ModelConfig,
    ModelContext,
    ModelParameters,
    params_to_sites,
    sites_to_params,
)

__all__ = [
    "FitResult",
    "telluric_model",
    "fit_spectrum",
    "fit_timeseries",
]

logger = logging.getLogger("enuma.inference.fit")


class FitResult(NamedTuple):
    """Everything produced by :func:`fit_spectrum` or :func:`fit_timeseries`.
    """
    params: ModelParameters
    sites: Dict[str, jnp.ndarray]   # optimised free + pinned site values
    losses: List[float]
    data: dict                      # the original loader dict
    obs_mask: jnp.ndarray           # boolean validity mask (after saturation cull)
    context: ModelContext
    config: ModelConfig
    fit_config: "FitConfig"
    dry_species: Tuple[str, ...]
    layout: Layout = Layout()
    # (recorded_steps, {site_name: trace}) from run_svi; None when plots are
    # regenerated from a saved best-fit (no SVI history available).
    param_history: Optional[tuple] = None

    def plot(self, output_dir: str = ".", *,
             comparison_fits: Optional[str] = None) -> None:
        """Write the standard diagnostic PDF set into ``output_dir``.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if self.layout.n_exp:
            from enuma.plotting import plot_timeseries_diagnostics
            plot_timeseries_diagnostics(self, str(out))
            return

        from enuma.plotting import (
            plot_fit_results,
            plot_instrument_diagnostics,
            plot_loss_history,
            plot_param_convergence,
            plot_residuals_diagnostic,
        )
        obs_flux = jnp.asarray(self.data["flux"])
        plot_fit_results(
            self.params, self.context, self.config, obs_flux, self.obs_mask,
            output_path=str(out / "fit_results.pdf"),
            fit_species=self.fit_config.fit_species,
            comparison_fits=comparison_fits,
        )
        plot_instrument_diagnostics(
            self.params, self.context, self.config,
            output_path=str(out / "fit_instrument.pdf"),
        )
        plot_residuals_diagnostic(
            self.params, self.context, self.config, obs_flux, self.obs_mask,
            output_path=str(out / "fit_residuals_diagnostic.pdf"),
            comparison_fits=comparison_fits,
        )
        if self.losses:
            plot_loss_history(self.losses, output_path=str(out / "fit_loss_history.pdf"))
        if self.param_history is not None:
            plot_param_convergence(self.param_history,
                                   output_path=str(out / "fit_param_convergence.pdf"))


def _load_observation(data: dict, fit_config: FitConfig):
    """Load the observation arrays and put them on the JAX device."""
    device = jax.devices()[0]
    put = lambda a: jax.device_put(jnp.asarray(a), device)
    obs_flux, obs_err = put(data["flux"]), put(data["err"])
    sat_mask = data["flux"] >= fit_config.saturation_mask_threshold
    obs_mask = put(np.asarray(data["mask"]) & sat_mask)
    return obs_flux, obs_err, obs_mask


def telluric_model(*, context: ModelContext, config: ModelConfig,
                   obs_flux: jnp.ndarray, obs_err: jnp.ndarray, obs_mask: jnp.ndarray,
                   dry_species: Tuple[str, ...], layout: Layout,
                   fixed_values: Dict[str, jnp.ndarray]):
    """NumPyro model"""
    fixed_values = fixed_values or {}
    sites = draw_sites(layout, context, config, dry_species, fixed_values)
    params = sites_to_params(sites, dry_species)

    # Continuum smoothness: soft prior on adjacent-node differences. 
    # cont = params.continuum_coeffs
    # numpyro.sample(
    #     "cont_smoothness",
    #     dist.Normal(0.0, 0.05).expand(list(cont.shape[:-1]) + [cont.shape[-1] - 1]),
    #     obs=jnp.diff(cont, axis=-1),
    # )

    model_flux = (forward_model(params, context, config) if layout.n_exp == 0
                  else forward_model_batched(params, context, config, layout))
    scale = obs_err if obs_err is not None else 0.01
    with handlers.mask(mask=obs_mask if obs_mask is not None else True):
        numpyro.sample("obs", dist.Normal(model_flux, scale), obs=obs_flux)


def _run_svi(model_kwargs: dict, init_values: Dict[str, jnp.ndarray],
             fit_config: FitConfig, *, print_freq: int, rng_seed: int):
    """``run_svi`` with the optimiser + convergence knobs taken from ``fit_config``."""
    return run_svi(
        telluric_model, model_kwargs, init_values=init_values,
        max_steps=fit_config.max_steps, learning_rate=fit_config.learning_rate,
        convergence_ftol=fit_config.convergence_ftol,
        convergence_patience=fit_config.convergence_patience,
        convergence_check_every=fit_config.convergence_check_every,
        convergence_min_steps=fit_config.convergence_min_steps,
        print_freq=print_freq, rng_seed=rng_seed,
    )


_PER_ORDER_SITES = ("resolution_coeffs", "wave_coeffs", "continuum_coeffs")


def _take_orders(data: dict, order_idx: np.ndarray) -> dict:
    """Shallow copy of ``data`` with the per-order arrays (wave/flux/err/mask) sliced
    to ``order_idx`` along the order axis (-2); other entries pass through."""
    out = dict(data)
    for key in ("wave", "flux", "err", "mask"):
        if key in out:
            out[key] = np.take(np.asarray(out[key]), order_idx, axis=-2)
    return out


def _set_orders(base: jnp.ndarray, order_idx: np.ndarray, rows: jnp.ndarray) -> jnp.ndarray:
    """``base`` with its order-axis (-2) entries at ``order_idx`` replaced by ``rows``."""
    sl = (slice(None),) * (base.ndim - 2) + (jnp.asarray(order_idx),)
    return base.at[sl].set(rows)


def _slice_observation_orders(data: dict, fit_windows: Tuple[Tuple[float, float], ...]):
    """``(order_idx, fit_data)`` for the loaded orders overlapping any ``fit_windows``
    span, with ``fit_data`` sliced down to those orders."""
    wave = np.asarray(data["wave"])              # (O, P); order axis = -2
    o_min, o_max = wave.min(axis=-1), wave.max(axis=-1)   # (O,)
    keep = np.zeros(wave.shape[-2], dtype=bool)
    for a, b in fit_windows:
        keep |= (o_max >= a) & (o_min <= b)
    order_idx = np.nonzero(keep)[0]
    if order_idx.size == 0:
        raise ValueError(f"fit_windows {fit_windows} overlap none of the loaded orders "
                         f"(spanning {float(wave.min()):.1f}-{float(wave.max()):.1f} nm).")
    return order_idx, _take_orders(data, order_idx)


def _scatter_to_full(sub_params: ModelParameters, full_context: ModelContext,
                     full_config: ModelConfig, fit_config: FitConfig,
                     dry_species: Tuple[str, ...], layout: Layout,
                     order_idx: np.ndarray) -> ModelParameters:
    """Lift the windowed stage-1 params onto the full order set (the stage-2 seed):
    global (atmosphere) sites pass through; per-order sites take their full-order
    default with the fitted ``order_idx`` rows spliced in."""
    full_fixed, full_free = partition_sites(full_context, full_config, fit_config,
                                            dry_species, layout)
    full_default = {**full_fixed, **full_free}
    sub_sites = params_to_sites(sub_params)
    out: Dict[str, jnp.ndarray] = {}
    for name, default_val in full_default.items():
        out[name] = (_set_orders(default_val, order_idx, sub_sites[name])
                     if name in _PER_ORDER_SITES else sub_sites[name])
    return sites_to_params(out, dry_species)


def _refit_nuisance(seeded_params: ModelParameters, data: dict, rest_idx: np.ndarray,
                    fit_config: FitConfig, dry_species: Tuple[str, ...], layout: Layout,
                    *, print_freq: int, rng_seed: int, opacities=None) -> ModelParameters:
    """Stage 2: fit only the per-order continuum/LSF/wave for the ``rest_idx`` orders
    (the ones stage 1 skipped), holding the seeded atmosphere frozen, and splice the
    fitted rows back into ``seeded_params``."""
    if rest_idx.size == 0:
        return seeded_params

    rest_data = _take_orders(data, rest_idx)
    obs_flux, obs_err, obs_mask = _load_observation(rest_data, fit_config)
    rest_ctx, rest_cfg = build_context(rest_data, fit_config, layout, opacities=opacities)

    seeded_sites = params_to_sites(seeded_params)
    rest_fixed, rest_free = partition_sites(rest_ctx, rest_cfg, fit_config, dry_species, layout)
    fixed_values, init_values = {}, {}
    for site in iter_sites(dry_species):
        name = site.name
        if name not in _PER_ORDER_SITES:
            fixed_values[name] = seeded_sites[name]        # frozen stage-1 atmosphere
        elif site.is_free(fit_config):
            init_values[name] = rest_free[name]            # nuisance to fit
        else:
            fixed_values[name] = rest_fixed[name]          # pinned nuisance
    if not init_values:
        return seeded_params

    opt_sites, _, _ = _run_svi(
        dict(context=rest_ctx, config=rest_cfg, obs_flux=obs_flux, obs_err=obs_err,
             obs_mask=obs_mask, dry_species=dry_species, layout=layout,
             fixed_values=fixed_values),
        init_values, fit_config, print_freq=print_freq, rng_seed=rng_seed)

    for name in _PER_ORDER_SITES:
        if name in opt_sites:
            seeded_sites[name] = _set_orders(seeded_sites[name], rest_idx, opt_sites[name])
    return sites_to_params(seeded_sites, dry_species)


def _fit_full(data: dict, fit_config: FitConfig, n_exp: int, *,
              print_freq: int, rng_seed: int, opacities=None) -> FitResult:
    """Single-pass fit core: load → build context → partition sites → SVI → reassemble."""
    obs_flux, obs_err, obs_mask = _load_observation(data, fit_config)
    context, config = build_context(data, fit_config, Layout(n_exp=n_exp), opacities=opacities)
    dry_species = dry_species_of(context, tuple(fit_config.species))
    per_exp = per_exposure_sites(fit_config, dry_species) if n_exp else frozenset()
    layout = Layout(n_exp=n_exp, per_exposure=per_exp)
    fixed_values, init_values = partition_sites(context, config, fit_config, dry_species, layout)

    opt_sites, losses, param_history = _run_svi(
        dict(context=context, config=config, obs_flux=obs_flux, obs_err=obs_err,
             obs_mask=obs_mask, dry_species=dry_species, layout=layout,
             fixed_values=fixed_values),
        init_values, fit_config, print_freq=print_freq, rng_seed=rng_seed)

    all_sites = {**opt_sites, **fixed_values}
    return FitResult(
        params=sites_to_params(all_sites, dry_species), sites=all_sites, losses=losses,
        data=data, obs_mask=obs_mask, context=context, config=config,
        fit_config=fit_config, dry_species=dry_species, layout=layout,
        param_history=param_history,
    )


def _fit_windowed(data: dict, fit_config: FitConfig, n_exp: int, *,
                  print_freq: int, rng_seed: int) -> FitResult:
    """Two-stage windowed fit. Stage 1 fits the full model on the orders overlapping
    ``fit_config.fit_windows``; stage 2 freezes that atmosphere and fits the per-order
    continuum/LSF/wave for the remaining orders, so the result spans every order."""
    order_idx, fit_data = _slice_observation_orders(data, fit_config.fit_windows)
    n_total = int(np.asarray(data["wave"]).shape[-2])
    if order_idx.size >= n_total:   # windows already cover every order
        return _fit_full(data, fit_config, n_exp, print_freq=print_freq, rng_seed=rng_seed)

    opacities = load_opacities(fit_config, np.asarray(data["wave"], dtype=np.float64))

    logger.info("Stage 1: fit windows %s, %d/%d orders.",
                list(fit_config.fit_windows), order_idx.size, n_total)
    sub = _fit_full(fit_data, fit_config, n_exp, print_freq=print_freq, rng_seed=rng_seed,
                    opacities=opacities)

    _, _, full_mask = _load_observation(data, fit_config)
    full_context, full_config = build_context(data, fit_config, Layout(n_exp=n_exp),
                                              opacities=opacities)
    dry_species = dry_species_of(full_context, tuple(fit_config.species))
    seeded = _scatter_to_full(sub.params, full_context, full_config, fit_config,
                              dry_species, sub.layout, order_idx)

    rest_idx = np.setdiff1d(np.arange(n_total), order_idx)
    logger.info("Stage 2: fit continuum/LSF/wave for the remaining %d orders, "
                "atmosphere frozen.", rest_idx.size)
    full_params = _refit_nuisance(seeded, data, rest_idx, fit_config, dry_species,
                                  sub.layout, print_freq=print_freq, rng_seed=rng_seed,
                                  opacities=opacities)

    return FitResult(
        params=full_params, sites=params_to_sites(full_params), losses=sub.losses,
        data=data, obs_mask=full_mask, context=full_context, config=full_config,
        fit_config=fit_config, dry_species=dry_species, layout=sub.layout,
        param_history=sub.param_history,
    )


def _run_fit(data: dict, fit_config: FitConfig, n_exp: int, *,
             print_freq: int, rng_seed: int) -> FitResult:
    if fit_config.fit_windows:
        return _fit_windowed(data, fit_config, n_exp,
                                 print_freq=print_freq, rng_seed=rng_seed)
    return _fit_full(data, fit_config, n_exp, print_freq=print_freq, rng_seed=rng_seed)


def fit_spectrum(
    data: dict,
    fit_config: FitConfig = FitConfig(),
    *,
    print_freq: int = 50,
    rng_seed: int = 42,
    save_params_json_path: str = "best_fit.json",
    save_spectrum_txt_path: str = "best_fit.txt",
) -> FitResult:
    """Fit a telluric model to a single multi-order exposure.

    Args:
        data: dict from ``load_fits_spectra`` (wave, flux, err, mask, airmass).
        fit_config: controls species, what to fit, polynomial degrees, GP
            priors, and optimiser settings.
        print_freq: progress-bar update cadence (cosmetic).
        rng_seed: PRNG seed for SVI initialisation.
        save_params_json_path / save_spectrum_txt_path: optional output paths.

    Returns:
        A :class:`FitResult`; call ``result.plot(output_dir)`` to write the PDFs.
    """
    logger.info(
        "species=%s, fit_species=%s, steps=%d, backend=%s",
        list(fit_config.species), list(resolve_fit_species(fit_config)),
        fit_config.max_steps, jax.default_backend().upper()
    )
    result = _run_fit(data, fit_config, 0, print_freq=print_freq, rng_seed=rng_seed)
    
    save_best_fit_params_json(result.params, save_params_json_path, species=fit_config.species)
    save_best_fit_spectrum_txt(result.params, result.context, result.config,
                                   save_spectrum_txt_path)
    return result


def fit_timeseries(data: dict, fit_config: FitConfig = FitConfig(), *,
                   print_freq: int = 10, rng_seed: int = 42) -> FitResult:
    """Jointly fit one night of multi-exposure spectra."""
    n_exp = int(np.asarray(data["flux"]).shape[0])
    _pe = lambda flag: "per-exposure" if flag else "shared"
    logger.info(
        "%d exposures, airmass %.2f–%.2f, per_exposure_dry=%s, per_exposure_temperature=%s, per_exposure_h2o=%s, stellar=%s",
        n_exp, float(np.min(data["airmass"])), float(np.max(data["airmass"])),
        _pe(fit_config.dry_vmr_per_exposure), _pe(fit_config.temperature_per_exposure),
        _pe(fit_config.h2o_per_exposure), fit_config.stellar_enabled,
    )

    result = _run_fit(data, fit_config, n_exp, print_freq=print_freq, rng_seed=rng_seed)

    #TODO: save best-fit params and models

    return result