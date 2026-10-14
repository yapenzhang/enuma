"""Learn the stellar spectrum together with the tellurics, from one or several nights.

The loop of ``docs/stellar.rst``:

0. Fit every night with the fixed NewEra template (the existing fit). This gives
   the tellurics θ and the stellar RV, which then defines the template's rest frame.
1. **Star step** (θ fixed): read ``A_n`` off the forward model, refit the prior's
   hyperparameters by the evidence, and solve for the template and its error bars
   (:mod:`enuma.inference.stellar_gp`). All nights share one template.
2. **Telluric step** (template fixed): refit θ night by night with the existing SVI,
   warm-started from the previous round.
3. Repeat; finish with a star step at the final θ.

The anchor of the template prior (NewEra, flat, or a file) stays fixed in every
round; only the template, θ and the hyperparameters change.
"""

import dataclasses
import logging
import time
from typing import List, NamedTuple, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

from enuma.config import FitConfig
from enuma.inference import stellar_gp as sg
from enuma.inference.fit import FitResult, _fit_full, _optimize, _prepare
from enuma.inference.sites import partition_sites
from enuma.model.forward import bounded, legendre_polyval
from enuma.model.opacity import get_opacities
from enuma.model.setup import _opacity_wave_range, _slice_opacities
from enuma.state import sites_to_params

__all__ = [
    "MultiNightResult",
    "fit_nights",
]

logger = logging.getLogger("enuma.inference.stellar_em")

# Below this |star − observatory| velocity, stellar and telluric lines of the same
# molecule overlap within ~1 LSF width.
_LOW_VELOCITY_KMS = 3.0


class MultiNightResult(NamedTuple):
    """Result of :func:`fit_nights`."""
    #: Final fit of every night: the learned template is in ``context`` (so
    #: ``forward_model_batched`` reproduces the model) and in ``stellar``.
    nights: Tuple[FitResult, ...]
    #: The learned template with error bars and hyperparameters.
    stellar: sg.StellarTemplate
    #: One dict per star step (round), see :func:`fit_nights`.
    history: List[dict]
    #: The fixed-template fits that seeded the loop, one per night.
    initial: Tuple[FitResult, ...]


class _Night(NamedTuple):
    """One night prepared for the loop (template mode)."""
    data: dict
    prep: object                 # fit._Prepared with the template-mode context/config
    free: Tuple[str, ...]        # free site names
    obs_flux: jnp.ndarray        # (N, O, P), finite
    obs_w: jnp.ndarray           # (N, O, P) inverse variances, 0 where masked
    model_wave: np.ndarray       # float64 model grid
    geom: sg.NightGeometry
    spec: sg.ProbeSpec


def _shared_opacities(datasets, fit_config: FitConfig):
    """Load the opacity grids once over the union of the nights' ranges."""
    lo, hi = zip(*(_opacity_wave_range(np.asarray(d["wave"], np.float64)) for d in datasets))
    return get_opacities(list(fit_config.species), (min(lo), max(hi)),
                         opacity_dir=fit_config.opacity_dir)


def _initial_fit(data: dict, fit_config: FitConfig, opacities, *, print_freq, rng_seed):
    """The fixed-template seed fit: NewEra when its parameters are given, else
    tellurics only."""
    fc = fit_config
    use_newera = fc.stellar_teff is not None and fc.stellar_logg is not None
    fc1 = dataclasses.replace(fc, stellar_model="newera", stellar_enabled=use_newera)
    logger.info("Initial fit with the %s.", "NewEra template" if use_newera else
                "tellurics only (no stellar_teff/logg)")
    n_exp = int(np.asarray(data["flux"]).shape[0])
    return _fit_full(data, fc1, n_exp, print_freq=print_freq, rng_seed=rng_seed,
                     opacities=opacities)


def _observation_arrays(prep):
    """Finite flux and inverse-variance weights (0 where masked or invalid)."""
    flux, err, mask = prep.obs_flux, prep.obs_err, prep.obs_mask
    ok = mask & jnp.isfinite(flux) & jnp.isfinite(err) & (err > 0)
    w = jnp.where(ok, 1.0 / jnp.where(ok, err, 1.0) ** 2, 0.0).astype(jnp.float32)
    return jnp.where(ok, flux, 0.0).astype(jnp.float32), w


def _seed_sites(free_init: dict, fixed: dict, initial: FitResult) -> dict:
    """Template-mode sites: the free ones from the seed fit where shapes agree."""
    sites, fresh = dict(fixed), []
    for name, default in free_init.items():
        seed = initial.sites.get(name)
        ok = seed is not None and np.shape(seed) == np.shape(default)
        sites[name] = jnp.asarray(seed) if ok else default
        if not ok:
            fresh.append(name)
    if fresh:
        logger.info("Seed fit has no matching value for %s; starting from the defaults.", fresh)
    return sites


def _max_wave_shift_pix(sites: dict, ctx, config) -> float:
    """Largest wavelength-solution shift (pixels) in a set of sites, with the
    coefficients bounded as the model uses them."""
    coeffs = np.asarray(bounded(jnp.asarray(sites["wave_coeffs"]), 0.0,
                                jnp.asarray(config.wave_prior_pix)))
    # coeffs: (O, k), or (N, O, k) per exposure
    x = np.asarray(ctx.obs_wave_norm)                             # (O, P)
    coeffs = coeffs.reshape((-1,) + coeffs.shape[-2:])
    shifts = [legendre_polyval(c, x[o]) for exp in coeffs for o, c in enumerate(exp)]
    return float(np.max(np.abs(shifts)))


def _warn_degeneracies(fit_config: FitConfig, velocities: Sequence[np.ndarray]) -> None:
    if fit_config.dry_vmr_per_exposure or fit_config.temperature_per_exposure:
        logger.warning(
            "stellar_model='gp' with dry_vmr_per_exposure=%s / temperature_per_exposure=%s: "
            "anything fitted per exposure can trade exactly with the template within a night "
            "(see 'Learning the stellar spectrum' in the docs). Sharing them over the night "
            "is recommended.",
            fit_config.dry_vmr_per_exposure, fit_config.temperature_per_exposure)
    v_min = min(float(np.min(np.abs(v))) for v in velocities)
    if v_min < _LOW_VELOCITY_KMS:
        logger.warning("The star moves only %.1f km/s relative to the observatory; stellar "
                       "and telluric lines of the same molecules overlap, which weakens their "
                       "separation.", v_min)


def fit_nights(datasets: Sequence[dict], fit_config: FitConfig = FitConfig(), *,
               initial: Optional[Sequence[FitResult]] = None,
               print_freq: int = 10, rng_seed: int = 42) -> MultiNightResult:
    """Learn the stellar spectrum and the tellurics from one or several nights.

    All nights must observe the same star with the same instrument setting. Each
    night keeps its own telluric and instrument parameters; the stellar spectrum
    is one template, in the star's rest frame, shared by every night.

    Args:
        datasets: night dicts, as for :func:`~enuma.inference.fit.fit_timeseries`.
        fit_config: the fit configuration; the ``stellar_*`` knobs of the learned
            template (``stellar_reference``, ``stellar_node_kms``,
            ``stellar_gp_length_kms``, ``stellar_gp_sigma``,
            ``stellar_em_iterations``, ``stellar_mstep_steps``, ``stellar_em_tol``)
            apply. ``stellar_model`` itself is not consulted.
        initial: optional seed fits, one per night (e.g. earlier
            ``fit_timeseries`` results with the NewEra template). Without them every
            night is first fitted with the fixed template.
        print_freq: progress-bar update cadence (cosmetic).
        rng_seed: PRNG seed for the SVI runs.

    Returns:
        A :class:`MultiNightResult`. Each entry of ``history`` describes one star
        step: ``round``, ``neg2_lnz`` (−2 ln evidence without the data-only
        constant), ``chi2``, ``penalty``, ``logdet_post``, ``logdet_prior``, the
        hyperparameters (``sigma``, ``scale``, ``alpha`` and ``ell_kms``, per order),
        the dry-species columns per night (``dry``), and, after the
        telluric step, ``mstep_loss`` (final SVI loss per night).
    """
    fc = fit_config
    if fc.fit_windows:
        raise NotImplementedError("fit_windows is not supported with the learned template.")
    datasets = list(datasets)
    if initial is not None and len(initial) != len(datasets):
        raise ValueError("Pass one initial fit per night.")

    opacities = _shared_opacities(datasets, fc)

    # 0. Seed fits (fixed template) → tellurics and the stellar rest frame.
    if initial is None:
        initial = [_initial_fit(d, fc, opacities, print_freq=print_freq, rng_seed=rng_seed)
                   for d in datasets]
    initial = tuple(initial)
    seeded_star = [r.config.stellar_enabled for r in initial]
    if all(seeded_star) and fc.fit_rv:
        rvs = [float(np.asarray(r.sites["rv_systemic"])) for r in initial]
        rv = float(np.median(rvs))
        if np.ptp(rvs) > 0.5:
            logger.warning("Seed fits disagree on the systemic RV (%s km/s); using %.2f.",
                           ", ".join(f"{x:.2f}" for x in rvs), rv)
    else:
        rv = float(fc.rv_kms)
    vsini0 = (float(np.median([np.asarray(r.sites["vsini"]) for r in initial]))
              if all(seeded_star) else float(fc.vsini))

    # Template mode: no NewEra in the context, RV pinned, rotation off (the template
    # is already the broadened spectrum).
    fc_gp = dataclasses.replace(fc, stellar_enabled=False)
    fc_m = dataclasses.replace(fc_gp, max_steps=fc.stellar_mstep_steps)
    nights, sites, velocities = [], [], []
    for data, init_fit in zip(datasets, initial):
        n_exp = int(np.asarray(data["flux"]).shape[0])
        prep = _prepare(data, fc_gp, n_exp, opacities=opacities)
        config = prep.config._replace(stellar_enabled=True, stellar_rot_half_width=0,
                                      stellar_rv_init=rv, stellar_vsini_init=0.0)
        ctx = prep.context._replace(stellar_dv_kms=prep.context.model_dv_kms)
        fixed, free_init = partition_sites(ctx, config, fc_gp, prep.dry_species, prep.layout)
        prep = prep._replace(context=ctx, config=config, fixed_values=fixed)
        flux, w = _observation_arrays(prep)
        model_wave = _slice_opacities(opacities,
                                      _opacity_wave_range(np.asarray(data["wave"], np.float64)))[1]
        nights.append(_Night(data, prep, tuple(free_init), flux, w, model_wave, None, None))
        sites.append(_seed_sites(free_init, fixed, init_fit))
        velocities.append(rv + np.asarray(ctx.baryrv, np.float64))
    _warn_degeneracies(fc, velocities)

    # Node ranges absorb the seed's wavelength shifts with room for the refits.
    seed_shift = max(_max_wave_shift_pix(s, n.prep.context, n.prep.config)
                     for s, n in zip(sites, nights))
    margin_pix = max(16.0, 1.5 * seed_shift + 8.0)
    grid = sg.build_stellar_grid([n.model_wave for n in nights],
                                 [np.asarray(n.data["wave"], np.float64) for n in nights],
                                 velocities, node_kms=fc.stellar_node_kms,
                                 lsf_kernel_width=nights[0].prep.config.lsf_kernel_width,
                                 wave_shift_pix=margin_pix)
    nights = [n._replace(
        geom=sg.night_geometry(grid, n.model_wave, np.asarray(n.prep.context.order_start_idx),
                               n.prep.config.order_size),
        spec=sg.probe_spec(grid, np.asarray(n.data["wave"], np.float64), n.model_wave.size))
        for n in nights]

    beta_ref, envelope = sg.reference_on_nodes(grid, fc, vsini_kms=vsini0)
    prior = sg.make_prior(grid, beta_ref, envelope)
    hyper = sg.initial_hyper(grid.node_lo.size, fc)
    beta0 = sg.anchor(prior, hyper)

    history: List[dict] = []
    results: List[Optional[FitResult]] = [None] * len(nights)
    losses: List[List[float]] = [[] for _ in nights]
    n_rounds = fc.stellar_em_iterations
    r = 0
    while True:
        final = r >= n_rounds
        # 1. Star step at the current tellurics. The continuum is refitted first (it is
        #    linear given the template), so smooth mismatch never reaches the template.
        t0 = time.time()
        G = b = chi2_0 = None
        for i, night in enumerate(nights):
            ctx_t = night.prep.context._replace(
                stellar_template=sg.nodes_to_grid(beta0, night.geom, night.spec.n_model))
            sites[i]["continuum_coeffs"] = sg.refit_continuum(
                sites_to_params(sites[i], night.prep.dry_species), ctx_t, night.prep.config,
                night.prep.layout, night.obs_flux, night.obs_w)
            s = sites[i]
            params = sites_to_params(s, night.prep.dry_species)
            Gi, bi, ci, overflow = sg.normal_equations(
                params, night.prep.context, night.prep.config, night.prep.layout,
                night.obs_flux, night.obs_w, beta0, night.geom, night.spec)
            if bool(overflow):
                shift = _max_wave_shift_pix(s, night.prep.context, night.prep.config)
                raise RuntimeError(
                    "A pixel's template window fell outside its node row: the fitted "
                    f"wavelength-solution shifts (up to {shift:.1f} pixels) outgrew the "
                    f"template's node margins ({margin_pix:.0f} pixels).")
            G, b, chi2_0 = ((Gi, bi, ci) if G is None else (G + Gi, b + bi, chi2_0 + ci))
        t1 = time.time()
        # The hyperparameters follow the tellurics: refit them in every star step.
        hyper, _ = sg.fit_hyperparameters(G, b, beta0, prior, hyper)
        post = sg.posterior(G, b, chi2_0, beta0, prior, hyper, with_variance=final)
        post.neg2_lnz.block_until_ready()
        t2 = time.time()
        hv = {k: np.asarray(v) for k, v in sg.hyper_values(hyper).items()}
        history.append(dict(
            round=r, neg2_lnz=float(post.neg2_lnz), chi2=float(post.chi2),
            penalty=float(post.penalty), logdet_post=float(post.logdet_post),
            logdet_prior=float(post.logdet_prior), sigma=hv["sigma"], scale=hv["scale"],
            alpha=hv["alpha"], ell_kms=hv["ell_kms"],
            dry=[{k[len("dry_vmr_"):]: np.asarray(v).ravel().tolist()
                  for k, v in s.items() if k.startswith("dry_vmr_")} for s in sites]))
        logger.info("Star step %d: -2lnZ = %.1f, chi2 = %.1f; median over orders: "
                    "ell = %.2f km/s, sigma = %.3f, scale = %.2f (normal equations %.0f s, "
                    "evidence and solve %.0f s).", r,
                    history[-1]["neg2_lnz"], history[-1]["chi2"],
                    float(np.median(hv["ell_kms"])), float(np.median(hv["sigma"])),
                    float(np.median(hv["scale"])), t1 - t0, t2 - t1)
        beta0 = post.mean
        if final:
            break
        if fc.stellar_em_tol > 0 and r > 0:
            prev = history[-2]["neg2_lnz"]
            if abs(prev - history[-1]["neg2_lnz"]) <= fc.stellar_em_tol * abs(prev):
                n_rounds = r + 1          # converged: one more telluric step, then finish

        # 2. Telluric step with the template fixed.
        mstep_loss = []
        for i, night in enumerate(nights):
            template = sg.nodes_to_grid(beta0, night.geom, night.spec.n_model)
            prep = night.prep._replace(
                context=night.prep.context._replace(stellar_template=template),
                init_values={k: sites[i][k] for k in night.free})
            res = _optimize(prep, night.data, fc_m, print_freq=print_freq, rng_seed=rng_seed)
            results[i], sites[i] = res, dict(res.sites)
            losses[i].extend(res.losses)
            mstep_loss.append(float(res.losses[-1]))
        history[-1]["mstep_loss"] = mstep_loss
        r += 1

    template = sg.stellar_template(grid, prior, hyper, post, rv)
    out = []
    for i, night in enumerate(nights):
        # The final template goes with the final sites (including the continuum
        # refitted for the last star step).
        ctx = night.prep.context._replace(
            stellar_template=sg.nodes_to_grid(beta0, night.geom, night.spec.n_model))
        last = results[i]
        out.append(FitResult(
            params=sites_to_params(sites[i], night.prep.dry_species), sites=sites[i],
            losses=losses[i], data=night.data, obs_mask=night.prep.obs_mask, context=ctx,
            config=night.prep.config, fit_config=fc, dry_species=night.prep.dry_species,
            layout=night.prep.layout,
            param_history=None if last is None else last.param_history,
            stellar=template, stellar_history=history))
    return MultiNightResult(nights=tuple(out), stellar=template, history=history,
                            initial=initial)
