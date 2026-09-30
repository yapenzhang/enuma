from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from enuma.config import FitConfig
from enuma.state import (
    Layout,
    ModelConfig,
    ModelContext,
    ModelParameters,
    sites_to_params,
)

__all__ = [
    "Param",
    "resolve_fit_species",
    "dry_species_of",
    "iter_sites",
    "per_exposure_sites",
    "default_params",
    "partition_sites",
    "draw_sites",
]


@dataclass(frozen=True)
class Param:
    name: str
    # ``init``/``prior`` are lazy in ``(context, config, layout)``: ``layout``
    # prepends the per-exposure ``(N,)`` axis for the sites that carry one.
    init: Callable[[ModelContext, ModelConfig, Layout], jnp.ndarray]      # default guess
    prior: Callable[[ModelContext, ModelConfig, Layout], dist.Distribution]
    is_free: Callable[[FitConfig], bool]                                 # FitConfig toggle


def resolve_fit_species(fit_config: FitConfig) -> Tuple[str, ...]:
    """Species whose columns are free (``fit_species`` or, if None, all of ``species``)."""
    if fit_config.fit_species is None:
        return tuple(fit_config.species)
    return tuple(fit_config.fit_species)


def _resolution_loc(config):
    """Init/prior loc for the *log-space* resolution polynomial. The constant term is 
    ``log(resolution_init)`` and the λ-dependent terms are 0."""
    return jnp.zeros(config.n_resolution_coeffs).at[-1].set(float(np.log(config.resolution_init)))


_T_LATENT = Param(
    "t_latent",
    init=lambda c, cfg, L: jnp.zeros(L.lead("t_latent") + (c.temp_chol_L.shape[1],)),
    prior=lambda c, cfg, L: dist.Normal(0.0, 1.0).expand(
        list(L.lead("t_latent") + (c.temp_chol_L.shape[1],))),
    is_free=lambda f: f.fit_temperature,
)
_H2O_LATENT = Param(
    "h2o_latent",
    init=lambda c, cfg, L: jnp.zeros(L.lead("h2o_latent") + (c.h2o_chol_L.shape[1],)),
    prior=lambda c, cfg, L: dist.Normal(0.0, 1.0).expand(
        list(L.lead("h2o_latent") + (c.h2o_chol_L.shape[1],))),
    is_free=lambda f: "h2o" in resolve_fit_species(f),
)
_RESOLUTION = Param(
    "resolution_coeffs",
    init=lambda c, cfg, L: jnp.broadcast_to(
        jnp.tile(_resolution_loc(cfg), (c.obs_wave.shape[0], 1)),
        L.lead("resolution_coeffs") + (c.obs_wave.shape[0], cfg.n_resolution_coeffs)),
    prior=lambda c, cfg, L: dist.Normal(_resolution_loc(cfg), 0.05).expand(
        list(L.lead("resolution_coeffs") + (c.obs_wave.shape[0], cfg.n_resolution_coeffs))),
    is_free=lambda f: f.fit_resolution,
)
_WAVE = Param(
    "wave_coeffs",
    init=lambda c, cfg, L: jnp.zeros(L.lead("wave_coeffs") + (c.obs_wave.shape[0], cfg.n_wave_coeffs)),
    prior=lambda c, cfg, L: dist.Normal(0.0, 0.5).expand(
        list(L.lead("wave_coeffs") + (c.obs_wave.shape[0], cfg.n_wave_coeffs))),
    is_free=lambda f: f.fit_wave_solution,
)
_CONTINUUM = Param(
    "continuum_coeffs",
    init=lambda c, cfg, L: jnp.ones(
        L.lead("continuum_coeffs") + (c.obs_wave.shape[0], c.cont_bspline_matrix.shape[1])),
    prior=lambda c, cfg, L: dist.Normal(1.0, 0.1).expand(
        list(L.lead("continuum_coeffs") + (c.obs_wave.shape[0], c.cont_bspline_matrix.shape[1]))),
    is_free=lambda f: f.fit_continuum,
)

# Stellar Doppler shift + rotation. Free only when the stellar template is on
# (``stellar_enabled``); the init/prior mean come from the config (the systemic-RV
# seed and the vsini seed). They are pinned constants the forward model ignores
# when stellar is off. The free site is the *systemic* velocity; the barycentric
# term rides on ``context.baryrv`` and the forward model forms the net shift
# ``rv_systemic + baryrv`` (see :func:`enuma.model.forward.forward_model`), so a single
# exposure and a night share one systemic site.
_RV_PRIOR_SIGMA_KMS = 10.0       # rv prior width about the seed
_VSINI_PRIOR_SIGMA_KMS = 10.0    # vsini prior width about the seed
_RV_SYSTEMIC = Param(
    "rv_systemic",
    init=lambda c, cfg, L: jnp.asarray(cfg.stellar_rv_init),
    prior=lambda c, cfg, L: dist.Normal(cfg.stellar_rv_init, _RV_PRIOR_SIGMA_KMS),
    is_free=lambda f: f.stellar_enabled and f.fit_rv,
)
_VSINI = Param(
    "vsini",
    init=lambda c, cfg, L: jnp.asarray(cfg.stellar_vsini_init),
    prior=lambda c, cfg, L: dist.TruncatedNormal(
        cfg.stellar_vsini_init, _VSINI_PRIOR_SIGMA_KMS, low=0.0),
    is_free=lambda f: f.stellar_enabled and f.fit_vsini,
)


def _dry_site(species: str) -> Param:
    """A dry-species scalar log10 column-offset site (``(1,)``, or ``(N, 1)`` when
    fit per exposure)."""
    name = f"dry_vmr_{species}"
    return Param(
        name,
        init=lambda c, cfg, L: jnp.zeros(L.lead(name) + (1,)),
        prior=lambda c, cfg, L: dist.Normal(0.0, 0.3).expand(list(L.lead(name) + (1,))),
        is_free=lambda f, s=species: s in resolve_fit_species(f),
    )


def dry_species_of(context: ModelContext,
                   fit_vmr_species: Optional[Tuple[str, ...]]) -> Tuple[str, ...]:
    """Dry species (H2O excluded) to give free columns to."""
    species = fit_vmr_species if fit_vmr_species is not None else tuple(context.vmr_ref_centers)
    return tuple(s for s in species if s != "h2o")


def iter_sites(dry_species: Tuple[str, ...]) -> Tuple[Param, ...]:
    """The full ordered site spec, with the dry-column family spliced in."""
    return (_T_LATENT, _H2O_LATENT, *(_dry_site(s) for s in dry_species),
            _RESOLUTION, _WAVE, _CONTINUUM, _RV_SYSTEMIC, _VSINI)


def per_exposure_sites(fit_config: FitConfig, dry_species: Tuple[str, ...]) -> frozenset:
    """Site names that carry a leading per-exposure ``(N,)`` axis in a night fit.

    The continuum is always per exposure; the temperature/H2O profiles, the
    dry-species columns, and the instrument LSF / wavelength solution are per
    exposure or shared per the ``FitConfig`` knobs (``temperature_per_exposure`` /
    ``h2o_per_exposure`` / ``dry_vmr_per_exposure`` / ``resolution_per_exposure`` /
    ``wave_per_exposure``). A single-exposure fit never calls this — it uses the
    default empty ``Layout``.
    """
    names = {"continuum_coeffs"}
    if fit_config.temperature_per_exposure:
        names.add("t_latent")
    if fit_config.h2o_per_exposure:
        names.add("h2o_latent")
    if fit_config.dry_vmr_per_exposure:
        names |= {f"dry_vmr_{s}" for s in dry_species}
    if fit_config.resolution_per_exposure:
        names.add("resolution_coeffs")
    if fit_config.wave_per_exposure:
        names.add("wave_coeffs")
    return frozenset(names)


def default_params(context: ModelContext, config: ModelConfig,
                   dry_species: Tuple[str, ...]) -> ModelParameters:
    """The initial-guess parameters (each site at its default init value)."""
    sites = {s.name: s.init(context, config, Layout()) for s in iter_sites(dry_species)}
    return sites_to_params(sites, dry_species)


def partition_sites(
    context: ModelContext, config: ModelConfig, fit_config: FitConfig,
    dry_species: Tuple[str, ...], layout: Layout = Layout(),
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """Split the registry sites into (fixed constants, free-site initialisers).

    Each site's init value comes from ``Param.init(context, config, layout)`` and
    the free/fixed decision from ``Param.is_free`` — so the policy lives in one
    place (:func:`iter_sites`) and the same partitioner serves the single-exposure
    (default ``Layout()``) and time-series (batched) paths.
    """
    fixed, free_init = {}, {}
    for site in iter_sites(dry_species):
        target = free_init if site.is_free(fit_config) else fixed
        target[site.name] = site.init(context, config, layout)
    return fixed, free_init


def draw_sites(layout: Layout, context: ModelContext, config: ModelConfig,
               dry_species: Tuple[str, ...],
               fixed_values: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """Sample (or pin) every registry site for one ``layout``.

    A site in ``fixed_values`` is pinned to its constant (so FitConfig toggles map
    to constants AutoDelta never sees); otherwise it is sampled from its prior. The
    ``layout`` decides per-site shape, so the single-exposure and time-series
    models share this one drawing loop. Returns ``{site_name: value}``.
    """
    def draw(site: Param):
        if site.name in fixed_values:
            return fixed_values[site.name]
        return numpyro.sample(site.name, site.prior(context, config, layout))

    return {s.name: draw(s) for s in iter_sites(dry_species)}
