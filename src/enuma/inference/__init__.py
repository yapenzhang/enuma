"""The fitting layer: parameter registry, NumPyro model, SVI runner, and the
high-level fit entry points.

``sites`` is the registry of fittable quantities (and samples them via
``draw_sites``); ``optimizer`` runs AutoDelta SVI; ``fit`` defines the generative
model (``telluric_model``) and wires everything into :func:`fit_spectrum` /
:func:`fit_timeseries`.
"""

from enuma.inference.sites import (
    Param,
    default_params,
    draw_sites,
    dry_species_of,
    iter_sites,
    partition_sites,
    per_exposure_sites,
    resolve_fit_species,
)
from enuma.inference.optimizer import run_svi
from enuma.inference.fit import FitResult, fit_spectrum, fit_timeseries, telluric_model

__all__ = [
    "Param",
    "default_params",
    "dry_species_of",
    "iter_sites",
    "partition_sites",
    "per_exposure_sites",
    "resolve_fit_species",
    "draw_sites",
    "telluric_model",
    "run_svi",
    "FitResult",
    "fit_spectrum",
    "fit_timeseries",
]
