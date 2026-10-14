import json
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from enuma.model.forward import forward_model, telluric_only
from enuma.state import ModelConfig, ModelContext, ModelParameters, sites_to_params

__all__ = [
    "save_best_fit_params_json",
    "load_best_fit_params_json",
    "save_best_fit_spectrum_txt",
    "generate_best_fit_spectrum",
]


def _jsonify(value):
    """Recursively convert jnp/np arrays + scalars to JSON-serialisable values."""
    if isinstance(value, (jnp.ndarray, np.ndarray)):
        return np.asarray(jax.device_get(value)).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def save_best_fit_params_json(
    params: ModelParameters,
    path: str,
    *,
    species: Optional[Tuple[str, ...]] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Save best-fit parameters to a human-readable JSON file."""
    if species is None:
        species = ("h2o", *params.dry_vmr_scalars.keys())
    payload = {"species": list(species), "params": _jsonify(params._asdict())}
    if metadata:
        payload["metadata"] = {k: str(v) for k, v in metadata.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_best_fit_params_json(
    path: str,
    *,
    species: Optional[Tuple[str, ...]] = None,
) -> ModelParameters:
    """Load best-fit parameters saved by :func:`save_best_fit_params_json`."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if species is None:
        species = tuple(payload.get("species", ()))
    if not species:
        raise ValueError("Missing 'species' in JSON; pass species explicitly.")

    raw = payload["params"]
    sites = {k: jnp.asarray(v) for k, v in raw.items() if k != "dry_vmr_scalars"}
    for s, v in raw.get("dry_vmr_scalars", {}).items():
        sites[f"dry_vmr_{s}"] = jnp.asarray(v)
    return sites_to_params(sites, tuple(species))


def save_best_fit_spectrum_txt(
    params: ModelParameters,
    context: ModelContext,
    config: ModelConfig,
    path: str,
) -> None:
    """Save the best-fit spectrum as a text file with the telluric transmission split out.

    Columns: ``wavelength_nm``, ``flux_model`` (the full model = telluric x continuum,
    x stellar when enabled), and ``flux_telluric`` (the telluric transmission alone —
    no continuum, stellar disabled). Dividing the observation by ``flux_telluric``
    removes the tellurics.
    """
    model_flux = np.asarray(jax.device_get(forward_model(params, context, config)))
    telluric_flux = np.asarray(jax.device_get(telluric_only(params, context, config)))
    wave = np.asarray(jax.device_get(context.obs_wave))
    data = np.column_stack((wave.ravel(), model_flux.ravel(), telluric_flux.ravel()))
    np.savetxt(path, data, fmt="%.10e",
               header="wavelength_nm flux_model flux_telluric", comments="")


def generate_best_fit_spectrum(
    params: ModelParameters,
    context: ModelContext,
    config: ModelConfig,
    *,
    obs_flux: Optional[jnp.ndarray] = None,
    obs_mask: Optional[jnp.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Generate the best-fit model spectrum (and optionally attach obs/mask)."""
    model_flux = np.asarray(jax.device_get(forward_model(params, context, config)))
    result = {"wave": context.obs_wave, "flux_model": model_flux}
    if obs_flux is not None:
        result["flux_data"] = obs_flux
    if obs_mask is not None:
        result["mask"] = obs_mask
    return result
