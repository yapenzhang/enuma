"""The model's data structures
  * ``ModelParameters`` — pytree leaves JAX differentiates through.
  * ``ModelContext``    — fixed per setup but JAX-traced (grids, opacities).
  * ``ModelConfig``     — static ints/strings passed via ``static_argnums``.
"""

from dataclasses import dataclass
from typing import Dict, NamedTuple, Tuple

import jax.numpy as jnp
import numpy as np

__all__ = [
    "Layout",
    "ModelParameters",
    "params_to_sites",
    "sites_to_params",
    "SiteLocation",
    "ModelContext",
    "ModelConfig",
]


@dataclass(frozen=True)
class Layout:
    """Per-fit shape policy: which sites carry a leading per-exposure ``N`` axis.

    A single-exposure fit uses the default (``n_exp=0``), where every site is
    un-batched and ``lead`` is always ``()`` — so the registry produces exactly the
    original single-exposure shapes. A time-series night uses ``n_exp=N`` with
    ``per_exposure`` naming the sites that get a leading ``(N,)`` axis (built by
    :func:`enuma.inference.sites.per_exposure_sites` from the ``FitConfig`` sharing knobs).
    The same ``Layout`` drives the registry, the partitioner, the generative model,
    and the batched forward, so one code path serves both fit kinds.
    """
    n_exp: int = 0
    per_exposure: frozenset = frozenset()

    def lead(self, name: str) -> Tuple[int, ...]:
        return (self.n_exp,) if (self.n_exp and name in self.per_exposure) else ()


class ModelParameters(NamedTuple):
    """Everything the fit optimises (JAX differentiates through these)."""
    t_latent: jnp.ndarray                    # (n_active,) active bottom layers, ≤ n_layers-1
    h2o_latent: jnp.ndarray                  # (n_active,) active bottom layers, ≤ n_layers-1
    dry_vmr_scalars: Dict[str, jnp.ndarray]  # per-species scalar log10 offsets
    wave_coeffs: jnp.ndarray                 # (n_orders, n_wave_coeffs)
    resolution_coeffs: jnp.ndarray           # (n_orders, n_resolution_coeffs)
    continuum_coeffs: jnp.ndarray            # (n_orders, n_cont_nodes)
    # Only used when ``ModelConfig.stellar_enabled``; otherwise pinned constants
    # the forward model ignores.
    rv_systemic: jnp.ndarray = np.array(0.0, np.float32)  # scalar, km/s
    vsini: jnp.ndarray = np.array(1.0, np.float32)        # scalar, km/s


# The structured leaf that fans out to per-species flat sites, and the flat-key
# prefix it uses: ``dry_vmr_scalars={"co2": ...}`` <-> ``{"dry_vmr_co2": ...}``.
_DRY_FIELD = "dry_vmr_scalars"
_DRY_PREFIX = "dry_vmr_"


def params_to_sites(params: ModelParameters) -> Dict[str, jnp.ndarray]:
    """Flatten parameters into a ``{numpyro_site_name: value}`` dict.

    The inverse of :func:`sites_to_params`; together they are the single bridge
    between ``ModelParameters`` and the flat site dicts used by NumPyro,
    ``init_to_value``, and JSON I/O. Generic over ``ModelParameters._fields``: a
    new field appears as a site automatically.
    """
    sites: Dict[str, jnp.ndarray] = {}
    for field, value in params._asdict().items():
        if field == _DRY_FIELD:
            for species, v in value.items():
                sites[f"{_DRY_PREFIX}{species}"] = v
        else:
            sites[field] = value
    return sites


def sites_to_params(sites: Dict[str, jnp.ndarray],
                    dry_species: Tuple[str, ...]) -> ModelParameters:
    """Rebuild ``ModelParameters`` from a flat site dict (inverse of
    :func:`params_to_sites`).
    """
    kwargs: Dict[str, jnp.ndarray] = {}
    for field in ModelParameters._fields:
        if field == _DRY_FIELD:
            kwargs[field] = {s: sites[f"{_DRY_PREFIX}{s}"]
                             for s in dry_species if s != "h2o"}
        elif field in sites:
            kwargs[field] = sites[field]
    return ModelParameters(**kwargs)


class SiteLocation(NamedTuple):
    """Observatory geodetic location resolved at setup (an astropy site or the
    ``FitConfig.site_location`` override), carried on the context for downstream use."""
    lat_deg: float
    lon_deg: float
    alt_m: float


class ModelContext(NamedTuple):
    """Fixed-but-traced data and grids consumed by ``forward_model``."""
    # Layer grid (altitudes in cm; reference T/P/VMR at layer centres).
    z_boundaries: jnp.ndarray
    z_centers: jnp.ndarray
    t_ref_centers: jnp.ndarray
    p_ref_centers: jnp.ndarray
    p_ref_boundaries: jnp.ndarray
    vmr_ref_centers: Dict[str, jnp.ndarray]

    # Opacities: species name -> (grid_values, OpacityGridInfo).
    opacity_grids: Dict[str, Tuple[jnp.ndarray, tuple]]

    # Wavelengths.
    model_wave: jnp.ndarray         # high-res simulation grid
    obs_wave: jnp.ndarray           # observed grid (n_orders, n_pixels)
    obs_wave_norm: jnp.ndarray      # per-order [-1, 1] coord for the wave-shift poly
    obs_d_wave: jnp.ndarray         # exact per-pixel Δλ on obs grid (nm/pixel)
    order_start_idx: jnp.ndarray    # (n_orders,) model-grid start of each order chunk
    order_d_wave: jnp.ndarray       # (n_orders,) representative pixel spacing for LSF

    # Per-order resampling geometry in *offset* space (λ − order_ref_wave), 
    # so that float32 is not a limiting factor.
    order_ref_wave: jnp.ndarray     # (n_orders,) per-order reference wavelength
    model_wave_off: jnp.ndarray     # (n_orders, order_size) model grid − ref
    obs_wave_off: jnp.ndarray       # (n_orders, n_pixels) obs grid − ref

    # Cholesky factors of the per-layer GP priors (built once at setup, with
    # amplitude/length-scale/kernel baked in). In the non-centered
    # parametrization these expand the unit-normal latents into smooth physical
    # deviations: ``deviation = L @ latent`` (``L Lᵀ`` is the GP covariance).
    temp_chol_L: jnp.ndarray        # (n_layers, n_layers)
    h2o_chol_L: jnp.ndarray         # (n_layers, n_layers)

    cont_bspline_matrix: jnp.ndarray  # (n_pixels, n_cont_nodes)
    # Per-exposure observation terms: (N,) for a timeseries, else scalar/(1,).
    airmass: jnp.ndarray              # airmass, from FITS header
    baryrv: jnp.ndarray               # barycentric RV term (km/s)

    # Normalized Stellar template on the opacity grid
    stellar_template: jnp.ndarray = np.zeros(1, np.float32)      # (len(model_wave),)
    # The grid's representative per-pixel velocity step (km/s).
    stellar_dv_kms: jnp.ndarray = np.array(0.0, np.float32)  # scalar, km/s/pixel

    # Atmospheric winds (GDAS): horizontal components on the layer-centre grid
    # and the per-layer line-of-sight Doppler velocity used by the wind term.
    # ``wind_v_los`` is (n_layers,) for a single exposure, (N, n_layers) for a
    # night. ``model_dv_kms`` is the model grid's velocity step (uniform grid).
    wind_u_centers: jnp.ndarray = np.zeros(1, np.float32)    # (n_layers,) m/s, eastward
    wind_v_centers: jnp.ndarray = np.zeros(1, np.float32)    # (n_layers,) m/s, northward
    wind_v_los: jnp.ndarray = np.zeros(1, np.float32)        # (n_layers,) or (N, n_layers) km/s
    model_dv_kms: jnp.ndarray = np.array(0.0, np.float32)    # scalar, km/s per model sample

    # Empirical instrument LSF (``ModelConfig.lsf_profile == "custom"``): a fixed,
    # normalised kernel already resampled onto the model grid (odd length, centred).
    # The analytic gaussian/voigt profiles leave the default sentinel in place.
    lsf_kernel: jnp.ndarray = np.zeros(1, np.float32)


class ModelConfig(NamedTuple):
    """Static configuration; must be a static argument to ``jax.jit``."""
    lsf_kernel_width: int = 101
    # Constant-sigma chunks the variable-sigma LSF splits each order into.
    lsf_n_chunks: int = 20
    order_size: int = 20000          # per-order extent in model-grid samples
    # Polynomial coefficient counts (degree+1) for R(λ) and the wave shift,
    # evaluated on an intra-order coordinate in [-1, 1]; each order independent.
    n_resolution_coeffs: int = 2
    resolution_init: float = 0.6     # prior mean of the constant R term (×1e5)
    n_wave_coeffs: int = 3
    lsf_profile: str = "gaussian"    # "gaussian" or "voigt"
    lsf_voigt_gamma_ratio: float = 0.01
    # Stellar template multiplication (static, so it gates the forward-model
    # branch at trace time and the pure-telluric path stays bit-identical).
    stellar_enabled: bool = False
    stellar_rot_half_width: int = 151  # rotation-kernel half-width in model pixels
    stellar_epsilon: float = 0.6       # linear limb-darkening coeff for vsini
    stellar_rv_init: float = 0.0       # rv prior mean + SVI seed (km/s; barycentric+systemic)
    stellar_vsini_init: float = 2.0    # vsini prior mean + SVI seed (km/s)
    # Per-layer atmospheric wind Doppler shift (first-order). Static so it gates
    # the forward-model branch at trace time; off => bit-identical to no-wind.
    wind_enabled: bool = False
