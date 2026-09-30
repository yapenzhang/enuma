import logging
from typing import NamedTuple, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
from astropy.coordinates import EarthLocation
from scipy.interpolate import BSpline

from enuma.config import FitConfig
from enuma.constants import C_KMS
from enuma.model.forward import LSF_PROFILE_CHOICES
from enuma.model.opacity import get_opacities
from enuma.model.profile import (
    build_altitude_grid,
    build_gp_factors,
    build_reference_atmosphere,
)
from enuma.model.stellar import build_stellar_from_config
from enuma.state import Layout, ModelConfig, ModelContext, SiteLocation

__all__ = [
    "OPACITY_WAVE_PAD_FRAC",
    "load_opacities",
    "setup_model",
    "get_observatory",
    "baryrv_kms",
    "wind_los_kms",
    "build_context",
]

logger = logging.getLogger("enuma.model.setup")

# Wavelength padding (fraction of obs_wave extent) when loading opacity tables
OPACITY_WAVE_PAD_FRAC = 0.01


def _bspline_matrix(x_array: np.ndarray, n_weights: int, degree: int = 3) -> np.ndarray:
    """Dense B-spline design matrix for continuum model, shape ``(len(x_array), n_weights)``."""
    x_min, x_max = np.min(x_array), np.max(x_array)
    n_interior = n_weights - degree + 1
    if n_interior < 2:
        raise ValueError(f"For degree={degree}, n_weights must be >= {degree + 1}")
    base_knots = np.linspace(x_min, x_max, n_interior)
    knots = np.concatenate(([x_min] * degree, base_knots, [x_max] * degree))
    return BSpline.design_matrix(x_array, knots, degree).toarray()


class _OrderMeta(NamedTuple):
    """Per-order geometry for the jitted resampler (precomputed once)."""
    order_size: int
    order_start_idx: np.ndarray         # (n_orders,) model-grid start of each chunk
    order_d_wave: np.ndarray            # (n_orders,) representative model Δλ for LSF
    obs_wave_norm: jnp.ndarray          # (n_orders, n_pix) [-1, 1] wave-shift coord
    obs_d_wave: jnp.ndarray             # (n_orders, n_pix) exact per-pixel Δλ
    order_ref_wave: np.ndarray          # (n_orders,) per-order reference wavelength
    model_wave_off: np.ndarray          # (n_orders, order_size) model λ − ref
    obs_wave_off: np.ndarray            # (n_orders, n_pix) obs λ − ref


def _order_metadata(model_wave, obs_wave) -> _OrderMeta:
    """Per-order chunk size, start index, pixel spacing, and wavelength offsets.
    Precomputed once so the jitted forward model carries no searchsorted work.
    """
    model_wave_np = np.asarray(model_wave, dtype=np.float64)
    obs_wave_np = np.asarray(obs_wave, dtype=np.float64)

    obs_len = obs_wave_np[:, -1] - obs_wave_np[:, 0]
    obs_center = (obs_wave_np[:, -1] + obs_wave_np[:, 0]) / 2.0
    model_dwave = np.interp(obs_center, model_wave_np[:-1], np.diff(model_wave_np))
    order_size = int((obs_len / model_dwave).max() * 1.2)

    max_start = max(model_wave_np.shape[0] - order_size, 0)
    order_start_idx = np.clip(
        np.searchsorted(model_wave_np, obs_wave_np[:, 0] - 1.0), 0, max_start,
    ).astype(np.int32)

    order_d_wave = np.array([
        np.median(np.diff(model_wave_np[s: s + order_size])) for s in order_start_idx
    ], dtype=np.float32)

    # Per-order model chunk (float64) and its midpoint reference. The model and
    # obs grids are both stored as offsets from this same reference so the
    # resampler's coordinates (edges, bin offsets, the cumulative integral's
    # abscissa) stay near zero where float32 has full relative precision.
    model_chunks = np.stack([model_wave_np[s: s + order_size] for s in order_start_idx])
    order_ref_wave = 0.5 * (model_chunks[:, 0] + model_chunks[:, -1])
    model_wave_off = (model_chunks - order_ref_wave[:, None]).astype(np.float32)
    obs_wave_off = (obs_wave_np - order_ref_wave[:, None]).astype(np.float32)

    # Per-order [-1, 1] coordinate for the wave-shift polynomial.
    obs_centered = obs_wave_np - obs_wave_np.mean(axis=1, keepdims=True)
    obs_wave_norm = jnp.asarray(
        (obs_centered / np.abs(obs_centered).max(axis=1, keepdims=True)).astype(np.float32))

    d = np.diff(obs_wave_np, axis=1)
    obs_d_wave = jnp.asarray(np.concatenate([d, d[:, -1:]], axis=1).astype(np.float32))

    return _OrderMeta(
        order_size=order_size,
        order_start_idx=order_start_idx,
        order_d_wave=order_d_wave,
        obs_wave_norm=obs_wave_norm,
        obs_d_wave=obs_d_wave,
        order_ref_wave=order_ref_wave.astype(np.float32),
        model_wave_off=model_wave_off,
        obs_wave_off=obs_wave_off,
    )


def _opacity_wave_range(obs_wave) -> Tuple[float, float]:
    """Padded (min, max) wavelength span to load opacities over (so out-of-range
    lines still contribute through their wings)."""
    return (float(obs_wave[0, 0]) * (1.0 - OPACITY_WAVE_PAD_FRAC),
            float(obs_wave[-1, -1]) * (1.0 + OPACITY_WAVE_PAD_FRAC))


def load_opacities(fit_config: FitConfig, obs_wave) -> Tuple[dict, np.ndarray]:
    return get_opacities(list(fit_config.species), _opacity_wave_range(obs_wave),
                         opacity_dir=fit_config.opacity_dir)


def _slice_opacities(opacities: Tuple[dict, np.ndarray],
                     wave_range: Tuple[float, float]) -> Tuple[dict, np.ndarray]:
    """Slice prebuilt opacity grids (from :func:`load_opacities`) to ``wave_range``.
    """
    full_grids, full_wave = opacities
    w_lo, w_hi = wave_range
    i0 = int(np.searchsorted(full_wave, w_lo))
    i1 = int(np.searchsorted(full_wave, w_hi, side="right"))
    if i0 >= i1:
        raise ValueError(f"No prebuilt opacity data in range {w_lo:.3f}-{w_hi:.3f} nm.")
    sub_wave = full_wave[i0:i1]
    sub_grids = {s: (g[..., i0:i1], info) for s, (g, info) in full_grids.items()}
    return sub_grids, sub_wave


def _load_lsf_kernel(path: str, model_dv_kms: float) -> jnp.ndarray:
    """Load an empirical LSF from a text file and resample it onto the model grid.

    The file is two whitespace-separated columns — velocity offset (km/s) and kernel
    amplitude (``#`` comment lines allowed). Velocity is the instrument-natural,
    wavelength-independent unit, and the model grid is uniform in velocity
    (``model_dv_kms``), so the kernel maps onto integer model-grid offsets exactly.
    The result is centred at zero offset, odd-length, and normalised to unit sum.
    """
    raw = np.loadtxt(path, comments="#", dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError(f"LSF kernel file {path!r} must have two columns and multiple "
                         "rows (velocity[km/s], amplitude).")
    v_kms, amp = raw[:, 0], raw[:, 1]
    order = np.argsort(v_kms)
    v_kms, amp = v_kms[order], amp[order]

    half = max(int(np.ceil(np.max(np.abs(v_kms)) / float(model_dv_kms))), 1)
    offsets = np.arange(-half, half + 1) * float(model_dv_kms)   # odd, centred at 0
    kernel = np.interp(offsets, v_kms, amp, left=0.0, right=0.0)

    total = kernel.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"LSF kernel from {path!r} sums to {total}; expected a "
                         "positive profile spanning the velocity grid.")
    return jnp.asarray(kernel / total, dtype=jnp.float32)


def setup_model(fit_config: FitConfig, obs_wave, *,
                airmass, location: Optional[SiteLocation] = None,
                pwv_mm: Optional[float] = None, obs_mjd=None,
                opacities: Optional[Tuple[dict, np.ndarray]] = None,
                ) -> Tuple[ModelContext, ModelConfig]:
    """Build the (context, config) pair consumed by ``forward_model``."""
    fc = fit_config
    if fc.lsf_profile not in LSF_PROFILE_CHOICES:
        raise ValueError(f"Unknown lsf_profile {fc.lsf_profile!r}. Choices: {LSF_PROFILE_CHOICES}.")
    if fc.wind_enabled and fc.reference_profile_source != "gdas":
        raise ValueError("wind_enabled=True requires reference_profile_source='gdas' "
                         "(winds are read from the GDAS analysis).")
    if location is None:
        raise ValueError(
            "Observatory location is unavailable. Set FitConfig.observatory to a site "
            "astropy can resolve (EarthLocation.get_site_names()), pass an EarthLocation, "
            "or set FitConfig.site_location=(lat_deg, lon_deg, alt_m).")

    wave_range = _opacity_wave_range(obs_wave)
    if opacities is None:
        opacity_grids, model_wave = get_opacities(list(fc.species), wave_range,
                                                  opacity_dir=fc.opacity_dir)
    else:
        opacity_grids, model_wave = _slice_opacities(opacities, wave_range)
    missing = [s for s in fc.species if s not in opacity_grids]
    if missing:
        raise ValueError(f"Missing opacity grids for species: {missing}.")

    meta = _order_metadata(model_wave, obs_wave)
    order_size = meta.order_size

    z_boundaries, z_centers = build_altitude_grid(fc, location.alt_m)
    t_ref_centers, p_ref_centers, p_ref_boundaries, vmr_ref_centers, winds = build_reference_atmosphere(
        fc, z_centers, z_boundaries, fc.cache_dir,
        obs_mjd=obs_mjd, location=location, pwv_mm=pwv_mm)
    temp_chol_L, h2o_chol_L = build_gp_factors(fc, z_centers)

    # Model grid velocity step (the opacity grid is uniform in velocity) and the
    # GDAS winds on the layer grid (zeros when winds are unavailable).
    model_dv_kms = float(np.median(np.diff(np.log(np.asarray(model_wave, dtype=np.float64)))) * C_KMS)
    wind_u_centers, wind_v_centers = (
        winds if winds is not None else (jnp.zeros_like(z_centers), jnp.zeros_like(z_centers)))

    # Empirical LSF: resample the measured kernel onto the (velocity-uniform) model
    # grid once; the analytic profiles leave the sentinel in place.
    lsf_kernel = (_load_lsf_kernel(fc.lsf_kernel_file, model_dv_kms)
                  if fc.lsf_profile == "custom" else jnp.zeros(1))

    # Continuum B-spline basis on per-pixel x in [0, 1].
    cont_bspline_matrix = _bspline_matrix(np.linspace(0.0, 1.0, obs_wave.shape[1]), fc.continuum_n_nodes)

    stellar_template, stellar_dv_kms = build_stellar_from_config(fc, model_wave, fc.cache_dir)

    context = ModelContext(
        z_boundaries=z_boundaries,
        z_centers=z_centers,
        t_ref_centers=t_ref_centers,
        p_ref_centers=p_ref_centers,
        p_ref_boundaries=p_ref_boundaries,
        vmr_ref_centers=vmr_ref_centers,
        opacity_grids=opacity_grids,
        model_wave=jnp.asarray(model_wave),
        obs_wave=jnp.asarray(obs_wave),
        obs_wave_norm=meta.obs_wave_norm,
        obs_d_wave=meta.obs_d_wave,
        order_start_idx=jnp.array(meta.order_start_idx),
        order_d_wave=jnp.array(meta.order_d_wave),
        order_ref_wave=jnp.asarray(meta.order_ref_wave),
        model_wave_off=jnp.asarray(meta.model_wave_off),
        obs_wave_off=jnp.asarray(meta.obs_wave_off),
        temp_chol_L=temp_chol_L,
        h2o_chol_L=h2o_chol_L,
        cont_bspline_matrix=jnp.array(cont_bspline_matrix),
        airmass=jnp.asarray(airmass),
        baryrv=jnp.zeros_like(jnp.asarray(airmass)),
        stellar_template=stellar_template,
        stellar_dv_kms=stellar_dv_kms,
        wind_u_centers=wind_u_centers,
        wind_v_centers=wind_v_centers,
        wind_v_los=jnp.zeros_like(z_centers),
        model_dv_kms=jnp.asarray(model_dv_kms),
        lsf_kernel=lsf_kernel,
    )
    config = ModelConfig(
        lsf_kernel_width=int(fc.lsf_kernel_width),
        lsf_n_chunks=int(fc.lsf_n_chunks),
        order_size=order_size,
        n_resolution_coeffs=int(fc.n_resolution_coeffs),
        n_wave_coeffs=int(fc.n_wave_coeffs),
        resolution_init=float(fc.resolution),
        lsf_profile=fc.lsf_profile.lower(),
        lsf_voigt_gamma_ratio=float(fc.lsf_voigt_gamma_ratio),
        stellar_enabled=bool(fc.stellar_enabled),
        stellar_rot_half_width=int(fc.stellar_rot_half_width),
        stellar_epsilon=float(fc.stellar_epsilon),
        stellar_rv_init=float(fc.rv_kms) if fc.stellar_enabled else 0.0,
        stellar_vsini_init=float(fc.vsini),
        wind_enabled=bool(fc.wind_enabled),
    )
    return context, config


# ---------------------------------------------------------------------------
# High-level: resolve the observation inputs from a loaded-data dict, then setup_model
# ---------------------------------------------------------------------------

def get_observatory(observatory: Union[str, EarthLocation, None]) -> Optional[EarthLocation]:

    if observatory is None:
        return None
    if isinstance(observatory, EarthLocation):
        return observatory
    try:
        return EarthLocation.of_site(str(observatory))
    except Exception as exc:
        logger.warning("Could not resolve observatory %r via astropy (%s); use a name "
                       "from EarthLocation.get_site_names() or pass an EarthLocation. "
                       "Continuing without it.", observatory, exc)
        return None


def baryrv_kms(ra_deg, dec_deg, mjd, observatory, *, n: Optional[int] = None) -> np.ndarray:
    """Absolute barycentric RV term ``= -v_bary`` (km/s) for a target from an observatory.

    Scalar when ``n is None`` (a single exposure), else a length-``n`` array (one
    per exposure of a night). On missing coordinates/time or an unresolved observatory
    it returns zero (correction disabled, with a warning). The net stellar shift is
    ``baryrv + rv_systemic`` (formed in :func:`enuma.model.forward.forward_model`).
    """
    loc = get_observatory(observatory)
    zero = np.zeros(()) if n is None else np.zeros(int(n))
    if ra_deg is None or dec_deg is None or mjd is None or loc is None:
        logger.warning("Missing target coords/time or unresolved observatory %r; "
                       "barycentric RV term set to 0.", observatory)
        return zero
    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from astropy.time import Time

        sc = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        t = Time(np.asarray(mjd, float), format="mjd", scale="utc")
        vbary = sc.radial_velocity_correction(
            "barycentric", obstime=t, location=loc).to(u.km / u.s).value
    except Exception as exc:
        logger.warning("Barycentric correction failed (%s); barycentric RV term set to 0.", exc)
        return zero
    return np.asarray(-vbary)


def wind_los_kms(u_centers, v_centers, ra_deg, dec_deg, mjd, observatory,
                 *, n: Optional[int] = None) -> np.ndarray:
    """Per-layer line-of-sight wind velocity (km/s) from horizontal GDAS winds.

    Projects (u eastward, v northward) winds (m/s) onto the line of sight,
    ``v_los = sin(z)·(u·sin A + v·cos A)``, with azimuth ``A`` and zenith angle
    ``z`` from an astropy AltAz transform of the target at the observation
    time(s). Positive = motion away from the observer (redshift). Shape
    ``(n_layers,)`` for a single exposure, else ``(n, n_layers)``. Returns zeros
    on missing coords/time or an unresolved observatory.

    NOTE: the overall sign depends on the GDAS wind / azimuth conventions and
    should be verified on-sky (cf. the systemic-RV sign).
    """
    u_ms = np.asarray(u_centers, dtype=float)
    v_ms = np.asarray(v_centers, dtype=float)
    n_layers = u_ms.shape[0]
    zero = np.zeros(n_layers) if n is None else np.zeros((int(n), n_layers))
    loc = get_observatory(observatory)
    if ra_deg is None or dec_deg is None or mjd is None or loc is None:
        logger.warning("Missing target coords/time or unresolved observatory %r; "
                       "wind LOS set to 0.", observatory)
        return zero
    try:
        import astropy.units as u
        from astropy.coordinates import AltAz, SkyCoord
        from astropy.time import Time

        sc = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        t = Time(np.asarray(mjd, float), format="mjd", scale="utc")
        altaz = sc.transform_to(AltAz(obstime=t, location=loc))
        az = np.atleast_1d(altaz.az.to_value(u.rad))
        zen = np.atleast_1d((np.pi / 2.0) - altaz.alt.to_value(u.rad))
    except Exception as exc:
        logger.warning("AltAz transform failed (%s); wind LOS set to 0.", exc)
        return zero

    # (n_exp, n_layers): per-exposure geometry × per-layer winds, m/s → km/s.
    v_los = (np.sin(zen)[:, None]
             * (u_ms[None, :] * np.sin(az)[:, None] + v_ms[None, :] * np.cos(az)[:, None])
             ) / 1000.0
    return v_los[0] if n is None else v_los


def _resolve_airmass(fit_config: FitConfig, data: dict) -> jnp.ndarray:
    if fit_config.airmass is not None:
        return jnp.array([float(fit_config.airmass)])
    if data.get("airmass") is not None:
        return jnp.array([float(data["airmass"])])
    logger.warning("No airmass in header and FitConfig.airmass is None; using 1.0.")
    return jnp.array([1.0])


def _resolve_pwv_mm(fit_config: FitConfig, data: dict) -> Optional[float]:
    if fit_config.pwv_mm is not None:
        return float(fit_config.pwv_mm)
    if data.get("pwv_mm") is not None:
        return float(data["pwv_mm"])
    logger.warning("No PWV in header and FitConfig.pwv_mm is None; skipping H2O scaling.")
    return None


def _earth_location(fit_config: FitConfig) -> Optional[EarthLocation]:
    """Resolve the observatory to an EarthLocation: astropy's site registry first,
    then the explicit ``FitConfig.site_location=(lat_deg, lon_deg, alt_m)`` fallback."""
    loc = get_observatory(fit_config.observatory)
    if loc is not None:
        return loc
    if fit_config.site_location is not None:
        lat, lon, alt = fit_config.site_location
        return EarthLocation.from_geodetic(lon, lat, alt)
    return None


def _site_location(loc: Optional[EarthLocation]) -> Optional[SiteLocation]:
    """Geodetic lat/lon/altitude of a resolved EarthLocation as a SiteLocation."""
    if loc is None:
        return None
    return SiteLocation(lat_deg=float(loc.lat.to_value("deg")),
                        lon_deg=float(loc.lon.to_value("deg")),
                        alt_m=float(loc.height.to_value("m")))


def build_context(data: dict, fit_config: FitConfig,
                  layout: Layout = Layout(), *,
                  opacities: Optional[Tuple[dict, np.ndarray]] = None,
                  ) -> Tuple[ModelContext, ModelConfig]:
    """Build the (context, config) pair for a single exposure or a timeseries.

    Only ``layout.n_exp`` is consulted, so a bare ``Layout(n_exp=N)`` or the full
    per-exposure layout both work. ``n_exp == 0`` builds a single-exposure context
    with the resolved scalar airmass baked in (plus the scalar barycentric term when
    stellar is on). ``n_exp > 0`` builds the heavy context once (the wave grid is
    shared across the night) and attaches the per-exposure ``airmass`` and ``baryrv``
    arrays — ``baryrv`` from ``data['baryrv']`` when present, else derived by
    :func:`enuma.model.setup.baryrv_kms` from the coords / per-exposure MJDs / site.
    """
    # Keep the obs_wave in float64 into setup, downcasted to float32 only once it lands on device.
    obs_wave = np.asarray(data["wave"], dtype=np.float64)
    earth_loc = _earth_location(fit_config)
    location = _site_location(earth_loc)
    # One median observation time for the date/site reference profile;
    obs_mjd = data.get("mjd")
    if obs_mjd is not None and np.ndim(obs_mjd) > 0:
        obs_mjd = float(np.median(obs_mjd))
    obs_kw = dict(location=location, pwv_mm=_resolve_pwv_mm(fit_config, data),
                  obs_mjd=obs_mjd, opacities=opacities)
    
    # single-exposure
    if layout.n_exp == 0:
        context, config = setup_model(fit_config, obs_wave,
                                      airmass=_resolve_airmass(fit_config, data), **obs_kw)
        if fit_config.stellar_enabled:
            baryrv = baryrv_kms(data.get("target_ra"), data.get("target_dec"),
                                data.get("mjd"), earth_loc)
            context = context._replace(baryrv=jnp.asarray(baryrv, dtype=jnp.float32))
        if fit_config.wind_enabled:
            v_los = wind_los_kms(context.wind_u_centers, context.wind_v_centers,
                                 data.get("target_ra"), data.get("target_dec"),
                                 data.get("mjd"), earth_loc)
            context = context._replace(wind_v_los=jnp.asarray(v_los, dtype=jnp.float32))
        return context, config

    # timeseries
    context, config = setup_model(fit_config, obs_wave, airmass=jnp.asarray(1.0), **obs_kw)
    airmass = np.asarray(data["airmass"])
    baryrv = data.get("baryrv")
    if baryrv is None:   # not stored (e.g. FITS) — derive from coords + MJD + site
        baryrv = baryrv_kms(data.get("target_ra"), data.get("target_dec"),
                            data.get("mjd"), earth_loc, n=airmass.shape[0])
    context = context._replace(
        airmass=jnp.asarray(airmass, dtype=jnp.float32),       # (N,)
        baryrv=jnp.asarray(baryrv, dtype=jnp.float32),         # (N,) km/s
    )
    if fit_config.wind_enabled:
        v_los = wind_los_kms(context.wind_u_centers, context.wind_v_centers,
                             data.get("target_ra"), data.get("target_dec"),
                             data.get("mjd"), earth_loc, n=airmass.shape[0])
        context = context._replace(wind_v_los=jnp.asarray(v_los, dtype=jnp.float32))
    return context, config
