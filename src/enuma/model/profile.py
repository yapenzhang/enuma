"""Reference atmosphere: RFM-format .atm files + optional NOAA GDAS columns.

``get_standard_atmosphere`` interpolates T, P, and per-species VMR onto a target
altitude grid. Dry species always come from the bundled MIPAS .atm file; T, P,
and H2O can instead come from a date/site-specific GDAS analysis column
(auto-fetched and cached from the public AWS mirror).

The ``FitConfig``-driven builders ``setup_model`` consumes also live here: the
layer grid (``build_altitude_grid``), the reference T/P/VMR on it
(``build_reference_atmosphere``, with H2O scaled to the target PWV), and the
per-layer GP prior factors (``build_gp_factors``).
"""

import datetime as dt
import logging
import os
import tempfile
from typing import Optional

import jax.numpy as jnp
import numpy as np

from enuma.config import FitConfig
from enuma.constants import G_EARTH, M_DRY_AIR, M_H2O, M_O3, MMW_DRY_AIR, MMW_H2O
from enuma.model._paths import ENV_VAR, cache_root, data_root
from enuma.state import SiteLocation

__all__ = [
    "REFERENCE_PROFILE_CHOICES",
    "fetch_gdas_profile",
    "read_atm",
    "get_standard_atmosphere",
    "build_altitude_grid",
    "build_reference_atmosphere",
    "GP_KERNEL_CHOICES",
    "build_gp_factors",
]

logger = logging.getLogger("enuma.model.profile")

REFERENCE_PROFILE_CHOICES = ("mipas", "gdas")


# ---------------------------------------------------------------------------
# GDAS fetch + cache (public AWS mirror noaa-gfs-bdp-pds; mid-2021 onward)
# ---------------------------------------------------------------------------

def fetch_gdas_profile(mjd: float, lat: float, lon: float,
                       cache_dir: str = None, force_refresh: bool = False) -> str:
    """Fetch and cache a GDAS analysis column for (MJD, lat, lon).

    The observation time (MJD, UTC) is snapped to the nearest analysis cycle
    (00/06/12/18 UTC) and (lat, lon) to the 0.25° grid; the profile is cached  
    under ``$ENUMA_CACHE/gdas`` (or ``~/.cache/enuma/gdas``).
    Returns the NetCDF path
    """
    from astropy.time import Time
    date_utc = Time(float(mjd), format="mjd", scale="utc").to_datetime(timezone=dt.timezone.utc)
    # Snap to the nearest analysis cycle (lon folded into [0, 360)).
    cycle = min(
        (date_utc.replace(hour=h, minute=0, second=0, microsecond=0) + off
         for h in (0, 6, 12, 18)
         for off in (-dt.timedelta(days=1), dt.timedelta(0), dt.timedelta(days=1))),
        key=lambda c: abs((c - date_utc).total_seconds()),
    )
    lat_cell, lon_cell = round(lat * 4) / 4.0, round((lon % 360.0) * 4) / 4.0

    out_path = cache_root(cache_dir, subdir="gdas") / f"gdas_{cycle:%Y%m%d_%H}_lat{lat_cell:+06.2f}_lon{lon_cell:06.2f}.nc"
    if out_path.exists() and not force_refresh:
        logger.debug("GDAS cache hit: %s", out_path)
        return str(out_path)

    logger.info("Fetching GDAS for cycle %s at lat=%.4f lon=%.4f (cell %.2f, %.2f) -> %s",
                cycle.strftime("%Y-%m-%d %H:%M UTC"), lat, lon, lat_cell, lon_cell, out_path)

    # Download the GRIB to a temp file, extract the column, cache the NetCDF.
    with tempfile.TemporaryDirectory() as tmp:
        grib_path = os.path.join(tmp, "gdas.grb2")
        _download_gdas_grib(grib_path, cycle)
        profile = _extract_gdas_column(grib_path, lat, lon).squeeze(drop=True)
        keep = [v for v in ("t", "q", "gh", "o3mr", "u", "v") if v in profile.data_vars]
        try:
            profile[keep].to_netcdf(str(out_path), engine="h5netcdf")
        except (ImportError, ValueError):
            profile[keep].to_netcdf(str(out_path))
    return str(out_path)


def _download_gdas_grib(local_path: str, cycle: dt.datetime) -> None:
    """Download a cycle's GDAS GRIB2 analysis from the public AWS mirror."""
    try:
        import s3fs
    except ImportError:
        raise ImportError("s3fs is required to fetch GDAS analyses. Install with: pip install s3fs")
    src = (f"noaa-gfs-bdp-pds/gdas.{cycle:%Y%m%d}/{cycle:%H}/atmos/"
           f"gdas.t{cycle:%H}z.pgrb2.0p25.f000")
    fs = s3fs.S3FileSystem(anon=True)
    if not fs.exists(src):
        raise FileNotFoundError(f"GDAS data not found on s3://{src} (mirror covers mid-2021 onward).")
    logger.info("Downloading s3://%s ...", src)
    fs.get(src, local_path)
    logger.info("  → %s  (%.1f MB)", local_path, os.path.getsize(local_path) / 1e6)


def _extract_gdas_column(grib_path: str, lat: float, lon: float):
    """Stitch isobaricInhPa (33 levels) + isobaricInPa (8 levels) and select the
    column nearest the telescope. Returns the merged xarray Dataset."""
    try:
        import xarray as xr
    except ImportError:
        raise ImportError("xarray + cfgrib are required to read GDAS GRIB2 files. "
                          "Install with: pip install xarray cfgrib")

    def _open(filter_keys):
        return xr.open_dataset(grib_path, engine="cfgrib",
                               backend_kwargs={"filter_by_keys": filter_keys})

    ds_hPa = _open({"typeOfLevel": "isobaricInhPa", "shortName": ["t", "q", "gh", "o3mr", "u", "v"]})
    ds_Pa = _open({"typeOfLevel": "isobaricInPa", "shortName": ["t", "q", "gh", "o3mr", "u", "v"]})
    ds_Pa = (ds_Pa.assign_coords(isobaricInPa=ds_Pa["isobaricInPa"] / 100.0)
                  .rename({"isobaricInPa": "isobaricInhPa"}))

    pres = (xr.concat([ds_hPa, ds_Pa], dim="isobaricInhPa")
              .sortby("isobaricInhPa", ascending=False))
    return pres.sel(latitude=lat, longitude=lon % 360.0, method="nearest")


def _load_gdas_profile(path):
    """Load a GDAS column NetCDF; return (z_km, p_hPa, T_K, vmr_h2o, vmr_o3,
    u_ms, v_ms) sorted by ascending altitude, or None if unreadable. ``gh`` is
    geopotential height in metres; ``q`` follows the ECMWF moist-air convention.
    ``u`` (eastward) / ``v`` (northward) winds are None when the column lacks them
    (e.g. a cache written before winds were retained)."""
    if path is None:
        return None
    try:
        import xarray as xr
    except ImportError as exc:
        logger.warning("xarray missing — cannot read GDAS profile: %s", exc)
        return None
    try:
        try:
            ds = xr.open_dataset(path, engine="h5netcdf")
        except (ImportError, ValueError):
            ds = xr.open_dataset(path)
    except (FileNotFoundError, OSError) as exc:
        logger.warning("Could not read GDAS profile '%s': %s", path, exc)
        return None

    ds = ds.squeeze(drop=True)
    try:
        p = np.asarray(ds["isobaricInhPa"].values, dtype=float)
        t = np.asarray(ds["t"].values, dtype=float)
        q = np.asarray(ds["q"].values, dtype=float)
        o3mr = np.asarray(ds["o3mr"].values, dtype=float)
        gh = np.asarray(ds["gh"].values, dtype=float)  # metres (gpm)
    except KeyError as exc:
        logger.warning("GDAS profile '%s' missing variable %s", path, exc)
        return None

    f_dry = 1.0 - q
    z_km = gh / 1000.0
    vmr_h2o = (q / f_dry) * (M_DRY_AIR / M_H2O)
    vmr_o3 = (o3mr / f_dry) * (M_DRY_AIR / M_O3)
    u = np.asarray(ds["u"].values, dtype=float) if "u" in ds.data_vars else None  # eastward m/s
    v = np.asarray(ds["v"].values, dtype=float) if "v" in ds.data_vars else None  # northward m/s

    order = np.argsort(z_km)
    u = u[order] if u is not None else None
    v = v[order] if v is not None else None
    return z_km[order], p[order], t[order], vmr_h2o[order], vmr_o3[order], u, v


# ---------------------------------------------------------------------------
# MIPAS .atm reader
# ---------------------------------------------------------------------------

def read_atm(filename):
    """Return the contents of an RFM .atm file as a dict.

    Always contains ``'nlev'`` (number of profile levels) plus one key per
    profile (lower-cased label) holding an ``nlev``-length numpy array. See
    http://eodg.atm.ox.ac.uk/RFM/sum/atmfil.html for the format.
    """
    with open(filename) as f:
        rec = "!"
        while rec[0] == "!":
            rec = f.readline()  # skip initial comments
        nlev = int(rec.split()[0])
        atm = {"nlev": nlev}
        rec = f.readline()
        while rec[0:4].lower() != "*end":
            if rec[0] == "!":
                rec = f.readline()
                continue
            key = rec.split()[0][1:].lower()  # strip '*', lower-case
            atm[key] = np.fromfile(f, sep=" ", count=nlev)
            rec = f.readline()
            if rec == "":
                break
    return atm


def _read_standard_atmosphere(model_name, species_list):
    """Read a MIPAS .atm scenario. Returns (height cm, pressure Ba, temp K,
    {species: log10(VMR)}) in natural order (increasing altitude)."""
    path = data_root() / f"{model_name}.atm"
    if not path.exists():
        raise FileNotFoundError(
            f"Reference atmosphere '{model_name}.atm' not found at {path}. "
            f"Set ${ENV_VAR} to the data directory.")
    atm = read_atm(str(path))
    height = atm["hgt"] * 1e5      # km → cm (increasing)
    press = atm["pre"] * 1e3       # hPa → Ba (decreasing)
    temp = atm["tem"]              # K

    vmr_log = {}
    for species in species_list:
        name = "".join(ele.lstrip("0123456789") for ele in species.split("_"))
        if name.lower() not in atm.keys():
            raise ValueError(f"Species '{name}' not found in {model_name}.atm")
        vmr_log[species] = jnp.log10(atm[name.lower()]) - 6.0  # ppm → log10(VMR)

    return height, press, temp, vmr_log


def get_standard_atmosphere(
    z_grid,
    species_list,
    *,
    source: str = "mipas",
    profile_nc_path: str = None,
    mipas_model: str = "equ",
):
    """Return a (T, P, VMR-by-species, winds) reference atmosphere on ``z_grid``.

    Dry species (CO2, CH4, N2O, CO, O2, NO, O3) always come from the MIPAS
    ``mipas_model`` .atm scenario. ``source`` controls only T, P, and H2O:

    * ``"mipas"`` (default): T, P, H2O from the same .atm file (no extra files).
    * ``"gdas"``: override T, P, H2O, O3 with the GDAS column at ``profile_nc_path``.

    ``winds`` is the (u, v) m/s horizontal wind on ``z_grid`` for ``"gdas"`` (when
    the column carries winds), else None.
    """
    if source not in REFERENCE_PROFILE_CHOICES:
        raise ValueError(f"Unknown reference-profile source '{source}'. Choices: {REFERENCE_PROFILE_CHOICES}")

    # MIPAS is always loaded: canonical dry-species VMRs (+ T/P/H2O for "mipas").
    mipas_height, mipas_press, mipas_temp, mipas_vmr_log = \
        _read_standard_atmosphere(mipas_model, species_list)

    if source == "mipas":
        tp = jnp.interp(z_grid, mipas_height, mipas_temp)
        p_ref = 10.0 ** jnp.interp(z_grid, mipas_height, jnp.log10(mipas_press))
        vmr_dict = {s: 10.0 ** jnp.interp(z_grid, mipas_height, mipas_vmr_log[s])
                    for s in species_list}
        return tp, p_ref, vmr_dict, None

    # GDAS path: T, P, H2O, O3 from the NetCDF column; other species from MIPAS.
    if profile_nc_path is None:
        raise ValueError(f"source='{source}' requires `profile_nc_path` to a column NetCDF.")
    loaded = _load_gdas_profile(profile_nc_path)
    if loaded is None:
        raise ValueError(f"Could not read {source.upper()} profile from {profile_nc_path!r}. "
                         "Run tests/test_gdas_opendata.py first.")
    z_obs_km, p_obs_hpa, t_obs, vmr_h2o_obs, vmr_o3_obs, u_obs, v_obs = loaded
    xp_obs = z_obs_km * 1e5                          # km → cm
    log_p_obs_ba = jnp.log10(p_obs_hpa * 1e3)        # hPa → Ba (log space)

    tp = jnp.interp(z_grid, xp_obs, t_obs)
    p_ref = 10.0 ** jnp.interp(z_grid, xp_obs, log_p_obs_ba)

    vmr_dict = {}
    for species in species_list:
        if species == "h2o":
            vmr_dict[species] = 10.0 ** jnp.interp(z_grid, xp_obs, np.log10(vmr_h2o_obs))
        elif species == "o3":
            vmr_dict[species] = 10.0 ** jnp.interp(z_grid, xp_obs, np.log10(vmr_o3_obs))
        else:
            vmr_dict[species] = 10.0 ** jnp.interp(z_grid, mipas_height, mipas_vmr_log[species])

    winds = None
    if u_obs is not None and v_obs is not None:
        winds = (jnp.interp(z_grid, xp_obs, u_obs), jnp.interp(z_grid, xp_obs, v_obs))
    return tp, p_ref, vmr_dict, winds


# ---------------------------------------------------------------------------
# Layer grid + reference atmosphere (FitConfig-driven; consumed by setup_model)
# ---------------------------------------------------------------------------

def build_altitude_grid(fc: FitConfig, obs_alt_m: float):
    """Altitude grid in cm. The power-law sampling (``z_sampling_exponent``)
    packs layers near the surface where most of the absorbing mass sits.
    """
    z_surface = obs_alt_m * 100.0          # m → cm
    z_top_cm = fc.z_top_km * 1e5           # km → cm
    s = jnp.linspace(0.0, 1.0, fc.n_layers) ** fc.z_sampling_exponent
    z_boundaries = z_surface + s * (z_top_cm - z_surface)
    z_centers = 0.5 * (z_boundaries[1:] + z_boundaries[:-1])
    return z_boundaries, z_centers


def _scale_h2o_to_pwv(vmr_ref_centers, p_ref_boundaries, pwv_mm):
    """Return ``vmr_ref_centers`` with H2O rescaled to a target PWV.

    The reference column is integrated with the same dry-air-column formula
    the forward model uses. No-op if PWV is None or H2O is absent.
    """
    if pwv_mm is None or "h2o" not in vmr_ref_centers:
        return vmr_ref_centers

    p_bnd = np.asarray(p_ref_boundaries)
    vmr_h2o = np.asarray(vmr_ref_centers["h2o"])
    air_column = np.abs(np.diff(p_bnd)) / (MMW_DRY_AIR * G_EARTH)
    pwv_ref_mm = float(np.sum(vmr_h2o * air_column)) * MMW_H2O * 10.0  # molec/cm² → mm
    if pwv_ref_mm <= 0:
        logger.warning("Reference H2O column integrates to PWV=%.3g mm — skipping scaling.", pwv_ref_mm)
        return vmr_ref_centers

    scale = pwv_mm / pwv_ref_mm
    logger.info("Scaled H2O reference by %.3f: reference PWV=%.3f mm → target PWV=%.3f mm.",
                scale, pwv_ref_mm, pwv_mm)
    return {**vmr_ref_centers, "h2o": jnp.asarray(vmr_h2o * scale)}


def _resolve_reference_profile_path(source, nc_path, obs_mjd, location, cache_dir):
    """For a date/site-specific source with no explicit path, auto-fetch one."""
    if source == "mipas" or nc_path is not None:
        return nc_path
    if obs_mjd is None or location is None:
        raise ValueError(
            f"reference_profile_source='{source}' requires either "
            "reference_profile_nc_path or (obs_mjd + a resolved observatory).")
    if source == "gdas":
        return fetch_gdas_profile(obs_mjd, location.lat_deg, location.lon_deg, cache_dir=cache_dir)
    raise NotImplementedError(f"Auto-fetch for source='{source}' is not implemented.")


def build_reference_atmosphere(fc: FitConfig, z_centers, z_boundaries, cache_dir,
                               *, obs_mjd, location: Optional[SiteLocation], pwv_mm):
    """Reference T/P/VMR on the layer grid (H2O scaled to the target PWV) plus the
    GDAS (u, v) winds in m/s on the layer centres (None when the source is not
    GDAS or the column carries no winds)."""
    profile_path = _resolve_reference_profile_path(
        fc.reference_profile_source, fc.reference_profile_nc_path,
        obs_mjd, location, cache_dir)
    t_ref_centers, p_ref_centers, vmr_ref_centers, winds = get_standard_atmosphere(
        z_centers, fc.species,
        source=fc.reference_profile_source,
        profile_nc_path=profile_path)
    _, p_ref_boundaries, _, _ = get_standard_atmosphere(
        z_boundaries, fc.species,
        source=fc.reference_profile_source,
        profile_nc_path=profile_path)
    vmr_ref_centers = _scale_h2o_to_pwv(vmr_ref_centers, p_ref_boundaries, pwv_mm)
    return t_ref_centers, p_ref_centers, p_ref_boundaries, vmr_ref_centers, winds


# ---------------------------------------------------------------------------
# Per-layer GP priors on the T and H2O profile deviations
# ---------------------------------------------------------------------------

GP_KERNEL_CHOICES = ("matern52", "matern32", "rbf", "rq")


def _kernel_matrix(z_grid_km, kernel_name: str, length_scale) -> jnp.ndarray:
    """Unit-amplitude covariance ``K[i,j] = k(|z_i - z_j| / length_scale)``.

    ``z_grid_km`` is the 1-D vector of layer-centre altitudes (km). Operation
    order mirrors tinygp's so the resulting Cholesky factor is bit-for-bit the
    same as before this kernel was inlined.
    """
    z = jnp.asarray(z_grid_km)
    diff = z[:, None] - z[None, :]                          # (n, n)
    if kernel_name == "matern52":
        r = jnp.abs(diff) / length_scale
        arg = np.sqrt(5.0) * r
        return (1.0 + arg + jnp.square(arg) / 3.0) * jnp.exp(-arg)
    if kernel_name == "matern32":
        r = jnp.abs(diff) / length_scale
        arg = np.sqrt(3.0) * r
        return (1.0 + arg) * jnp.exp(-arg)
    if kernel_name == "rbf":                                # ExpSquared / RBF
        r2 = jnp.square(diff) / jnp.square(length_scale)
        return jnp.exp(-0.5 * r2)
    if kernel_name == "rq":                                 # RationalQuadratic, alpha = 2
        r2 = jnp.square(diff) / jnp.square(length_scale)
        return (1.0 + 0.5 * r2 / 2.0) ** -2.0
    raise ValueError(f"Unknown GP kernel '{kernel_name}'. Choices: {GP_KERNEL_CHOICES}")


def _gp_layer_chol(z_grid_km, kernel_name, amplitude, length_scale,
                   cutoff_z_km, jitter: float = 1e-4) -> jnp.ndarray:
    """Rectangular non-centered GP factor ``L`` of shape ``(n_layers, n_active)``.

    A high-resolution telluric spectrum has essentially no sensitivity to the
    upper atmosphere (the absorbing column is exponentially bottom-heavy), so GP
    latents there are unconstrained. We therefore put a free GP deviation ONLY on
    the bottom layers (``z <= cutoff_z_km``). ``cutoff_z_km=None`` keeps every
    layer active (square ``L``).
    """
    if kernel_name not in GP_KERNEL_CHOICES:
        raise ValueError(f"Unknown GP kernel '{kernel_name}'. Choices: {GP_KERNEL_CHOICES}")

    def _chol(z_active):
        n_active = z_active.shape[0]
        K = (amplitude ** 2 * _kernel_matrix(z_active, kernel_name, length_scale)
             + jitter * jnp.eye(n_active))
        return jnp.linalg.cholesky(K)

    if cutoff_z_km is None:
        return _chol(z_grid_km)
    idx = np.flatnonzero(np.asarray(z_grid_km) <= cutoff_z_km)   # contiguous bottom layers
    L_active = _chol(z_grid_km[idx])
    return jnp.zeros((z_grid_km.shape[0], idx.size), dtype=L_active.dtype).at[idx, :].set(L_active)


def build_gp_factors(fc: FitConfig, z_centers):
    """Cholesky factors of the per-layer T and H2O GP priors on the altitude grid."""
    z_centers_km = jnp.asarray(np.asarray(z_centers) / 1e5)
    temp_chol_L = _gp_layer_chol(z_centers_km, fc.temp_gp_kernel, fc.temp_gp_amplitude,
                                 fc.temp_gp_length_scale_km, fc.temp_gp_cutoff_z_km)
    h2o_chol_L = _gp_layer_chol(z_centers_km, fc.h2o_gp_kernel, fc.h2o_gp_amplitude,
                                fc.h2o_gp_length_scale_km, fc.h2o_gp_cutoff_z_km)
    return temp_chol_L, h2o_chol_L
