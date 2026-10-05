"""PHOENIX-NewEra stellar template: fetch + cache + normalise, plus the
JAX-differentiable rotational broadening and RV shift applied at run time.
"""

import logging
import pathlib
import shutil
import urllib.error
import urllib.request

import h5py
import jax.numpy as jnp
import numpy as np

from enuma.config import FitConfig
from enuma.constants import C_KMS, MATMUL_PRECISION
from enuma.model._paths import cache_root

__all__ = [
    "STELLAR_GRID_CHOICES",
    "fetch_phoenix_spectrum",
    "normalize_template",
    "build_stellar_template",
    "build_stellar_from_config",
    "rotational_kernel",
    "apply_stellar",
]

logger = logging.getLogger("enuma.model.stellar")

STELLAR_GRID_CHOICES = ("phoenix-newera",)

# Hamburg FDR records hosting the PHOENIX-NewEra HSR HDF5 files. ``_newera_url``
# picks the V3 record for Teff >= 5000 K and falls back to the "additional" record.
_NEWERA_FDR = "https://www.fdr.uni-hamburg.de/record/16738/files/"
_NEWERA_FDR_V3 = "https://www.fdr.uni-hamburg.de/record/17670/files/"
_NEWERA_FDR_ADD = "https://www.fdr.uni-hamburg.de/record/17936/files/"
_NEWERA_SUFFIX = ".PHOENIX-NewEra-ACES-COND-2023.HSR.h5"

# Grid node spacing used to snap requested stellar parameters to the nearest
# available model (NewEra: Teff 2300-12000 K, logg 0-6, [M/H] -4..+0.5).
_TEFF_STEP = 100.0
_LOGG_STEP = 0.5
_FEH_STEP = 0.5


# ---------------------------------------------------------------------------
# Fetch + cache (setup time)
# ---------------------------------------------------------------------------

def _snap(value: float, step: float) -> float:
    """The (teff, logg, feh, alpha) request is snapped to the nearest grid node."""
    return round(value / step) * step


def _newera_filename(teff: float, logg: float, zscale: float, alpha: float) -> str:
    job = f"lte{teff:0=5.0f}{-logg:3.2f}"
    job += f"{zscale:0=+4.1f}" if zscale != 0.0 else "-" + f"{zscale:0=3.1f}"
    if alpha != 0.0:
        job += ".alpha=" + f"{alpha:0=+3.1f}"
    return job + _NEWERA_SUFFIX


def _newera_urls(teff: float, filename: str):
    primary = _NEWERA_FDR_V3 if teff >= 5000.0 else _NEWERA_FDR
    return [primary + filename + "?download=1", _NEWERA_FDR_ADD + filename + "?download=1"]


def _download(url: str, dest: pathlib.Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": "enuma"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out, length=1 << 20)
            tmp.replace(dest)
        return True
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise


def fetch_phoenix_spectrum(teff: float, logg: float, feh: float = 0.0,
                           alpha: float = 0.0, *, grid: str = "phoenix-newera",
                           wave_min_nm: float = None, wave_max_nm: float = None,
                           cache_dir: str = None, force_refresh: bool = False):
    """Fetch (and cache) one PHOENIX-NewEra HSR spectrum, sliced to a wave range."""
    if grid not in STELLAR_GRID_CHOICES:
        raise ValueError(f"Unknown stellar grid '{grid}'. Choices: {STELLAR_GRID_CHOICES}")

    t, g, z, a = _snap(teff, _TEFF_STEP), _snap(logg, _LOGG_STEP), _snap(feh, _FEH_STEP), alpha
    if (t, g, z) != (teff, logg, feh):
        logger.info("Snapped stellar params to NewEra grid node: "
                    "Teff %.0f→%.0f, logg %.2f→%.2f, [M/H] %.2f→%.2f.",
                    teff, t, logg, g, feh, z)
    filename = _newera_filename(t, g, z, a)
    dest = cache_root(cache_dir, subdir="phoenix-newera") / filename

    if not dest.exists() or force_refresh:
        urls = _newera_urls(t, filename)
        logger.info("Fetching PHOENIX-NewEra %s ...", filename)
        if not any(_download(url, dest) for url in urls):
            raise FileNotFoundError(
                f"PHOENIX-NewEra model '{filename}' not found "
                f"(tried {len(urls)} records). Check that the (Teff, logg, [M/H], "
                f"[alpha/Fe]) node exists.")
        logger.info("  → %s  (%.1f MB)", dest, dest.stat().st_size / 1e6)
    else:
        logger.debug("PHOENIX-NewEra cache hit: %s", dest)

    with h5py.File(dest, "r") as f:
        spec = f["PHOENIX_SPECTRUM"]
        wl_ang = spec["wl"][:]                       # vacuum Ångström (monotonic)
        if wave_min_nm is not None and wave_max_nm is not None:
            lo = int(np.searchsorted(wl_ang, wave_min_nm * 10.0))
            hi = int(np.searchsorted(wl_ang, wave_max_nm * 10.0, side="right"))
            if lo >= hi:
                raise ValueError(
                    f"No NewEra coverage in {wave_min_nm}-{wave_max_nm} nm.")
            sl = slice(lo, hi)
        else:
            sl = slice(None)
        log_flux = spec["flux"][sl]                  # log10(F_lambda) [erg/s/cm²/cm]
        wave_nm = wl_ang[sl] * 0.1                    # Ångström → nm

    flux = np.power(10.0, log_flux, dtype=np.float64)
    return wave_nm, flux


# ---------------------------------------------------------------------------
# Normalise + resample onto the model grid (setup time)
# ---------------------------------------------------------------------------

def normalize_template(wave_nm: np.ndarray, flux: np.ndarray,
                       bin_size: float = 5.0,
                       percentile: float = 99.0) -> np.ndarray:
    """Normalize the flux by taking a high percentile in segments ``bin_size`` wide 
    (same units as ``wave_nm``, i.e. nm), and linearly interpolate the continuum 
    back onto every pixel. 
    """
    n = flux.shape[0]
    span = float(np.ptp(wave_nm))
    n_bins = max(1, int(round(span / bin_size)))
    if n < n_bins * 4:
        cont = np.full_like(flux, np.percentile(flux, percentile))
        return flux / np.maximum(cont, 1e-30)

    edges = np.linspace(0, n, n_bins + 1).astype(int)
    centers = 0.5 * (wave_nm[edges[:-1]] + wave_nm[np.clip(edges[1:], 0, n - 1)])
    node_vals = np.array([np.percentile(flux[edges[i]:edges[i + 1]], percentile)
                          for i in range(n_bins)])
    cont = np.interp(wave_nm, centers, node_vals)
    
    return flux / np.maximum(cont, 1e-30)


def build_stellar_template(wave_nm: np.ndarray, flux: np.ndarray,
                           model_wave) -> tuple:
    """Resample the normalised template onto the model grid ``model_wave``."""
    flux_norm = normalize_template(wave_nm, flux)
    mw = np.asarray(model_wave, dtype=np.float64)
    template = np.interp(mw, wave_nm, flux_norm, left=1.0, right=1.0)
    dv_kms = float(np.median(C_KMS * np.gradient(mw) / mw))
    return jnp.asarray(template, dtype=jnp.float32), dv_kms


def build_stellar_from_config(fc: FitConfig, model_wave, cache_dir: str = None):
    """Fetch + resample the PHOENIX-NewEra template selected by ``fc`` onto
    ``model_wave``. Returns ``(template, dv_kms)``; a no-op pair when stellar is off.
    """
    if not fc.stellar_enabled:
        return jnp.zeros(1), jnp.asarray(0.0)
    if fc.stellar_grid not in STELLAR_GRID_CHOICES:
        raise ValueError(f"Unknown stellar_grid '{fc.stellar_grid}'. Choices: {STELLAR_GRID_CHOICES}")
    if fc.stellar_teff is None or fc.stellar_logg is None:
        raise ValueError("stellar_enabled=True requires stellar_teff and stellar_logg.")
    mw = np.asarray(model_wave)
    pad = (mw[-1] - mw[0]) * 0.01
    s_wave, s_flux = fetch_phoenix_spectrum(
        fc.stellar_teff, fc.stellar_logg, fc.stellar_feh, fc.stellar_alpha,
        grid=fc.stellar_grid,
        wave_min_nm=float(mw[0]) - pad, wave_max_nm=float(mw[-1]) + pad,
        cache_dir=cache_dir)
    stellar_template, stellar_dv_kms = build_stellar_template(s_wave, s_flux, model_wave)
    logger.info("Stellar template: PHOENIX NewEra Teff=%.0f logg=%.2f [M/H]=%.2f",
                fc.stellar_teff, fc.stellar_logg, fc.stellar_feh)
    return stellar_template, jnp.asarray(stellar_dv_kms)


# ---------------------------------------------------------------------------
# Rotational broadening + RV shift (run time, differentiable)
# ---------------------------------------------------------------------------

def rotational_kernel(vsini_kms, dv_kms, half_width: int, epsilon: float = 0.6):
    """Normalised Gray (2005) rotational broadening kernel on a velocity grid.

    Sampled at fixed pixel offsets ``k·dv_kms`` over ``[-half_width, half_width]``
    (``half_width`` static so the kernel shape is jit-friendly), zero outside
    ``|Δv| < vsini``, then normalised to unit sum. 
    """
    k = jnp.arange(-half_width, half_width + 1)
    x = (k * dv_kms) / jnp.maximum(vsini_kms, 1e-6)
    one_minus_x2 = 1.0 - x * x
    inside = one_minus_x2 > 0.0
    safe = jnp.where(inside, one_minus_x2, 1.0)
    g = 2.0 * (1.0 - epsilon) * jnp.sqrt(safe) + 0.5 * jnp.pi * epsilon * safe
    g = jnp.where(inside, g, 0.0)
    return g / jnp.sum(g)


def apply_stellar(template, dv_kms, rv_kms, vsini_kms,
                  half_width: int, epsilon: float = 0.6):
    """Rotationally broaden and Doppler-shift stellar template.

    The opacity grid is constant-resolution, i.e. uniform
    in ``ln λ`` with a fixed step ``dv_kms / C_KMS`` per pixel, so the Doppler
    shift is a single fractional-pixel shift shared by every pixel
    """
    kernel = rotational_kernel(vsini_kms, dv_kms, half_width, epsilon)
    broadened = 1.0 + jnp.convolve(template - 1.0, kernel, mode="same",
                                   precision=MATMUL_PRECISION)

    n = broadened.shape[0]
    # Net pixel shift on the log-uniform grid. Positive rv ⇒ redshift ⇒ each
    # observed pixel samples a bluer (lower-index) source pixel.
    s = C_KMS * jnp.log1p(rv_kms / C_KMS) / dv_kms
    s_floor = jnp.floor(s)
    s_frac = s - s_floor                      
    s_int = s_floor.astype(jnp.int32)

    idx = jnp.arange(n)
    hi = idx - s_int                           # upper neighbour, weight (1 - s_frac)
    lo = hi - 1                                # lower neighbour, weight s_frac
    valid = (lo >= 0) & (hi <= n - 1)
    flux = (s_frac * broadened[jnp.clip(lo, 0, n - 1)]
            + (1.0 - s_frac) * broadened[jnp.clip(hi, 0, n - 1)])
    return jnp.where(valid, flux, 1.0)
