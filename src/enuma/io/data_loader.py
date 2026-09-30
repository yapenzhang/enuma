import logging
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from astropy.io import fits

__all__ = [
    "load_fits_spectra",
    "load_h5_spectra",
    "validate_spectrum_shapes",
    "validate_timeseries_shapes",
]

logger = logging.getLogger("enuma.io.data_loader")


def _to_float(value) -> Optional[float]:
    """``float(value)`` or None (e.g. a sexagesimal string like '20:31:26.40')."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _header_float(header, *keys) -> Optional[float]:
    """First of ``keys`` present in the header that parses as a float, else None."""
    for key in keys:
        if key in header:
            val = _to_float(header[key])
            if val is not None:
                return val
    return None


def _header_pair_mean(header, start_key, end_key) -> Optional[float]:
    """Mean of a START/END header pair (e.g. ESO ambient cards), or None."""
    start = _header_float(header, start_key)
    end = _header_float(header, end_key)
    return 0.5 * (start + end) if start is not None and end is not None else None


def _read_header_airmass(header) -> Optional[float]:
    """Airmass from the standard ``AIRMASS`` card, else the ESO START/END mean."""
    direct = _header_float(header, "AIRMASS")
    if direct is not None:
        return direct
    return _header_pair_mean(header, "HIERARCH ESO TEL AIRM START", "HIERARCH ESO TEL AIRM END")


def _read_header_pwv_mm(header) -> Optional[float]:
    """Zenith PWV in mm: ESO AMBI IWV START/END mean, else PWV/IWV.
    """
    pair = _header_pair_mean(header, "HIERARCH ESO TEL AMBI IWV START", "HIERARCH ESO TEL AMBI IWV END")
    if pair is not None:
        return pair
    return _header_float(header, "PWV", "IWV")


def _read_header_target_radec(header) -> Tuple[Optional[float], Optional[float]]:
    """(target_ra_deg, target_dec_deg) in ICRS degrees from the ``RA``/``DEC``
    cards, or ``(None, None)``.

    Numeric cards are decimal degrees (e.g. ESO ``RA = 95.077239``). String cards
    are sexagesimal with RA in *hours* and Dec in degrees (e.g. Keck
    ``'20:31:26.40'`` / ``'+39:56:20.0'``); the ``[h]``/``[deg]`` comment on the RA
    card is used as a tie-breaker. Parsing goes through ``astropy.SkyCoord`` so the
    units are handled correctly.
    """
    if "RA" not in header or "DEC" not in header:
        return None, None
    ra_val, dec_val = header["RA"], header["DEC"]
    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord

        ra_num, dec_num = _to_float(ra_val), _to_float(dec_val)
        if ra_num is not None and dec_num is not None:
            sc = SkyCoord(ra=ra_num * u.deg, dec=dec_num * u.deg, frame="icrs")
        else:
            # Sexagesimal strings: RA in hours unless the comment flags degrees.
            ra_comment = str(header.comments["RA"]).lower()
            ra_unit = u.deg if "deg" in ra_comment else u.hourangle
            sc = SkyCoord(ra=str(ra_val), dec=str(dec_val),
                          unit=(ra_unit, u.deg), frame="icrs")
        return float(sc.ra.deg), float(sc.dec.deg)
    except Exception as exc:
        logger.warning("Could not parse RA/DEC (%r, %r): %s", ra_val, dec_val, exc)
        return None, None


def _read_header_mjd(header) -> Optional[float]:
    """Observation time as MJD (UTC) from ``MJD-OBS``, or ``None`` (with a warning).
    """
    mjd = _header_float(header, "MJD-OBS")
    if mjd is None:
        logger.warning("No MJD-OBS in header; observation time unavailable "
                       "(barycentric RV seed / reanalysis profile will degrade).")
    return mjd


def _mask_and_normalize(raw_flux, raw_wave_b, raw_err, *, wave_mask=None):
    """Validity mask + 95th-percentile continuum normalisation, rank-agnostic."""
    valid = (~np.isnan(raw_flux)) & (~np.isnan(raw_err)) & (~np.isnan(raw_wave_b)) & (raw_err > 0)
    for (w_min, w_max) in (wave_mask or []):
        valid[(raw_wave_b >= w_min) & (raw_wave_b <= w_max)] = False

    # TODO: temporary hack
    valid[..., :10] = False
    valid[..., -10:] = False

    norm = np.nanquantile(raw_flux, 0.95, axis=-1, keepdims=True)
    norm = np.where(norm > 0, norm, 1.0)
    clean_flux = np.nan_to_num(raw_flux / norm)
    clean_err = np.where(valid, raw_err / norm, np.inf)
    return clean_flux, clean_err, valid


def load_fits_spectra(
        file_path: str,
        wave_range: Tuple[float, float] = None,
        wave_mask: List[Tuple[float, float]] = None,
        ext_flux: str = "FLUX",
        ext_err: str = "FLUX_ERR",
        ext_wave: str = "WAVE",
        ext_mjd: str = "MJD",
        ext_airmass: str = "AIRMASS",
    ) -> Dict[str, np.ndarray]:
    """Load multi-order spectra from a FITS file — one exposure or timeseries.

    Args:
        file_path: Path to the FITS file.
        wave_range: Optional (min, max) nm; keeps only orders fully inside it.
        wave_mask: Optional list of (min, max) nm intervals to mask out.
        ext_flux / ext_err / ext_wave: FLUX/ERR/WAVE extensions (index or name).
        ext_mjd / ext_airmass: per-exposure MJD/AIRMASS extensions (night only).

    Returns:
        Dict of sanitised arrays — ``wave`` ``(O, P)``, ``flux``/``err``/``mask``
        ``(O, P)`` for one exposure or ``(N, O, P)`` for a night — plus observation
        metadata (``airmass``/``mjd`` scalar or ``(N,)``, ``pwv_mm``, ``target_ra``,
        ``target_dec``; None when absent).
    """
    with fits.open(file_path) as hdul:
        header = hdul[0].header
        raw_flux = np.asarray(hdul[ext_flux].data, dtype=np.float64)
        raw_wave = np.asarray(hdul[ext_wave].data, dtype=np.float64)   # (O, P), shared
        try:
            raw_err = np.asarray(hdul[ext_err].data, dtype=np.float64)
        except (IndexError, KeyError):
            raw_err = np.ones_like(raw_flux)                           # uniform noise
        target_ra, target_dec = _read_header_target_radec(header)
        extra = {"target_ra": target_ra, "target_dec": target_dec,
                 "pwv_mm": _read_header_pwv_mm(header)}
        series = raw_flux.ndim == 3
        if series:    # per-exposure airmass/MJD live in their own extensions
            airmass = np.asarray(hdul[ext_airmass].data, dtype=np.float64)   # (N,)
            mjd = np.asarray(hdul[ext_mjd].data, dtype=np.float64)           # (N,)
        else:         # one exposure: scalar airmass/MJD from the PRIMARY header
            airmass = _read_header_airmass(header)
            mjd = _read_header_mjd(header)

    # Order axis is -2 in both the (O, P) single and (N, O, P) night layouts.
    if wave_range is not None:
        keep = np.where((raw_wave[:, 0] >= wave_range[0]) & (raw_wave[:, -1] <= wave_range[1]))[0]
        if keep.size == 0:
            raise ValueError(f"No orders fully inside wave_range {wave_range} nm; widen it.")
        raw_wave = raw_wave[keep]
        raw_flux, raw_err = np.take(raw_flux, keep, axis=-2), np.take(raw_err, keep, axis=-2)

    # The shared (O, P) wave grid broadcasts over exposures for a night (no-op for one).
    clean_flux, clean_err, valid_mask = _mask_and_normalize(
        raw_flux, np.broadcast_to(raw_wave, raw_flux.shape), raw_err, wave_mask=wave_mask)
    total = valid_mask.size
    masked = total - int(np.sum(valid_mask))
    scope = (f"{raw_flux.shape[0]} exposures x {raw_wave.shape[0]}" if series
             else f"{raw_wave.shape[0]}")
    logger.info("Loaded %s orders. Masked %d/%d pixels (%.1f%%).",
                scope, masked, total, masked / total * 100)

    data = {"wave": raw_wave, "flux": clean_flux, "err": clean_err, "mask": valid_mask,
            "airmass": airmass, "mjd": mjd, **extra}
    (validate_timeseries_shapes if series else validate_spectrum_shapes)(data)
    return data


def load_h5_spectra(
    file_path: str,
    wave_range: Tuple[float, float] = None,
    wave_mask: List[Tuple[float, float]] = None,
) -> Dict[str, np.ndarray]:
    """Load one night of multi-exposure spectra from an HDF5 file.

    Expected datasets (wavelengths in nm):
      * ``flux`` (N_exp, N_orders, N_pixels), ``err`` same shape,
      * ``wave`` (N_orders, N_pixels) — a single grid SHARED by all exposures,
      * ``airmass`` (N_exp,)  — the time-series leverage,
      * ``baryrv``  (N_exp,)  — per-exposure barycentric RV (km/s),
      * ``mjd``     (N_exp,)  — optional per-exposure MJD (UTC); its median is the
        representative time for the date/site reanalysis-profile fetch (gdas),
      * ``phase``   (N_exp,)  — optional orbital phase (carried through, unused).
    """
    with h5py.File(file_path, "r") as f:
        raw_flux = np.asarray(f["flux"][:], dtype=np.float64)      # (N, O, P)
        raw_wave = np.asarray(f["wave"][:], dtype=np.float64)      # (O, P), shared
        raw_err = (np.asarray(f["err"][:], dtype=np.float64)
                   if "err" in f else np.ones_like(raw_flux))
        airmass = np.asarray(f["airmass"][:], dtype=np.float64)    # (N,)
        baryrv = (np.asarray(f["baryrv"][:], dtype=np.float64)
                  if "baryrv" in f else None)
        mjd = np.asarray(f["mjd"][:], dtype=np.float64) if "mjd" in f else None  # (N,)
        phase = np.asarray(f["phase"][:], dtype=np.float64) if "phase" in f else None

    if wave_range is not None:
        keep = np.where((raw_wave[:, 0] >= wave_range[0]) & (raw_wave[:, -1] <= wave_range[1]))[0]
        raw_wave, raw_flux, raw_err = raw_wave[keep], raw_flux[:, keep, :], raw_err[:, keep, :]
        if raw_wave.shape[0] == 0:
            raise ValueError(f"No orders fully inside wave_range {wave_range} nm; widen it.")

    clean_flux, clean_err, valid_mask = _mask_and_normalize(
        raw_flux, np.broadcast_to(raw_wave, raw_flux.shape), raw_err, wave_mask=wave_mask)
    total = valid_mask.size
    masked = total - int(np.sum(valid_mask))
    logger.info("Loaded %d exposures x %d orders. Masked %d/%d pixels (%.1f%%).",
                raw_flux.shape[0], raw_wave.shape[0], masked, total, masked / total * 100)

    out = {"wave": raw_wave, "flux": clean_flux, "err": clean_err, "mask": valid_mask,
           "airmass": airmass}
    if baryrv is not None:
        out["baryrv"] = baryrv                                 # (N,) km/s
    if mjd is not None:
        out["mjd"] = mjd                                       # (N,) per-exposure MJD (UTC)
    if phase is not None:
        out["phase"] = phase
    validate_timeseries_shapes(out)
    return out


def _shape(data: dict, key: str) -> tuple:
    return np.asarray(data[key]).shape


def validate_spectrum_shapes(data: dict) -> None:
    """Assert a single-exposure dict has wave/flux/err/mask all ``(O, P)``."""
    wave_shape = _shape(data, "wave")
    if len(wave_shape) != 2:
        raise ValueError(f"Single-exposure 'wave' must be (O, P); got {wave_shape}.")
    for key in ("flux", "err", "mask"):
        if key in data and _shape(data, key) != wave_shape:
            raise ValueError(f"'{key}' {_shape(data, key)} must match 'wave' {wave_shape} (O, P).")


def validate_timeseries_shapes(data: dict) -> None:
    """Assert a night dict has wave ``(O, P)`` and flux/err/mask ``(N, O, P)``."""
    wave_shape = _shape(data, "wave")
    if len(wave_shape) != 2:
        raise ValueError(f"Time-series 'wave' must be (O, P) shared grid; got {wave_shape}.")
    flux_shape = _shape(data, "flux")
    if len(flux_shape) != 3 or flux_shape[1:] != wave_shape:
        raise ValueError(
            f"Time-series 'flux' must be (N, O, P) with (O, P)={wave_shape}; got {flux_shape}.")
    for key in ("err", "mask"):
        if key in data and _shape(data, key) != flux_shape:
            raise ValueError(f"'{key}' {_shape(data, key)} must be (N, O, P)={flux_shape}.")
    if "airmass" in data and _shape(data, "airmass") != (flux_shape[0],):
        raise ValueError(f"'airmass' {_shape(data, 'airmass')} must be (N,)=({flux_shape[0]},).")
