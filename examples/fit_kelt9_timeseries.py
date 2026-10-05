import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import dataclasses
import numpy as np

from enuma import FitConfig, fit_timeseries
from enuma.io.data_loader import load_h5_spectra
from enuma.model.forward import forward_model_batched

# --- settings ---
DATA = "KELT9_2024-08-08_raw.h5"     # input multi-exposure .h5
WAVE_RANGE = (685.0, 773.0)          # nm; keeps orders fully inside  (685,773, 695)
SPECIES = ("h2o", "o2")              # h2o is per-exposure; the rest share a column
MAX_STEPS = 600                      # max SVI steps (adaptive stop)
STELLAR_TEFF = None                  # stellar template Teff (K); None disables it
STELLAR_LOGG = 4.5                   # stellar log g (cgs)
VSINI = 111.0                        # stellar vsini (km/s)
OUT_DIR = "kelt9_fit_out"            # output directory
LOG_LEVEL = logging.INFO


# def _subsample(data: dict, stride: int) -> dict:
#     if stride <= 1:
#         return data
#     sl = slice(None, None, stride)
#     out = dict(data)
#     for k in ("flux", "err", "mask", "airmass", "baryrv", "mjd", "phase"):
#         if k in data:
#             out[k] = data[k][sl]
#     return out


def main() -> int:
    logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s %(name)s %(message)s")
    out = pathlib.Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    wmin, wmax = WAVE_RANGE

    # --- load one night, restrict to the window, optionally thin exposures ---
    data = load_h5_spectra(DATA, wave_range=(wmin, wmax))
    N, O, _ = data["flux"].shape
    print(f"{N} exposures x {O} orders in {wmin:.0f}-{wmax:.0f} nm; "
          f"airmass {data['airmass'].min():.2f}-{data['airmass'].max():.2f}, "
          f"baryrv {data['baryrv'].min():.2f}-{data['baryrv'].max():.2f} km/s")
    if O == 0:
        raise SystemExit("No orders fully inside the window — widen WAVE_RANGE.")

    # --- configure the fit ---
    fc = FitConfig(
        species=SPECIES, fit_species=SPECIES,
        fit_temperature=True,   # per-exposure T profile
        fit_continuum=True,     # per-exposure blaze/throughput
        fit_resolution=True,    # shared LSF across the night
        fit_wave_solution=False,                   # shared wave solution
        dry_vmr_per_exposure=True,        # else shared (airmass lever)
        max_steps=MAX_STEPS,
        fit_windows=((688,690), (720,725), (755, 770), (815,820),),
        reference_profile_source="gdas",   # uses obs_date_utc (above) + site (below)
        observatory="keck",
    )
    if STELLAR_TEFF is not None:
        # Hot fast rotator: widen the rotation kernel to cover vsini; fix rv/vsini
        # (a near-featureless A0 window can't constrain them).
        fc = dataclasses.replace(
            fc, stellar_enabled=True, stellar_teff=STELLAR_TEFF,
            stellar_logg=STELLAR_LOGG, vsini=VSINI,
            fit_rv=False, fit_vsini=False, stellar_rot_half_width=401)

    # --- joint multi-exposure fit (shared dry VMRs/LSF/wave; per-exposure T/H2O/continuum) ---
    res = fit_timeseries(data, fc)

    # --- save best-fit sites + model/telluric cubes + diagnostic ---
    np.savez(out / "sites.npz",
             **{k: np.asarray(v) for k, v in res.sites.items()},
             airmass=data["airmass"], baryrv=data["baryrv"])
    full = np.asarray(forward_model_batched(res.params, res.context, res.config, res.layout))
    tell = np.asarray(forward_model_batched(
        res.params, res.context, res.config._replace(stellar_enabled=False), res.layout))
    np.savez(out / "model.npz", wave=data["wave"], obs=np.asarray(data["flux"]),
             full=full, telluric=tell, mask=np.asarray(res.obs_mask),
             airmass=data["airmass"])
    res.plot(out)   # full diagnostic suite (disentangle, profiles, parameters, LSF, losses)

    # The shared dry-species columns are the airmass-leveraged quantities.
    # for s in res.dry_species:
    #     dex = (np.asarray(res.sites[f"dry_vmr_{s}"]))
    #     print(dex)
    # print(f"final ELBO loss = {res.losses[-1]:.1f}; outputs written to {out.resolve()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
