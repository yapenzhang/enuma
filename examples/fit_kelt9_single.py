import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from enuma import FitConfig, fit_spectrum
from enuma.io.data_loader import load_h5_spectra

# --- settings ---
DATA = "KELT9_2024-08-08_raw.h5"     # input multi-exposure .h5
EXPOSURE = 10                        # which exposure (row) to fit on its own
WAVE_RANGE = (622.0, 840.0)          # nm; keeps orders fully inside
SPECIES = ("h2o", "o2")              # telluric species to model/fit
MAX_STEPS = 2000                     # max SVI steps (adaptive stop)
OUT_DIR = "kelt9_single_out"         # output directory
LOG_LEVEL = logging.INFO


def main() -> int:
    logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s %(name)s %(message)s")
    out = pathlib.Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    wmin, wmax = WAVE_RANGE

    # --- load the night, then slice out a single exposure into the dict shape
    #     the single-exposure pipeline expects (wave/flux/err/mask 2-D, scalar
    #     airmass — exactly what load_fits_spectra returns). ---
    ts = load_h5_spectra(DATA,
                         wave_range=(wmin, wmax)
                         )
    N, O, _ = ts["flux"].shape
    i = EXPOSURE
    if not (0 <= i < N):
        raise SystemExit(f"EXPOSURE {i} out of range (0..{N - 1}).")
    if O == 0:
        raise SystemExit("No orders fully inside the window — widen WAVE_RANGE.")

    # This exposure's own MJD drives the date/site reference profile (gdas);
    # build_context reduces it to one observation time (mipas ignores it).
    data = {
        "wave": ts["wave"],          # (O, P) shared grid
        "flux": ts["flux"][i],       # (O, P)
        "err":  ts["err"][i],        # (O, P)
        "mask": ts["mask"][i],       # (O, P)
        "airmass": float(ts["airmass"][i]),
        "mjd": float(ts["mjd"][i]),
    }
    print(f"Exposure {i}/{N - 1}: {O} orders in {wmin:.0f}-{wmax:.0f} nm, "
          f"airmass {data['airmass']:.3f}, baryrv {float(ts['baryrv'][i]):.2f} km/s, "
          f"mjd {data['mjd']:.5f}")


    fit_config = FitConfig(
        species=SPECIES,
        fit_species=SPECIES,
        fit_temperature=True,
        fit_continuum=True,
        fit_resolution=True,
        fit_wave_solution=True,
        max_steps=MAX_STEPS,
        reference_profile_source="gdas",
        observatory="keck",
        fit_windows=((688,690), (720,725), (755, 770), (815,820),),
        # opacity_dir = '/Users/yapzhang/Projects/telluric_jax/data/opacities_Toon/',
    )

    result = fit_spectrum(
        data, fit_config,
        save_params_json_path=str(out / "best_fit.json"),
        save_spectrum_txt_path=str(out / "best_fit.txt"),
    )
    result.plot(str(out))
    print(f"final ELBO loss = {result.losses[-1]:.1f}; outputs written to {out.resolve()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
