"""End-to-end single-exposure telluric fit on a CRIRES+ standard-star spectrum.

The canonical smoke run of the high-level ``fit_spectrum`` API: load the FITS,
build a ``FitConfig``, run the fit, and write the standard diagnostic PDFs plus
the best-fit JSON/TXT into an output directory. The spectrum and the molecfit
comparison both live in ``tests/``.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from enuma import FitConfig, fit_spectrum, load_fits_spectra
from enuma.cli import _setup_logging

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "tests"

# import enuma
# enuma.set_num_threads(1)

def main() -> int:
    _setup_logging(1)
    out = pathlib.Path("bench_fit_out")
    out.mkdir(parents=True, exist_ok=True)

    data = load_fits_spectra(str(_DATA_DIR / "zetCMa_PRIMARY_CRIRES_SPEC1D.fits"),
                             wave_range=(2050, 2500))
    fit_config = FitConfig(
        fit_species=("h2o", "co2", "n2o", "ch4", "co"),
        # Observatory: an astropy site name (EarthLocation.of_site); supplies
        # lat/lon/altitude for the barycentric RV and the gdas fetch.
        observatory="paranal",
        # Reference atmosphere: 'mipas' (default) or 'gdas' / 'ecmwf'
        # to pull T/P/H2O from a date-/site-specific column. With source='gdas'
        # and no explicit path, setup_model auto-fetches using the FITS MJD-OBS +
        # FitConfig.observatory lat/lon, caching under ~/.cache/enuma/gdas/.
        reference_profile_source="gdas",
        fit_temperature=True,
        fit_wave_solution=True,
        resolution=0.6,
        # opacity_dir="/Users/yapzhang/Projects/enuma/data/opacities/",
    )
    result = fit_spectrum(
        data, fit_config,
        save_params_json_path=str(out / "best_fit.json"),
        save_spectrum_txt_path=str(out / "best_fit.txt"),
    )
    result.plot(
        str(out),
        # comparison_fits=str(_DATA_DIR / "TELLURIC_zetCMa.fits"),
    )
    print(f"final ELBO loss = {result.losses[-1]:.1f}; outputs written to {out.resolve()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
