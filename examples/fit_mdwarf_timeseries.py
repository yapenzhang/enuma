import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from enuma import FitConfig, fit_timeseries
from enuma.cli import _setup_logging
from enuma.io.data_loader import load_fits_spectra

input_data = "SPEC_SERIES_G139-21_PRIMARY.fits"

def main():
    _setup_logging(1)
    data = load_fits_spectra(input_data, ext_flux="FLUX", ext_err="FLUX_ERR", ext_wave="WAVE",
                            #    wave_range=(wmin, wmax)
                       )
    species = ("h2o", "co2", "n2o", "ch4", "co")
    fc = FitConfig(
        fit_species=species,
        max_steps=1000,
        fit_temperature=True,   # per-exposure T profile
        fit_continuum=True,     # per-exposure blaze/throughput
        fit_resolution=True,    # shared LSF across the night
        fit_wave_solution=True,
        resolution=1.2, # x 1e5
        dry_vmr_per_exposure=False,   # else shared
        reference_profile_source="gdas",
        observatory="paranal",
        stellar_enabled=True, 
        stellar_teff=3200,
        stellar_logg = 5.0,
        rv_kms = 20.91,
    )

    # joint multi-exposure fit (shared dry VMRs/LSF/wave; per-exposure T/H2O/continuum)
    res = fit_timeseries(data, fc)

    out = pathlib.Path("mdwarf_fit_out")
    res.plot(out)

    print(f"final ELBO loss = {res.losses[-1]:.1f}; outputs written to {out.resolve()}/")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())