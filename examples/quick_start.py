"""Worked example covering common FitConfig usage patterns.

Run from the repository root:

    python examples/quick_start.py

Adjust `FITS_FILE` below to point at your data.
"""

from enuma import FitConfig, fit_telluric, load_fits_spectra


FITS_FILE = "tests/zetCMa_PRIMARY_CRIRES_SPEC1D.fits"

def example_minimal():
    """Default fit with all knobs free for the four strongest absorbers."""

    data = load_fits_spectra(FITS_FILE, 
                             wave_range=(2050, 2500)
                             )

    fit_config = FitConfig(
            species=("h2o", "co2", "n2o", "ch4", "co", "o3", "o2", "no"),
            fit_species=("h2o", "co2", "ch4", "co"),
            fit_temperature=True,
            fit_wave_solution=True,
            resolution=0.6, # R~0.6e5
            reference_profile_source="gdas",
        )
    result = fit_telluric(data, fit_config)
    result.plot()


if __name__ == "__main__":
    example_minimal()
