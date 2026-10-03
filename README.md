# enuma

*enuma* is a JAX-based forward model and fitter for telluric transmission in
high-resolution spectra. The forward model compute the line-by-line transmission spectrum 
through a layered atmosphere with flexible temperature and mixing ratio profiles informed by meterological data. 
The model is end-to-end differentiable, allowing fast fitting with NumPyro via Stochastic Variational Inference (SVI).

## Install

```bash
git clone https://github.com/yapenzhang/enuma
cd enuma
pip install -e .
```

Precomputed opacity files can be downloaded from [KEEPER](https://keeper.mpdl.mpg.de/d/923d6a75180046e68e6e/)
and placed in ``data/opacities/<species>.hdf5`` (e.g. ``h2o.hdf5``, ``co2.hdf5``, …). 

<!-- ### GPU Support

To run the model on a GPU, you must manually install the CUDA-enabled version of JAX corresponding to your system's CUDA version (e.g., CUDA 12):

```bash
pip install -U "jax[cuda12]"
```

In your code, call `enuma.set_device("gpu")` **before** loading data or constructing any JAX arrays:

```python
import enuma
enuma.set_device("gpu")  # or "cpu"

# Now proceed as normal
from enuma import FitConfig, fit_spectrum, load_fits_spectra
``` -->


## Quick start

```python
from enuma import FitConfig, fit_spectrum, load_fits_spectra

data = load_fits_spectra("zetCMa.fits", wave_range=(2050, 2500))

fc = FitConfig(
    species=("h2o", "co2", "ch4", "co", "n2o", "o2", "o3", "no"), # default
    fit_species=("h2o", "co2", "ch4", "co"),  # fit columns of these only
)

result = fit_spectrum(
    data, fc,
    save_params_json_path="best_fit.json",
    save_spectrum_txt_path="best_fit.txt",
)

result.plot()                      # writes fit_results.pdf + residuals_diagnostic.pdf
```

The best-fit parameters are saved to JSON and the best-fit spectrum to a plain text
file (wavelength, flux). To load them:

```python
from enuma import load_best_fit_params_json, generate_best_fit_spectrum

params = load_best_fit_params_json("best_fit.json")
spec = generate_best_fit_spectrum(params, result.context, result.config, obs_flux=data["flux"])
model_flux = spec["flux_model"]
```

See [`examples/quick_start.py`](examples/quick_start.py) for a worked example.

## CLI

The package ships a small `argparse`-based CLI for running.

```bash
# Run an SVI fit and write best_fit.json + best_fit.txt + the diagnostic PDFs.
python -m enuma fit zetCMa.fits \
    --species h2o,co2,ch4,co \
    --fit-species h2o,co2,ch4 \
    --out-dir ./fit_out

# Regenerate plots from a saved best-fit JSON.
python -m enuma plot zetCMa.fits --params fit_out/best_fit.json --out-dir ./fit_out
```

A YAML config mirrors `FitConfig` field-for-field; CLI flags override the YAML:

```yaml
# config.yaml
species: ["h2o", "co2", "ch4"]
fit_species: ["h2o", "co2"]
airmass: 1.4              # override a missing header value
```

```bash
python -m enuma fit zetCMa.fits --config config.yaml --out-dir ./fit_out
```

<!-- Verbosity: `-v` for INFO, `-vv` for DEBUG (default WARNING). Add `--device gpu`
to force GPU. The package follows JAX's float32 default for speed. -->
<!-- 
## What `FitConfig` controls

| Field | Meaning |
|---|---|
| `species` | All species included in the forward model (must have opacity files). |
| `fit_species` | Subset of `species` whose columns are FREE during the fit; others held at the reference VMR. `None` = fit all. |
| `fit_temperature` / `fit_resolution` / `fit_wave_solution` / `fit_continuum` | Boolean toggles for each parameter group. `False` pins at initial value. |
| `n_resolution_coeffs` / `n_wave_coeffs` | Polynomial degrees + 1. |
| `continuum_n_nodes` | Number of B-spline nodes per order. |
| `lsf_profile` / `lsf_voigt_gamma_ratio` | Instrument profile: `"gaussian"` or `"voigt"`; Voigt Lorentzian HWHM is `ratio × sigma_pix`. |
| `n_layers`, `*_gp_*`, `obs_alt_m`, … | Atmosphere / GP-prior hyperparameters. |
| `max_steps`, `learning_rate`, `saturation_mask_threshold` | Optimiser / data-prep settings. |

Pinned parameters are substituted as constants inside the NumPyro model, so they
neither appear in the AutoDelta latent space nor drift during SVI. -->

