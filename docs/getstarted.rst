.. _getstarted:

Get started
###########

*enuma* fits telluric absorption in high-resolution spectra by optimising a
differentiable physical model. The typical workflow is: load observations, build a
:class:`~enuma.config.FitConfig`, run :func:`~enuma.fit_spectrum`, and inspect or
plot the resulting :class:`~enuma.FitResult`.


Quick start
***********

.. code-block:: python

    from enuma import FitConfig, fit_spectrum, load_fits_spectra

    data = load_fits_spectra("zetCMa.fits", wave_range=(2050, 2500))

    result = fit_spectrum(
        data,
        FitConfig(
            species=("h2o", "co2", "ch4", "co", "n2o", "o2", "o3", "no"),
            fit_species=("h2o", "co2", "ch4", "co"),  # fit columns of these only
            max_steps=850,
        ),
        save_params_json_path="best_fit.json",
        save_spectrum_txt_path="best_fit.txt",
    )

    result.plot()                      # writes fit_results.pdf + residuals_diagnostic.pdf
    print(result.params.h2o_dex_dev)   # MAP H2O dex profile (GP: column + shape)
    print(result.params.dry_vmr_scalars)  # MAP dry-species columns (dex offset)


Configuring the fit
*******************

:class:`~enuma.config.FitConfig` is the configuration hub: it selects the species
in the forward model, which parameter groups are free vs. pinned (temperature,
resolution, wavelength solution, continuum), the atmosphere / GP-prior structure,
the instrument LSF profile, and the optimiser knobs. See the :ref:`api` reference for the
full field list.


Limiting CPU usage
*******************

On CPU, JAX uses every core by default. To cap it, call
:func:`enuma.set_num_threads` before the first computation, or export
``ENUMA_NUM_THREADS`` before starting Python (the CLI also takes ``--threads``):

.. code-block:: python

    import enuma
    enuma.set_num_threads(4)  # at most ~400% CPU


Best-fit I/O
************

The best-fit parameters are saved to JSON and the best-fit spectrum to a plain text
file (wavelength, flux). To load them:

.. code-block:: python

    from enuma import (
        load_best_fit_params_json,
        generate_best_fit_spectrum,
    )

    params = load_best_fit_params_json("best_fit.json")
    spec = generate_best_fit_spectrum(params, result.context, result.config,
                                      obs_flux=data["flux"])
    model_flux = spec["flux_model"]


Command-line interface
**********************

The package ships a small ``argparse``-based CLI for running fits without writing
Python.

.. code-block:: console

    # Run an SVI fit and write best_fit.json + best_fit.txt + the diagnostic PDFs.
    $ python -m enuma fit zetCMa.fits \
        --species h2o,co2,ch4,co \
        --fit-species h2o,co2,ch4 \
        --steps 900 \
        --out-dir ./fit_out

    # Time the forward model on the input spectrum (warmup + N iters).
    $ python -m enuma bench zetCMa.fits --species h2o,co2 --iters 50

    # Regenerate plots from a saved best-fit JSON.
    $ python -m enuma plot zetCMa.fits --params fit_out/best_fit.json --out-dir ./fit_out

A YAML config mirrors :class:`~enuma.config.FitConfig` field-for-field; CLI flags
override the YAML:

.. code-block:: yaml

    # config.yaml
    species: ["h2o", "co2", "ch4"]
    fit_species: ["h2o", "co2"]
    max_steps: 800
    learning_rate: 0.001
    airmass: 1.4              # override a missing/wrong header value

.. code-block:: console

    $ python -m enuma fit zetCMa.fits --config config.yaml --out-dir ./fit_out

Verbosity: ``-v`` for INFO, ``-vv`` for DEBUG (default WARNING). Add ``--device
gpu`` to force GPU.


