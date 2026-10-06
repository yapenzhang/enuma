.. _getstarted:

Get started
###########

*enuma* fits telluric absorption in high-resolution spectra by optimising a
differentiable physical model. The typical workflow is: load observations, build a
:class:`~enuma.config.FitConfig`, run :func:`~enuma.inference.fit.fit_spectrum` (single exposure) or
:func:`~enuma.inference.fit.fit_timeseries` (a sequence of exposures), and inspect or plot the
resulting :class:`~enuma.inference.fit.FitResult`.

Before running a fit, download the opacity grids (see :ref:`data`).


Quick start
***********

.. code-block:: python

    from enuma import FitConfig, fit_spectrum, load_fits_spectra

    data = load_fits_spectra("zetCMa.fits", wave_range=(2050, 2500))

    result = fit_spectrum(
        data,
        FitConfig(
            species=("h2o", "co2", "ch4", "co", "n2o"),
            fit_species=("h2o", "co2", "ch4", "co"),  # fit columns of these only
            observatory="paranal",                   # required: an astropy site name
            reference_profile_source="gdas",   # auto-fetched from MJD-OBS + site
            max_steps=850,
        ),
        save_params_json_path="best_fit.json",
        save_spectrum_txt_path="best_fit.txt",
    )

    result.plot("fit_out")                 # writes the diagnostic PDFs


:example:`quick_start.py` runs this fit end-to-end on a CRIRES+ K-band spectrum of
a standard star, using the date-specific GDAS atmosphere. :ref:`outputs` explains
what the fitted parameters, saved files and diagnostic plots contain.


.. _input-data:

Input data
**********

Every fit takes a plain ``dict`` of NumPy arrays. A single exposure and a time
series dataset differ only in the shapes: ``O`` is the
number of spectral orders, ``P`` the number of pixels per order, and ``N`` the number of
exposures.

.. list-table::
   :header-rows: 1
   :widths: 14 22 26 38

   * - Key
     - Single exposure
     - Time series
     - Meaning
   * - ``wave``
     - ``(O, P)``
     - ``(O, P)``, shared by all exposures
     - Wavelength (nm) of each pixel.
   * - ``flux``
     - ``(O, P)``
     - ``(N, O, P)``
     - Continuum-normalised flux (roughly 1 outside absorption lines).
   * - ``err``
     - ``(O, P)``
     - ``(N, O, P)``
     - Flux uncertainty, on the same scale as ``flux``.
   * - ``mask``
     - ``(O, P)``, bool
     - ``(N, O, P)``, bool
     - ``True`` for valid pixels.
   * - ``airmass``
     - scalar
     - ``(N,)``
     - Airmass of the observation.
   * - ``mjd``
     - scalar, optional
     - ``(N,)``, optional
     - Observation time (MJD, UTC). Needed for the GDAS profile and the barycentric
       velocity.
   * - ``baryrv``
     - n/a
     - ``(N,)``, optional
     - Barycentric RV term (km/s). Derived from ``target_ra``/``target_dec``,
       ``mjd`` and the observatory if absent.
   * - ``target_ra``, ``target_dec``, ``pwv_mm``
     - optional
     - optional
     - Target ICRS coordinates (deg) and onsite precipitable water vapour (mm).

Loading spectra
---------------

Two loaders produce this dict:

* :func:`~enuma.io.data_loader.load_fits_spectra` reads a FITS file with ``FLUX``, ``FLUX_ERR``
  and ``WAVE`` extensions (names configurable with ``ext_flux``/``ext_err``/
  ``ext_wave``). A 2-D ``FLUX`` gives a single exposure; a 3-D ``FLUX`` gives a
  time series, with per-exposure ``MJD`` and ``AIRMASS`` extensions. Metadata
  comes from the primary header: airmass (``AIRMASS``, or the mean of the ESO
  ``TEL AIRM START/END`` cards), ``MJD-OBS``, ``RA``/``DEC``, and PWV (the ESO
  ``TEL AMBI IWV START/END`` cards, or ``PWV``/``IWV``).
* :func:`~enuma.io.data_loader.load_h5_spectra` reads one night from an HDF5 file with datasets
  ``flux`` ``(N, O, P)``, ``err`` (optional), ``wave`` ``(O, P)``, ``airmass``
  ``(N,)`` and the optional ``baryrv``, ``mjd`` and ``phase`` ``(N,)``.

Both loaders:

* keep only the orders lying fully inside ``wave_range=(min_nm, max_nm)``;
* mask NaNs, non-positive errors, any ``wave_mask=[(min_nm, max_nm), ...]``
  intervals, and the 10 pixels at each order edge;
* normalise each order (and exposure) by its 95th-percentile flux.

Building the dict yourself
--------------------------

For other formats, build the dict directly. Normalise the flux to a continuum
of about 1 (the continuum prior and the saturation mask assume it) and replace
NaNs, which the mask then excludes. 
.. Check the shapes with
.. :func:`~enuma.io.data_loader.validate_spectrum_shapes` or
.. :func:`~enuma.io.data_loader.validate_timeseries_shapes`:

.. code-block:: python

    import numpy as np
    from enuma.io.data_loader import validate_timeseries_shapes

    norm = np.nanpercentile(flux, 95, axis=-1, keepdims=True)
    data = {
        "wave": wave,                                     # (O, P) nm
        "flux": np.nan_to_num(flux / norm),               # (N, O, P)
        "err": np.nan_to_num(err / norm, nan=np.inf),     # (N, O, P)
        "mask": np.isfinite(flux) & np.isfinite(err) & (err > 0),
        "airmass": airmass,                               # (N,)
        "mjd": mjd,                                       # (N,)
    }
    validate_timeseries_shapes(data)



.. _reference-atmosphere:


Configuring the fit
*******************

See API reference :class:`~enuma.config.FitConfig` for details on each configuration field.
Some notes on the important ones follow.


Observatory
-----------

:attr:`FitConfig.observatory <enuma.config.FitConfig.observatory>` is required.
It sets the surface altitude of the model atmosphere, the barycentric correction,
and the location of the GDAS profile. Pass any astropy site name
(``astropy.coordinates.EarthLocation.get_site_names()``, e.g. ``"paranal"``,
``"keck"``, ``"lasilla"``) or an ``EarthLocation``. When astropy can't resolve
the site (e.g. offline), set
:attr:`~enuma.config.FitConfig.site_location` ``= (lat_deg, lon_deg, alt_m)``.

Reference atmosphere profile source
-----------------------------------

The fit perturbs a *reference* atmosphere (see :ref:`method`), so a good
reference speeds up convergence and makes the result more robust.
:attr:`~enuma.config.FitConfig.reference_profile_source` selects where the
reference temperature, pressure and water-vapour profiles come from:

``"mipas"`` (default)
    The bundled MIPAS climatological atmosphere profiles. They are stored in ``data/`` 
    (e.g., ``equ.atm``, ``day.atm``, ``ngt.atm``), but not specific to the night or site.

``"gdas"``
    The NOAA GDAS analysis column nearest to the observation: the closest
    6-hourly cycle and 0.25° grid cell. It is fetched automatically from the
    observation MJD and the observatory, and cached (see :ref:`data`). Data are
    available from mid-2021. To use a column you already have, pass it with
    :attr:`~enuma.config.FitConfig.reference_profile_nc_path`. Recommended when
    the date is covered.

In both cases the dry species (CO2, CH4, N2O, CO, O2, …) start from the MIPAS
profiles; with ``"gdas"``, O3 also comes from GDAS. For a time series, the
median MJD of the night selects the GDAS column.

If the header or :attr:`~enuma.config.FitConfig.pwv_mm` provides the precipitable
water vapour, the reference H2O profile is rescaled to match it before the fit.

With the GDAS source you can also set
:attr:`~enuma.config.FitConfig.wind_enabled` ``=True`` in :class:`~enuma.config.FitConfig`. Each layer's absorption is
then Doppler-shifted by the line-of-sight GDAS wind.



.. _fit-windows:

Restricting the fit to wavelength windows
-----------------------------------------

Not every order is equally informative. Some have few telluric lines, others are
dominated by stellar features or have low signal-to-noise.
:attr:`~enuma.config.FitConfig.fit_windows` restricts the atmospheric fit to the
orders overlapping specific wavelength windows, then extends the solution to all
orders.

.. code-block:: python

    fc = FitConfig(
        species=("h2o", "o2"),
        observatory="keck",
        # O2, H2O bands
        fit_windows=((688, 690), (720, 725), (755, 770), (815, 820)),
    )

The fit then runs in two stages:

1. **Stage 1** fits the full model (atmosphere, instrument, continuum) on the
   orders that overlap any window. Windows select *whole orders*, not pixel
   ranges.
2. **Stage 2** freezes the fitted atmosphere and fits only the per-order
   continuum, resolution and wavelength solution of the remaining orders.

The returned :class:`~enuma.inference.fit.FitResult` covers every loaded order. Its
``losses`` and ``param_history`` describe stage 1. See
:example:`fit_kelt9_single.py` for a single-exposure example.


.. _stellar:

Including a stellar template
----------------------------

For cool stars, or any target with lots of stellar lines in the fitted range, 
a stellar template can be included in the fit with
:attr:`~enuma.config.FitConfig.stellar_enabled` ``=True``. The telluric
transmission is multiplied by a normalised  
`PHOENIX-NewEra spectrum <https://www.fdr.uni-hamburg.de/record/18108>`_, which is
Doppler-shifted and rotationally broadened:

.. code-block:: python

    fc = FitConfig(
        fit_species=("h2o", "co2", "n2o", "ch4", "co"),
        observatory="paranal",
        reference_profile_source="gdas",
        stellar_enabled=True,
        stellar_teff=3200,     # K
        stellar_logg=5.0,      # cgs
        rv_kms=20.91,          # systemic RV (km/s), initial guess
    )

* The stellar parameters, including ``stellar_teff``, ``stellar_logg``, ``stellar_feh`` and ``stellar_alpha``, are
  **fixed**. The model selects the nearest node of the PHOENIX-NewEra grid. The template
  is downloaded on first use and cached (see :ref:`data`).
* The systemic velocity and ``vsini`` are **fitted** by default, starting from
  ``rv_kms`` and ``vsini``, with Gaussian priors of width 10 km/s around those
  values. Set ``fit_rv=False`` / ``fit_vsini=False`` to hold them fixed.
* The barycentric correction is added to the systemic velocity automatically
  (computed from the header coordinates, MJD and observatory).
* The rotation kernel extends ``stellar_rot_half_width`` model pixels to each
  side, and one model pixel is about 0.3 km/s. The default of 151 pixels covers
  ±45 km/s; for fast rotators, raise it to at least ``vsini / 0.3``.


The fitted telluric-only transmission is available in the third
column of ``best_fit.txt`` (see :ref:`outputs`) and through
:func:`enuma.model.forward.telluric_only` (single exposure).

.. See :example:`fit_mdwarf_timeseries.py` for a full example of fitting an M dwarf this way.


.. Limiting CPU usage
.. ------------------

.. On CPU, JAX uses every core by default. To cap it, call
.. :func:`enuma.backend.set_num_threads` before the first computation, or export
.. ``ENUMA_NUM_THREADS`` before starting Python (the CLI also takes ``--threads``):

.. .. code-block:: python

..     import enuma
..     enuma.set_num_threads(4)  # at most ~400% CPU


.. _timeseries:

Fitting a time series
*********************

High-resolution observations often consist of a time series of the same target, 
for example a transit or a phase curve. Instead of fitting each
exposure on its own, :func:`~enuma.inference.fit.fit_timeseries` fits all exposures in **one
joint likelihood**. Parameters that cannot change much during the night can then
be shared by all exposures. For example, sharing the dry-species columns (CO2,
CH4, O2, …) lets the change in airmass over the night constrain them.


.. code-block:: python

    from enuma import FitConfig, fit_timeseries, load_h5_spectra

    data = load_h5_spectra("KELT9_2024-08-08_raw.h5", wave_range=(685.0, 773.0))

    fc = FitConfig(
        species=("h2o", "o2"), fit_species=("h2o", "o2"),
        observatory="keck",
        reference_profile_source="gdas",
        dry_vmr_per_exposure=False,      # one O2 column for the night (airmass lever)
        temperature_per_exposure=False,  # one temperature profile for the night
        max_steps=1000,
    )
    res = fit_timeseries(data, fc)
    res.plot("kelt9_fit_out")            # time-series diagnostic PDFs

See :example:`fit_kelt9_timeseries.py` and :example:`fit_mdwarf_timeseries.py`
for full examples of fitting optical and near-infrared time-series spectra, respectively.


Shared vs per-exposure parameters
---------------------------------

Each parameter group is either fitted separately for every exposure (it gets a
leading ``(N,)`` axis) or shared by the whole night:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Parameter
     - Per exposure if
   * - temperature profile (``t_latent``)
     - ``temperature_per_exposure=True``
   * - H2O profile (``h2o_latent``)
     - ``h2o_per_exposure=True``
   * - dry-species columns (``dry_vmr_<species>``)
     - ``dry_vmr_per_exposure=True``
   * - resolution R(λ) (``resolution_coeffs``)
     - ``resolution_per_exposure=True``
   * - wavelength solution (``wave_coeffs``)
     - ``wave_per_exposure=True``
   * - continuum (``continuum_coeffs``)
     - always per exposure
   * - stellar ``rv_systemic`` and ``vsini``
     - always shared

All flags default to ``True``, which is equivalent to fitting each exposure
independently except for the stellar parameters. A sensible starting point is:

* share the **dry-species columns** (``dry_vmr_per_exposure=False``) so the
  airmass variation constrains them;
* keep **H2O** per exposure, since water vapour varies on timescales of minutes;
* share the **temperature profile**, and for a stable spectrograph also the
  **resolution** and **wavelength solution**, to reduce the number of free
  parameters.

The barycentric velocity of each exposure is read from ``data["baryrv"]`` when
present (the HDF5 loader reads it). Otherwise it is computed from the target
coordinates in the header, the per-exposure MJDs and the observatory. It only
matters when a stellar template is fitted.


.. Saving the results
.. ------------------

.. Unlike :func:`~enuma.inference.fit.fit_spectrum`, :func:`~enuma.inference.fit.fit_timeseries` writes nothing
.. to disk. Save the fitted parameters and, if needed, the model cubes yourself, as
.. :example:`fit_kelt9_timeseries.py` does:

.. .. code-block:: python

..     import numpy as np
..     from enuma import forward_model_batched

..     # Best-fit parameters, one array per parameter ("site").
..     np.savez("sites.npz", **{k: np.asarray(v) for k, v in res.sites.items()})

..     # Full model (telluric x stellar x continuum) and telluric-only transmission,
..     # both (N, O, P).
..     full = forward_model_batched(res.params, res.context, res.config, res.layout)
..     tell = forward_model_batched(res.params, res.context,
..                                  res.config._replace(stellar_enabled=False), res.layout)

.. ``tell`` is the telluric transmission times the fitted continuum: the model of
.. the observation without the star. Dividing the observed flux by it removes both
.. the tellurics and the continuum.


.. Best-fit I/O
.. ************

.. :func:`~enuma.inference.fit.fit_spectrum` saves the best-fit parameters to JSON and the best-fit
.. spectrum to a text file with the columns ``wavelength_nm flux_model
.. flux_telluric``. To load them again:

.. .. code-block:: python

..     from enuma import (
..         load_best_fit_params_json,
..         generate_best_fit_spectrum,
..     )

..     params = load_best_fit_params_json("best_fit.json")
..     spec = generate_best_fit_spectrum(params, result.context, result.config,
..                                       obs_flux=data["flux"])
..     model_flux = spec["flux_model"]

.. See :ref:`outputs` for the file formats.


Command-line interface
**********************

The package ships a small ``argparse``-based CLI for fitting a single exposure
from a FITS file without writing Python.

.. code-block:: console

    # Run an SVI fit and write best_fit.json + best_fit.txt + the diagnostic PDFs.
    $ python -m enuma fit zetCMa.fits \
        --observatory paranal \
        --species h2o,co2,ch4,co \
        --fit-species h2o,co2,ch4 \
        --wave-range 2050,2500 \
        --steps 900 \
        --out-dir ./fit_out

    # Time the forward model on the input spectrum (warmup + N iters).
    $ python -m enuma bench zetCMa.fits --observatory paranal --species h2o,co2 --iters 50

    # Regenerate plots from a saved best-fit JSON.
    $ python -m enuma plot zetCMa.fits --observatory paranal \
        --params fit_out/best_fit.json --out-dir ./fit_out

Other flags include ``--fit-windows '2050,2080;2150,2180'``, ``--lsf-kernel
kernel.txt``, ``--airmass``, and the stellar options ``--stellar-teff``,
``--stellar-logg``, ``--stellar-feh``, ``--vsini``, ``--rv``, ``--no-fit-rv`` and
``--no-fit-vsini``. Verbosity: ``-v`` for INFO, ``-vv`` for DEBUG (default WARNING).
Add ``--device gpu`` to force GPU.
Run ``python -m enuma fit --help`` for the full list.


A YAML config mirrors :class:`~enuma.config.FitConfig` field-for-field; CLI flags
override the YAML:

.. code-block:: yaml

    # config.yaml
    species: ["h2o", "co2", "ch4"]
    fit_species: ["h2o", "co2"]
    observatory: paranal
    reference_profile_source: gdas
    max_steps: 800
    learning_rate: 0.001
    airmass: 1.4              # override a missing/wrong header value

.. code-block:: console

    $ python -m enuma fit zetCMa.fits --config config.yaml --out-dir ./fit_out

