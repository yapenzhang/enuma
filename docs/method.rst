.. _method:

How *enuma* works
#################

This page summarises the physical model and the fitting procedure. The
:ref:`api` documents the individual functions.

The forward model maps a set of parameters to the observed spectrum in four
steps:

1. **Atmosphere**: a layered model atmosphere with free temperature and H2O
   profiles and free column scalings of the other species.
2. **Radiative transfer**: line-by-line transmission through the layers at a
   resolving power of 10\ :sup:`6`.
3. **Instrument**: convolution with the line-spread function, wavelength-solution
   correction, and resampling onto the detector pixels.
4. **Continuum**: multiplication by a per-order spline continuum (with a scale per
   exposure in a time series), and optional multiplication by a stellar template
   spectrum.

Every step is written in JAX, so the whole model is differentiable and runs on
CPU or GPU.


Atmosphere
**********

Layer grid
----------

The atmosphere is divided into ``n_layers − 1`` layers between the observatory
altitude and ``z_top_km`` (80 km by default). Layer boundaries follow a power law,
``z = z_site + s^k (z_top − z_site)`` with ``s`` uniform in [0, 1] and
``k = z_sampling_exponent`` (default 2), which places more layers near the
ground, where most of the absorbing mass sits.

Reference atmosphere
--------------------

Each layer starts from a *reference* temperature, pressure and volume mixing
ratio (VMR) per species (see :ref:`reference-atmosphere`):

* temperature, pressure and H2O come from the MIPAS climatology or the GDAS
  analysis for the night and site;
* the dry species always start from MIPAS;
* the H2O profile is optionally rescaled to a measured precipitable water vapour.

The pressure of each layer is held at its reference value; the fit does not
re-integrate hydrostatic equilibrium. The air column of each layer follows from
its pressure difference, accounting for the water-vapour content (moist air).

Free atmospheric parameters
---------------------------

**Temperature profile.** The fit perturbs the reference temperature by a
relative deviation, ``T = T_ref · (1 + ΔT/T)``.

**H2O profile.** The H2O VMR is perturbed in log space,
``VMR = VMR_ref · 10^Δ``, with Δ in dex per layer.

Both deviations are smooth functions of altitude with Gaussian-process (GP)
priors. The kernel (``temp_gp_kernel`` / ``h2o_gp_kernel``: Matérn-5/2 by
default, or Matérn-3/2, RBF, rational quadratic), the amplitude
(``*_gp_amplitude``) and the length scale (``*_gp_length_scale_km``) are set in
:class:`~enuma.config.FitConfig`. 
The GPs use a *non-centred* parametrisation:
the fit optimises unit-normal latent variables (``t_latent``, ``h2o_latent``),
which are mapped to the deviations by the Cholesky factor of the GP covariance.
.. This keeps the optimisation well conditioned. 
Layers above a certain threshold altitude ``*_gp_cutoff_z_km``
(20 km by default) carry no latent and stay at the reference, because the spectrum barely has sensitivity there.

**Other species.** Each dry species in ``fit_species`` gets a single scalar
offset δ (in dex) that scales its whole profile, ``VMR = VMR_ref · 10^δ``. The
profile shape stays that of the reference.


Radiative transfer
******************

For each species, the absorption cross-section of each layer is interpolated
bilinearly in (log pressure, temperature) from precomputed opacity grids
(see :ref:`data`), which are sampled uniformly in velocity at R ≈ 10\ :sup:`6`. The
optical depth sums cross-section × VMR × air column over all layers and species,
and the transmission is

.. math::

   \mathcal{T}(\lambda) = \exp\left[-\,\tau(\lambda)\cdot X\right],

with ``X`` the airmass.

**Winds (optional).** With ``wind_enabled=True`` and GDAS profiles, each layer's
absorption is Doppler-shifted by the line-of-sight component of the GDAS wind
before the layers are summed, to first order in velocity.


Instrument
**********

Line-spread function
--------------------

The high-resolution spectrum of each order is convolved with the instrument
line-spread function (LSF). Its width follows from a resolving power that may vary
across the order:

.. code-block:: text

    ln R(x) = ln R₀ + Σ_k b_k P_k(x)  [+ g_n in a time series]

* ``x`` runs from −1 to 1 across the observed order, and ``P_k`` are Legendre
  polynomials. ``n_resolution_coeffs`` sets the number of terms: 1 gives a constant
  R; the default, 2, lets R vary linearly across the order.
* ``R₀ = resolution × 10⁵`` is the centre of the prior. The terms ``b_k`` are static,
  one set per order.
* In a time series with ``resolution_per_exposure=True`` (the default), each
  exposure adds one ln R offset ``g_n``, shared by all orders (seeing). The offsets
  have zero mean over the night, so ``R₀·exp(b₀)`` is the night-mean R level of
  the order.

Three LSF shapes are available through ``lsf_profile``:

``"gaussian"`` (default)
    A Gaussian with FWHM = λ / R.

``"voigt"``
    A Voigt profile whose Gaussian core is set by R and whose Lorentzian wings
    have a half-width of ``lsf_voigt_gamma_ratio`` × σ. Use it when the
    instrument profile has extended wings.

``"custom"``
    An **empirical LSF** measured for the instrument by passing the text file to ``lsf_kernel_file``:

    .. code-block:: text

        # velocity_kms  amplitude
        -6.0  0.0002
        -5.5  0.0010
        ...
         0.0  1.0000
        ...
         6.0  0.0002

    The kernel is resampled onto the model grid, normalised, and applied
    unchanged to every order. The resolution is then not fitted
    (``fit_resolution`` has no effect). The CLI flag is ``--lsf-kernel``.

Wavelength solution and resampling
----------------------------------

The fit can correct the wavelength solution of each order with a shift in pixels:

.. code-block:: text

    shift(x) = Σ_k a_k P_k(x)  [+ d_n in a time series]

* The Legendre terms ``a_k`` are static, one set per order, with
  ``n_wave_coeffs`` terms (3 by default: offset, linear, quadratic).
* In a time series with ``wave_per_exposure=True`` (the default), each exposure
  adds a constant offset ``d_n`` per order, for the instrument drift. The offsets
  have zero mean over the night, so ``a₀`` is the night-mean shift of the order.

The convolved model is then resampled to the wavelengths of the observed spectrum.

Priors and bounds
-----------------

Every wavelength-shift and resolution term has a Normal prior of width ``σ`` on its
fitted value ``u`` (in pixels or ln R). The model uses it through a bound:

.. code-block:: text

    t = B · tanh(u / B),    u ~ Normal(0, σ),    B = instrument_bound_sigmas · σ  (4σ)

* Well inside the bound, ``t ≈ u``, so the optimiser moves the terms in their
  physical units.
* ``|t|`` never exceeds ``B``. Orders with few telluric lines take their
  instrument largely from the stellar lines. There, an imperfect stellar template
  could otherwise drive R or the shift to unphysical values.
* The R bound is relative to ``resolution``, so set it within about a factor of
  2 of the true resolving power.

The widths are set per Legendre degree:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Term
     - Width (default)
     - Field
   * - static shift ``a_k`` (pixels)
     - 0.5, 0.3, 0.3
     - ``wave_prior_pix``
   * - per-exposure shift ``d_n`` (pixels)
     - 0.25
     - ``wave_drift_prior_pix``
   * - static ln R ``b_k``
     - 0.2, 0.075
     - ``resolution_prior``
   * - per-exposure ln R ``g_n``
     - 0.05
     - ``resolution_drift_prior``

After every fit, :func:`~enuma.inference.diagnostics.instrument_report` logs the
terms of every order. It warns about any term beyond 75 % of its bound, and says
whether that order's shift and R are set mainly by telluric or by stellar lines.

Continuum
---------

Each order is multiplied by a smooth continuum: a cubic B-spline across the order
with ``continuum_n_nodes`` weights (20 by default). This absorbs the residual
blaze, continuum absorption and normalisation errors.

* The shape is static: one set of weights per order, shared by all exposures.
* In a time series, each exposure and order adds a scale ``s_n`` for the varying
  throughput, with ``ln s_n ~ Normal(0, continuum_scale_prior)`` and zero mean
  over the night.


**Stellar template (optional).** With ``stellar_enabled=True``, the transmission
is multiplied by a normalised PHOENIX-NewEra spectrum on the same high-resolution
grid before instrument convolution. The template is rotationally broadened (vsini, linear limb darkening
``stellar_epsilon``) and shifted by the systemic plus barycentric velocity. See
:ref:`stellar`. With ``stellar_model="gp"`` the stellar spectrum is instead learned
from the data; see :ref:`stellar-learned`.



.. _method-inference:

Inference
*********

The model is written in `NumPyro <https://num.pyro.ai>`_ and fitted with
stochastic variational inference (SVI) using an **AutoDelta** guide. An AutoDelta
guide is a point mass, so maximising the ELBO is equivalent to maximising the
posterior density: the result is the **maximum a posteriori (MAP)** estimate.
*enuma* does not return posterior samples or parameter uncertainties.

The optimiser is Adam (``learning_rate``, default 3 × 10\ :sup:`−3`), starting
from the reference atmosphere and the instrument initial guesses. The fit stops
at ``max_steps`` or earlier, when converged. Every ``convergence_check_every``
steps the mean loss of the latest window is compared with that of the previous
window; when the relative improvement stays below ``convergence_ftol`` for
``convergence_patience`` consecutive checks (and at least
``convergence_min_steps`` steps have run), the fit stops.

The forward model runs in single precision (float32) for speed and memory efficiency.

