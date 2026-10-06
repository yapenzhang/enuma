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
4. **Continuum**: multiplication by a per-order spline continuum, and optional 
    multiplication by a stellar template spectrum.

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
across the order, ``R(x) = 10^5 · exp(poly(x))`` with ``x`` from −1 to 1 across the
order and ``n_resolution_coeffs`` coefficients (1 gives a constant R; the
default, 2, lets R vary linearly across the order). 
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

The fit can optionally correct the wavelength solution of each order with a polynomial shift
in pixels, ``poly(x)`` with ``n_wave_coeffs`` coefficients (3rd-order polynomial by default). 
The convolved model is then resampled to the wavelengths of the observed spectrum.

Continuum
---------

Each order (and, in a time series, each exposure) is multiplied by a smooth
continuum: a cubic B-spline across the order with ``continuum_n_nodes`` weights
(20 by default). This absorbs the residual blaze, continuum absorption, throughput variations and
normalisation errors.


**Stellar template (optional).** With ``stellar_enabled=True``, the transmission
is multiplied by a normalised PHOENIX-NewEra spectrum on the same high-resolution
grid before instrument convolution. The template is rotationally broadened (vsini, linear limb darkening
``stellar_epsilon``) and shifted by the systemic plus barycentric velocity. See
:ref:`stellar`.



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

