.. _stellar-learned:

Learning the stellar spectrum
#############################

For cool stars, a model template such as PHOENIX-NewEra (:ref:`stellar`) often
misses the observed stellar lines by several per cent. The telluric fit then tries
to absorb the mismatch and its columns, line-spread function and continuum are
biased. With :attr:`~enuma.config.FitConfig.stellar_model` ``="gp"``, *enuma*
learns the stellar spectrum from the data instead, while the tellurics keep their
physical model. It works for one night or several nights of the same star.

Quick start
***********

.. code-block:: python

    from enuma import FitConfig, fit_timeseries, fit_nights

    fc = FitConfig(
        fit_species=("h2o", "co2", "n2o", "ch4", "co"),
        observatory="paranal", reference_profile_source="gdas",
        dry_vmr_per_exposure=False,       # share over the night (see below)
        temperature_per_exposure=False,
        stellar_enabled=True, stellar_teff=3200, stellar_logg=5.0, rv_kms=20.91,
        stellar_model="gp",               # learn the stellar spectrum
        stellar_reference="newera",       # prior anchor: "newera", "flat" or a file
    )
    res = fit_timeseries(data, fc)        # one night
    res.plot("out")                       # + stellar_template / static_residuals / history

    multi = fit_nights([night1, night2], fc)   # several nights, one template
    multi.stellar.flux, multi.stellar.sigma    # template and its error bars

The night is first fitted with the fixed NewEra template (pass ``initial=`` to
:func:`~enuma.inference.stellar_em.fit_nights` to reuse earlier fits). Then the
template and the tellurics are refined in turn. See :example:`fit_mdwarf_gp.py`.


How it works
************

The data
--------

For exposure *n* and pixel *p*:

.. code-block:: text

    F[n,p] = c[n,p] · Instrument_p{ T_n(λ) · S_n(λ) } + noise

* ``S(λ)`` is the stellar spectrum in its own rest frame, with continuum 1. It is
  the unknown.
* ``S_n`` is the same spectrum Doppler-shifted into the telescope frame, with
  ``v_n = RV + barycentric velocity``.
* ``T_n = exp(−a_n · τ(λ; θ_n))`` is *enuma*'s telluric transmission, with ``a_n``
  the airmass.
* ``Instrument`` applies the LSF and integrates over the pixel.
* ``c`` is the continuum.

The star multiplies the atmosphere *before* the LSF, as in reality.

When can star and atmosphere be told apart?
-------------------------------------------

Drop the LSF, assume weak lines and take the log, with ``x = ln λ``:

.. code-block:: text

    y_n(x) = s(x − u_n) − a_n · t(x) + noise          (s = ln S, t = optical depth, u_n = v_n/c)

**Nothing changes between exposures.** Then only the sum ``s − a·t`` is measured,
and any pattern can be moved between the star and the atmosphere.

**Airmass changes but velocity does not (one night).** At each pixel, ``y_n`` is a
straight line in ``a_n``: the intercept is the star and the slope is the
atmosphere (a Langley plot). The intercept, however, is an extrapolation to
airmass 0, which multiplies the noise by ``√(1 + mean(a)²/var(a))``. That factor is
9.6 for a night at airmass 1.15–1.62.

What rescues one night is *enuma*'s physics. The telluric optical depth is not
free at every pixel: it is ``Σ θ_species · g_species(x)``, with line patterns ``g``
fixed by HITRAN and one column number per species. Each column is set by
thousands of lines. The star then follows pixel by pixel, without extrapolation.
Not knowing the star costs only the night's average absorption: a column shared
over the night is constrained by the airmass *change*, with an error larger by
``1/√(var(a)/mean(a²))``, ×9.6 for that night.

**An exact ambiguity.** A quantity fitted in every exposure can trade with the
star. For water fitted per exposure, ``s → s − k·g_water`` together with
``w_n → w_n − k/a_n`` leaves every exposure unchanged, for any ``k``. One night of
data cannot fix ``k``; the template prior does (below). For this reason, share the
dry columns and the temperature profile over the night. If the night's water
vapour is stable, sharing H2O as well (``h2o_per_exposure=False``) removes the
last exact ambiguity.

**Velocity changes too (several nights).** The star then moves against the
telluric lines, so a pattern fixed in the telescope frame can no longer hide in the
star. This needs the spread of ``v_n`` to reach about one LSF sigma,
``c / (2.355 R) ≈ 1.1–1.5 km/s``. Two nights a few km/s apart in barycentric
velocity already help.

The template
------------

The star is a list of numbers ``β``: the flux at nodes spaced
``stellar_node_kms`` (0.6 km/s) apart in its rest frame, joined by straight lines.
With the tellurics fixed, every step of the model is linear in ``β``, so the model
of exposure *n* is ``A_n · β`` for a matrix ``A_n``. A pixel only sees nodes within
the LSF window, so ``A_n`` is banded. It is read off the forward model exactly with
one JVP per "color": node *j* gets color ``j mod B``, and no pixel sees two nodes of
the same color.

Fitting the template
--------------------

With the tellurics fixed, the template minimises a least-squares term plus a
penalty:

.. code-block:: text

    χ²(β) + (β − β̄)ᵀ Q (β − β̄)
    χ²(β) = Σ_n Σ_p ( F[n,p] − (A_n·β)[p] )² / σ[n,p]²

The penalty says "the star looks like ``β̄``, plus line-like wiggles of typical size
σ and width ℓ".

* The anchor is ``β̄ = 1 + s_o · (reference − 1)``: the reference's lines scaled by
  one factor ``s_o`` per order. The evidence picks ``s_o``, and it can reach 0.
  The anchor's shape stays fixed in every round; it never becomes the learned
  template, which would count the data twice.
* ``Q`` is a Matérn-3/2 Gaussian process, written as a sparse (banded) precision.
  Its correlation length ℓ (per order, 1–10 km/s) is about a line width.
  Its amplitude is
  ``σ_j² = σ_o² + α² · (reference line depth near j)²``, looser where the
  reference predicts lines, because models get line positions right more often
  than depths.

Being quadratic, the problem is solved by one linear system,

.. code-block:: text

    (G + Q) · m = Σ_n A_nᵀ W_n F_n + Q·β̄,      G = Σ_n A_nᵀ W_n A_n,   W_n = 1/σ²

``G + Q`` is block-tridiagonal, so a block Cholesky factorisation solves it in
milliseconds (:mod:`enuma.inference.banded`). Its inverse gives the error bar of
every node. For a single node without LSF the solution reads

.. code-block:: text

    m = [ Σ_n w_n · F_n/(c_n·T_n) + β̄/σ² ] / [ Σ_n w_n + 1/σ² ],     w_n = (c_n·T_n/σ_n)²

* Where the atmosphere is transparent, this is the noise-weighted average of
  data / (continuum × telluric).
* Under saturated telluric lines, ``w_n → 0`` and the template falls back to the
  anchor.

Choosing the penalty strength
-----------------------------

σ_o, ℓ, s_o and α (all per order) are set by the **evidence**: the probability of the data with the
template integrated out,

.. code-block:: text

    −2 ln Z = χ²(m) + (m − β̄)ᵀQ(m − β̄) + ln det(G + Q) − ln det Q + const.

The log-determinants charge for flexibility, so the evidence neither overfits nor
underfits. For one node with prior width σ and one measurement ``y`` with noise ε,
the evidence peaks at ``σ² = y² − ε²``: the prior width matches the real signal.
Unlike wobble's cross-validation, this works with a single night.

The loop
--------

1. Fit each night with the fixed NewEra template, giving tellurics θ and the RV.
   The RV is then held fixed: it defines the template's rest frame.
2. **Star step**: refit the continuum (linear once the template is fixed), build
   ``A_n`` at θ, refit the hyperparameters, and solve for the template.
3. **Telluric step**: refit θ with the template fixed. This is the usual SVI,
   warm-started, with ``stellar_mstep_steps`` steps.
4. Repeat ``stellar_em_iterations`` times, then finish with a star step at the
   final θ, which also gives the error bars.

Several nights share one template; each keeps its own tellurics.


What one night and several nights deliver
*****************************************

.. list-table::
   :header-rows: 1

   * -
     - one night
     - several nights (velocity spread ≳ 1.5 km/s)
   * - Fit to the data
     - noise level
     - noise level
   * - Shared dry columns
     - from the airmass change
     - + Doppler
   * - Water column (per exposure)
     - set by the template prior
     - by the data
   * - Errors of the telluric model (line list, shapes)
     - end up in the template
     - separated
   * - Template
     - with error bars; falls back to the anchor under saturated lines
     - better constrained

The per-exposure telluric correction ``F / T^obs`` is as good as the physical
telluric model in both cases. In one night, the *static* part of any
telluric-model error is carried by the template.

When the star moves by less than a few km/s relative to the observatory, stellar
and telluric lines of the same molecule (H2O, CO) overlap and their separation
weakens; *enuma* warns below 3 km/s.

Current limitations
-------------------

* **Slow convergence of shared columns.** During a telluric step the template is
  fixed, and it holds every shared column near its starting value. The template
  already contains the compensating pattern, so only the weak airmass (or Doppler)
  signal pulls the column, and each round moves it a little. A column that the
  NewEra seed fit pushed far off persists. That happens to weakly constrained
  species (such as N2O or CO outside their bands) that absorbed the NewEra
  mismatch. Several nights, more rounds (``stellar_em_iterations``) or better
  starting columns help. Re-solving the template while the tellurics move would
  remove the effect; that is the planned next step.
* All nights must share the instrument setting, and orders that overlap in
  wavelength are not supported yet.


Options
*******

.. list-table::
   :header-rows: 1

   * - ``FitConfig`` field
     - Default
     - Meaning
   * - ``stellar_model``
     - ``"newera"``
     - ``"gp"`` learns the stellar spectrum
   * - ``stellar_reference``
     - ``"newera"``
     - prior anchor: ``"newera"``, ``"flat"``, or a two-column file (nm, flux)
   * - ``stellar_node_kms``
     - 0.6
     - node spacing (km/s)
   * - ``stellar_gp_length_kms``
     - 3.0
     - starting correlation length ℓ (refined per order by the evidence)
   * - ``stellar_gp_sigma``
     - 0.05
     - starting prior amplitude σ_o (refined by the evidence)
   * - ``stellar_em_iterations``
     - 5
     - rounds of (star step, telluric step)
   * - ``stellar_mstep_steps``
     - 300
     - SVI steps per telluric step
   * - ``stellar_em_tol``
     - 0
     - stop early when −2 ln Z changes by less than this fraction

All numerical code is JAX (jit-compiled, differentiable, GPU-ready). A star step
costs about as much as 25 SVI steps of the telluric fit: about a minute for 38
exposures × 18 orders on a laptop CPU, most of it spent choosing the
hyperparameters. The telluric steps dominate the total time.


Outputs
*******

The result's ``stellar`` field is a
:class:`~enuma.inference.stellar_gp.StellarTemplate`, saved by ``res.plot()`` as
``stellar_template.npz``:

* ``wave``: rest-frame node wavelengths;
* ``flux``, ``sigma``: the template and its error bars;
* ``anchor``, ``prior_sigma``, ``reference``: the prior;
* ``hyper``, ``neg2_lnz``, ``rv_kms``: the fitted hyperparameters, the evidence and
  the RV that defines the rest frame.

``res.context`` holds the template on the model grid, so
:func:`~enuma.model.forward.forward_model_batched` reproduces the full model, and
the telluric transmission is the model with ``stellar_enabled`` off and a unit
continuum. ``res.plot()`` adds three figures:

* ``stellar_template.pdf``: the template with error bars against the anchor;
* ``static_residuals.pdf``: the night-mean residual per order, which the
  residual image subtracts;
* ``stellar_history.pdf``: the evidence, the hyperparameters and the columns,
  round by round.


Comparison with wobble
**********************

.. list-table::
   :header-rows: 1

   * -
     - wobble
     - *enuma*
   * - Tellurics
     - learned per pixel (mean + patterns × airmass)
     - physical model: columns + temperature
   * - One night
     - degenerate
     - works through the physical tellurics
   * - Star × atmosphere
     - multiplied after the LSF, in log flux
     - multiplied before the LSF
   * - Regularisation
     - cross-validation over many epochs
     - evidence, from one night
   * - Output
     - best template
     - template + error bars
   * - Coverage
     - per order
     - one template over orders and nights


Testing
*******

:example:`inject_mdwarf.py` builds synthetic nights from a real one, with known
tellurics and a star outside the NewEra family. It fits them with the fixed
template, with the learned template and a flat anchor, and with the learned
template and the NewEra anchor, then reports how well each recovers the telluric
transmission (static and per-exposure errors) and the columns. ``--nights 2
--shift-kms 5`` adds a second night offset in barycentric velocity.
