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
        stellar_reference="newera",       # prior mean: "newera", "flat" or a file
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

.. When can star and atmosphere be told apart?
.. -------------------------------------------

.. Drop the LSF, assume weak lines and take the log, with ``x = ln λ``:

.. .. code-block:: text

..     y_n(x) = s(x − u_n) − a_n · t(x) + noise          (s = ln S, t = optical depth, u_n = v_n/c)

.. **Nothing changes between exposures.** Then only the sum ``s − a·t`` is measured,
.. and any pattern can be moved between the star and the atmosphere.

.. **Airmass changes but velocity does not (one night).** At each pixel, ``y_n`` is a
.. straight line in ``a_n``: the intercept is the star and the slope is the
.. atmosphere (a Langley plot). The intercept, however, is an extrapolation to
.. airmass 0, which multiplies the noise by ``√(1 + mean(a)²/var(a))``. That factor is
.. 9.6 for a night at airmass 1.15–1.62.

.. What rescues one night is *enuma*'s physics. The telluric optical depth is not
.. free at every pixel: it is ``Σ θ_species · g_species(x)``, with line patterns ``g``
.. fixed by HITRAN and one column number per species. Each column is set by
.. thousands of lines. The star then follows pixel by pixel, without extrapolation.
.. Not knowing the star costs only the night's average absorption: a column shared
.. over the night is constrained by the airmass *change*, with an error larger by
.. ``1/√(var(a)/mean(a²))``, ×9.6 for that night.

.. **An exact ambiguity.** A quantity fitted in every exposure can trade with the
.. star. For water fitted per exposure, ``s → s − k·g_water`` together with
.. ``w_n → w_n − k/a_n`` leaves every exposure unchanged, for any ``k``. One night of
.. data cannot fix ``k``; the template prior does (below). For this reason, share the
.. dry columns and the temperature profile over the night. If the night's water
.. vapour is stable, sharing H2O as well (``h2o_per_exposure=False``) removes the
.. last exact ambiguity.

.. **Velocity changes too (several nights).** The star then moves against the
.. telluric lines, so a pattern fixed in the telescope frame can no longer hide in the
.. star. This needs the spread of ``v_n`` to reach about one LSF sigma,
.. ``c / (2.355 R) ≈ 1.1–1.5 km/s``. Two nights a few km/s apart in barycentric
.. velocity already help.

The template
------------

The star is a list of numbers ``β``: the flux at nodes spaced
``stellar_node_kms`` (0.6 km/s) apart in its rest frame, joined by straight lines.
With the tellurics fixed, every step of the model is linear in ``β``, so the model
of exposure *n* is ``A_n · β`` for a matrix ``A_n``. A pixel only sees the nodes
within its LSF window, fewer than ``B`` consecutive ones, so ``A_n`` is banded.
``B`` is the block size: the LSF kernel plus the widest pixel, in nodes, with a
margin.

``A_n`` is read off the forward model exactly with JVPs (forward-mode derivatives):
one JVP nudges some of the nodes and returns how every pixel changes. One JVP per
node would take thousands per exposure. Instead, the nodes are split into ``B``
interleaved groups, group *c* holding nodes *c*, *c + B*, *c + 2B*, …, and each JVP
nudges one whole group. No pixel sees two nodes of the same group, so a pixel's
change comes from a single node, known from the pixel's position, and the ``B``
JVPs give every non-zero entry of ``A_n``.

Fitting the template
--------------------

**The model is linear in the nodes.** With the tellurics, the continuum and the
instrument fixed, the template enters ``F = c · Instrument{T · S}`` once, and every
step between the nodes and the pixels is linear in it: the interpolation between
nodes, the Doppler shift, the multiplication by ``c`` and ``T``, the LSF and the
pixel integration. Exposure :math:`n` is therefore modelled exactly as

.. math::

   \boldsymbol{\mu}_n(\boldsymbol{\beta}) = A_n \boldsymbol{\beta},

with :math:`A_n` the Jacobian of the model (pixels × nodes) and
:math:`\boldsymbol{\beta}` the node values of one order. The noise is Gaussian
with weights :math:`W_n = \mathrm{diag}(1/\sigma_{n,p}^2)`, zero on masked pixels.

**The prior.** The template's prior is :math:`\boldsymbol{\beta} \sim
\mathcal{N}(\bar{\boldsymbol{\beta}}, Q^{-1})`. It says "the star looks like the
reference :math:`\bar{\boldsymbol{\beta}}`, plus line-like wiggles of typical size
σ and width ℓ":

* The mean :math:`\bar{\boldsymbol{\beta}}` is the reference spectrum on the
  nodes (``stellar_reference``: NewEra, flat, or a file). It stays fixed in every
  round; it never becomes the learned template, which would count the data twice.
* The precision is a Matérn-3/2 Gaussian process, written in its sparse form,

  .. math::

     Q = D^{-1} Q_\ell D^{-1}, \qquad
     Q_\ell = \tau^2 h\, \Lambda^{\mathsf T} \Lambda, \qquad
     \Lambda = \kappa^2 - \Delta_h, \qquad \kappa = \sqrt{3}/\ell.

  :math:`\Delta_h` is the second difference over the node spacing :math:`h`, so
  :math:`\Lambda` is tridiagonal and :math:`Q` pentadiagonal; :math:`(\kappa^2 -
  \Delta)^2` is the precision operator of a 1-D Matérn-3/2 field. The constant
  :math:`\tau^2 = \frac{1}{2\pi h}\int_{-\pi}^{\pi} \mathrm{d}\theta\,
  \big[\kappa^2 + \tfrac{4}{h^2}\sin^2\tfrac{\theta}{2}\big]^{-2}` gives
  :math:`Q_\ell` unit marginal variance. The correlation length ℓ (per order,
  1–10 km/s) is about a line width.
* :math:`D = \mathrm{diag}(d_j)` sets the amplitude per node,
  :math:`d_j^2 = \sigma_o^2 + \alpha^2 \varrho_j^2`, with :math:`\varrho_j` the
  reference's line depth near node :math:`j` (smoothed over ℓ). The prior is
  looser where the reference predicts lines, because models get line positions
  right more often than depths.

**The objective.** By Bayes' theorem the posterior is :math:`p(\boldsymbol{\beta}
\mid F) \propto p(F \mid \boldsymbol{\beta})\, p(\boldsymbol{\beta}) \propto
e^{-J(\boldsymbol{\beta})/2}`, with

.. math::

   J(\boldsymbol{\beta}) = \chi^2(\boldsymbol{\beta})
       + (\boldsymbol{\beta} - \bar{\boldsymbol{\beta}})^{\mathsf T} Q\,
         (\boldsymbol{\beta} - \bar{\boldsymbol{\beta}}),
   \qquad
   \chi^2(\boldsymbol{\beta}) = \sum_n (F_n - A_n \boldsymbol{\beta})^{\mathsf T}
       W_n (F_n - A_n \boldsymbol{\beta}).

**The normal equations.** :math:`J` is quadratic. Its gradient is

.. math::

   \nabla J = -2 \sum_n A_n^{\mathsf T} W_n (F_n - A_n \boldsymbol{\beta})
              + 2 Q (\boldsymbol{\beta} - \bar{\boldsymbol{\beta}}),

and it vanishes at the posterior mean :math:`\mathbf{m}`:

.. math::

   (G + Q)\, \mathbf{m} = \sum_n A_n^{\mathsf T} W_n F_n + Q \bar{\boldsymbol{\beta}},
   \qquad
   G = \sum_n A_n^{\mathsf T} W_n A_n.

The Hessian :math:`2(G + Q)` is positive definite, because :math:`Q` is and
:math:`G` is positive semi-definite. So :math:`\mathbf{m}` is the unique minimum,
and completing the square gives, exactly,

.. math::

   J(\boldsymbol{\beta}) = J(\mathbf{m})
       + (\boldsymbol{\beta} - \mathbf{m})^{\mathsf T} (G + Q)
         (\boldsymbol{\beta} - \mathbf{m}).

The posterior is therefore Gaussian, :math:`\boldsymbol{\beta} \mid F \sim
\mathcal{N}\big(\mathbf{m}, (G + Q)^{-1}\big)`, and the error bar of node
:math:`j` is :math:`\sqrt{[(G + Q)^{-1}]_{jj}}`.

**Solving about the current template.** Each star step expands about the current
template :math:`\boldsymbol{\beta}^0`. With :math:`\boldsymbol{\delta} =
\boldsymbol{\beta} - \boldsymbol{\beta}^0` and the residuals :math:`\mathbf{r}_n =
F_n - A_n \boldsymbol{\beta}^0`, the same equations read

.. math::

   (G + Q)\, \boldsymbol{\delta} = \mathbf{b} + Q (\bar{\boldsymbol{\beta}} - \boldsymbol{\beta}^0),
   \qquad
   \mathbf{b} = \sum_n A_n^{\mathsf T} W_n \mathbf{r}_n,
   \qquad
   \mathbf{m} = \boldsymbol{\beta}^0 + \boldsymbol{\delta}.

Since the model is linear, the solution does not depend on
:math:`\boldsymbol{\beta}^0`; the increment only keeps the right-hand side small,
which matters in float32. The χ² at the solution needs no further pass over the
data:

.. math::

   \chi^2(\mathbf{m}) = \chi^2(\boldsymbol{\beta}^0)
       - 2\, \boldsymbol{\delta}^{\mathsf T} \mathbf{b}
       + \boldsymbol{\delta}^{\mathsf T} G\, \boldsymbol{\delta}.

So the data enter only through :math:`G`, :math:`\mathbf{b}` and
:math:`\chi^2(\boldsymbol{\beta}^0)`. They are accumulated once per star step,
straight into banded storage, and everything after that, including the choice of
the hyperparameters below, never runs the forward model again.

**The system is block-tridiagonal.** A pixel sees the nodes within its LSF window,
fewer than :math:`B` consecutive ones, where :math:`B` is the block size (the
number of node groups above). Hence :math:`(A_n^{\mathsf T} W_n A_n)_{ij} =
\sum_p w_{n,p} A_{n,pi} A_{n,pj}` vanishes unless :math:`|i - j| < B`: :math:`G`
is banded with half-bandwidth :math:`B - 1`, and :math:`Q` is pentadiagonal. Cut
an order's nodes into blocks of :math:`B` consecutive nodes. Every non-zero entry
of :math:`M = G + Q` then lies in a diagonal block or next to it, so :math:`M` is
block-tridiagonal, with :math:`B \times B` blocks :math:`M_{k,k}` and
:math:`M_{k+1,k} = M_{k,k+1}^{\mathsf T}`. No pixel sees two orders and the prior
does not couple them, so every order is its own system.

**Block Cholesky factorisation.** Write :math:`M = L L^{\mathsf T}` with
:math:`L` block lower-bidiagonal: lower-triangular diagonal blocks
:math:`L_{k,k}` and full blocks :math:`L_{k+1,k}` below them. Matching the blocks
of :math:`L L^{\mathsf T}` with those of :math:`M`,

.. math::

   M_{k,k-1} = L_{k,k-1} L_{k-1,k-1}^{\mathsf T}, \qquad
   M_{k,k} = L_{k,k-1} L_{k,k-1}^{\mathsf T} + L_{k,k} L_{k,k}^{\mathsf T},

gives the recursion, from the first block to the last,

.. math::

   L_{k,k-1} = M_{k,k-1} L_{k-1,k-1}^{-\mathsf T}, \qquad
   L_{k,k} = \mathrm{chol}\big(M_{k,k} - L_{k,k-1} L_{k,k-1}^{\mathsf T}\big).

Solving :math:`M \mathbf{x} = \mathbf{y}` is then one forward and one backward
substitution,

.. math::

   \mathbf{z}_k = L_{k,k}^{-1} \big(\mathbf{y}_k - L_{k,k-1} \mathbf{z}_{k-1}\big),
   \qquad
   \mathbf{x}_k = L_{k,k}^{-\mathsf T} \big(\mathbf{z}_k - L_{k+1,k}^{\mathsf T} \mathbf{x}_{k+1}\big),

and the log-determinant is :math:`\ln\det M = 2 \sum_k \sum_i \ln (L_{k,k})_{ii}`.
For :math:`K` blocks each step costs :math:`O(K B^3)` operations, against
:math:`O(K^3 B^3)` for the dense matrix (:mod:`enuma.inference.banded`).

**Scaling.** The diagonal of :math:`G` spans orders of magnitude, from
transparent pixels to saturated lines and to padded nodes that see no data.
:math:`M` is therefore factored with a unit diagonal, :math:`\tilde{M} = S M S`
with :math:`S = \mathrm{diag}(1/\sqrt{M_{jj}})`, and the results are mapped back:

.. math::

   \mathbf{x} = S \tilde{M}^{-1} S\, \mathbf{y}, \qquad
   \ln\det M = \ln\det \tilde{M} - 2 \sum_j \ln S_{jj}, \qquad
   [M^{-1}]_{jj} = S_{jj}^2\, [\tilde{M}^{-1}]_{jj}.

This keeps the float32 Cholesky factorisation accurate.

**Error bars.** The error bars need only the diagonal of :math:`\Sigma = M^{-1} =
L^{-\mathsf T} L^{-1}`. The identity :math:`\Sigma L = L^{-\mathsf T}` has a block
upper-triangular right-hand side whose diagonal blocks are
:math:`L_{k,k}^{-\mathsf T}`. Its blocks :math:`(k{+}1, k)` and :math:`(k, k)` give
a recursion from the last block to the first (Takahashi's),

.. math::

   \Sigma_{k+1,k} = -\Sigma_{k+1,k+1} L_{k+1,k} L_{k,k}^{-1}, \qquad
   \Sigma_{k,k} = L_{k,k}^{-\mathsf T}
       \big(L_{k,k}^{-1} - L_{k+1,k}^{\mathsf T} \Sigma_{k+1,k}\big),

which forms only the blocks of :math:`\Sigma` within the band, again in
:math:`O(K B^3)`.

**One node, no LSF.** With a single node and no LSF, :math:`A_n = c_n T_n`, and
the normal equations reduce to

.. math::

   m = \frac{\sum_n w_n F_n / (c_n T_n) + \bar{\beta} / d^2}{\sum_n w_n + 1/d^2},
   \qquad w_n = \Big(\frac{c_n T_n}{\sigma_n}\Big)^2.

* Where the atmosphere is transparent, this is the noise-weighted average of
  data / (continuum × telluric).
* Under saturated telluric lines, :math:`w_n \to 0` and the template falls back to
  the reference.

Choosing the penalty strength
-----------------------------

σ_o, α and ℓ (all per order) are set by the **evidence**, the probability of the
data with the template integrated out. With the completed square above and
:math:`\int e^{-\mathbf{x}^{\mathsf T} M \mathbf{x}/2}\, \mathrm{d}\mathbf{x} =
(2\pi)^{K_\beta/2} (\det M)^{-1/2}` for :math:`K_\beta` nodes,

.. math::

   Z = \int p(F \mid \boldsymbol{\beta})\, p(\boldsymbol{\beta})\,
       \mathrm{d}\boldsymbol{\beta}
     = \prod_{n,p} (2\pi\sigma_{n,p}^2)^{-1/2}\,
       \Big(\frac{\det Q}{\det(G + Q)}\Big)^{1/2} e^{-J(\mathbf{m})/2},

so

.. math::

   -2 \ln Z = \chi^2(\mathbf{m})
       + (\mathbf{m} - \bar{\boldsymbol{\beta}})^{\mathsf T} Q\,
         (\mathbf{m} - \bar{\boldsymbol{\beta}})
       + \ln\det(G + Q) - \ln\det Q + \sum_{n,p} \ln 2\pi\sigma_{n,p}^2.

Both log-determinants come from block Cholesky factors, and the last term does
not depend on the hyperparameters. The evidence and its gradient (by automatic
differentiation through the factorisation) therefore cost :math:`O(K B^3)` per
evaluation; L-BFGS maximises it, 20 iterations per star step.

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
     - with error bars; falls back to the reference under saturated lines
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
  wavelength need the overlap masked in one of them (below).


Overlapping orders
******************

Every order has its own row of template nodes, so two orders cannot both have
unmasked pixels at one wavelength. Orders that overlap, as in most echelle
spectrographs, or that lie closer together than about the LSF window, are
therefore masked apart before the fit starts. ``fit_timeseries`` and
``fit_nights`` do this themselves with
:func:`~enuma.inference.stellar_em.mask_order_overlaps`, which can also be called
on its own to see the mask:

.. code-block:: python

    from enuma import mask_order_overlaps

    masked = mask_order_overlaps(night, fc)       # or a list of nights
    masked["mask"]                                # what the fit will leave out

For each pair of neighbouring orders that are too close, it masks the overlap in
the order with less S/N there. If the two orders' unmasked pixels are still too
close, it then masks a few more pixels on whichever side carries less S/N. The gap
it leaves has room on each side for half a probe window plus the largest
wavelength shift the fit allows, and it is widened by the spread of the
barycentric velocities so that it holds in the stellar rest frame. With the
default ``lsf_kernel_width`` and 2 km/s pixels (IGRINS) it is about 56 km/s.

The masked pixels drop out of the whole fit, the seed fit and the tellurics
included, and each masking is logged. Order pairs that are far enough apart are
left alone and keep exactly the nodes they had before.


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
     - prior mean: ``"newera"``, ``"flat"``, or a two-column file (nm, flux)
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
* ``reference``, ``prior_sigma``: the prior mean and width;
* ``hyper``, ``neg2_lnz``, ``rv_kms``: the fitted hyperparameters, the evidence and
  the RV that defines the rest frame.

``res.context`` holds the template on the model grid, so
:func:`~enuma.model.forward.forward_model_batched` reproduces the full model, and
the telluric transmission is the model with ``stellar_enabled`` off and a unit
continuum. ``res.plot()`` adds three figures:

* ``stellar_template.pdf``: the template with error bars against the reference;
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
template, with the learned template and a flat reference, and with the learned
template and the NewEra reference, then reports how well each recovers the telluric
transmission (static and per-exposure errors) and the columns. ``--nights 2
--shift-kms 5`` adds a second night offset in barycentric velocity.
