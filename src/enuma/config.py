from dataclasses import dataclass
from typing import Optional, Tuple

__all__ = [
    "FitConfig",
]


@dataclass(frozen=True)
class FitConfig:
    """Configuration for a telluric fit: the single entry point for every user knob.

    ``FitConfig`` is a frozen dataclass; build one with keyword arguments and derive
    variants with ``dataclasses.replace(fc, max_steps=500)``. The fields fall into these groups:

    * **Species and free parameters**: ``species``, ``fit_species``, and the
      ``fit_*`` toggles that make each parameter group free or pinned.
    * **Time series**: the ``*_per_exposure`` flags, which choose whether a parameter
      group is fitted per exposure or shared across a night
      (:func:`~enuma.inference.fit.fit_timeseries` only).
    * **Model sizes**: polynomial degrees and continuum B-spline nodes.
    * **Atmosphere**: layer grid, GP priors on the T / H2O profiles, observatory
      location, and the reference-profile source (``mipas`` / ``gdas``).
    * **Observation overrides**: ``airmass`` and ``pwv_mm`` when headers are missing
      or wrong.
    * **Instrument LSF**: resolution seed, profile shape, and an optional empirical
      kernel file.
    * **Optimiser**: Adam step cap, learning rate, adaptive stopping, saturation mask.
    * **Data locations**: opacity directory and download cache.
    * **Stellar template**: PHOENIX-NewEra template selection and the stellar RV /
      rotation.

    Names are case-insensitive (canonicalised to lower case), and sequences are
    converted to tuples on construction.
    """

    # ---- Species and free parameters ----
    #: Species included in the forward model. Each needs an opacity grid
    #: ``<opacity_dir>/<species>.hdf5``.
    species: Tuple[str, ...] = ("h2o", "co2", "n2o", "ch4", "co", "o3", "o2", "no")
    #: Subset of ``species`` whose columns are FREE in the fit. ``None`` fits every
    #: species; an empty tuple fits no columns (all pinned at the reference VMR).
    fit_species: Optional[Tuple[str, ...]] = None

    #: Fit the instrument resolution R(λ) per order (else pinned at ``resolution``).
    fit_resolution: bool = True
    #: Fit the per-order wavelength-solution correction (else no shift).
    fit_wave_solution: bool = True
    #: Fit the per-order B-spline continuum (else pinned at 1).
    fit_continuum: bool = True
    #: Fit the temperature-profile deviation (else the reference T profile is used).
    fit_temperature: bool = True

    #: Optional ``((min_nm, max_nm), ...)`` windows that restrict the full fit to the
    #: orders overlapping them. A second stage then freezes the fitted atmosphere
    #: and fits only the per-order continuum / LSF / wavelength solution of the
    #: REMAINING orders. ``None`` fits every loaded order in one stage.
    fit_windows: Optional[Tuple[Tuple[float, float], ...]] = None

    # ---- Time-series only ----
    #: Fit the dry-species columns per exposure (``True``) or share one set across
    #: the night (``False``); time series only.
    dry_vmr_per_exposure: bool = True
    #: Fit the temperature profile per exposure, or share it; time series only.
    temperature_per_exposure: bool = True
    #: Fit the H2O profile per exposure, or share it; time series only.
    h2o_per_exposure: bool = True
    #: Fit the resolution R(λ) per exposure, or share it; time series only.
    resolution_per_exposure: bool = True
    #: Fit the wavelength solution per exposure, or share it; time series only.
    wave_per_exposure: bool = True

    # ---- Polynomial degrees / sizes ----
    #: Number of coefficients of the per-order log-resolution polynomial
    #: (1 = constant R, 2 = linear in λ).
    n_resolution_coeffs: int = 2
    #: Number of coefficients of the per-order wavelength-shift polynomial
    #: (1 = constant shift, 2 = + linear, 3 = + quadratic).
    n_wave_coeffs: int = 3
    #: Number of cubic B-spline weights in the per-order continuum.
    continuum_n_nodes: int = 20

    # ---- GP priors on the per-layer T and H2O dex deviations ----
    #: GP kernel for the temperature deviation, one of ``"matern52"``,
    #: ``"matern32"``, ``"rbf"`` or ``"rq"`` (rational quadratic).
    temp_gp_kernel: str = "matern52"
    #: GP kernel for the H2O log10-VMR deviation.
    h2o_gp_kernel: str = "matern52"
    #: GP amplitude of the relative temperature deviation ΔT/T (σ).
    temp_gp_amplitude: float = 0.02
    #: GP length scale of the temperature deviation (km).
    temp_gp_length_scale_km: float = 1.0
    #: GP amplitude of the H2O deviation (σ, in dex).
    h2o_gp_amplitude: float = 0.5
    #: GP length scale of the H2O deviation (km).
    h2o_gp_length_scale_km: float = 1.0
    #: Altitude (km above sea level) above which temperature layers carry no free GP
    #: latent and stay at the reference (the spectrum is barely sensitive there).
    #: ``None`` frees every layer.
    temp_gp_cutoff_z_km: Optional[float] = 20.0
    #: Altitude (km above sea level) above which H2O layers stay at the reference.
    #: ``None`` frees every layer.
    h2o_gp_cutoff_z_km: Optional[float] = 20.0

    # ---- Atmosphere structure ----
    #: Number of altitude-grid boundaries; the model has ``n_layers - 1`` layers.
    n_layers: int = 50
    #: Top of the model atmosphere (km).
    z_top_km: float = 80.0
    #: Power-law exponent of the altitude sampling; > 1 packs layers near the
    #: surface, where most of the absorbing mass sits.
    z_sampling_exponent: float = 2.0
    #: Observatory, as an astropy site name (see ``EarthLocation.get_site_names()``,
    #: e.g. ``"keck"``, ``"paranal"``) or an ``EarthLocation``. Required (or use
    #: ``site_location``), since it sets the surface altitude, the barycentric RV, and
    #: the location of the ``gdas`` profile.
    observatory: Optional[object] = None
    #: Explicit ``(lat_deg, lon_deg, alt_m)`` fallback when astropy cannot resolve
    #: ``observatory`` (unknown site name / offline).
    site_location: Optional[Tuple[float, float, float]] = None

    # ---- Observation overrides ----
    #: Airmass override for single-exposure fits. If ``None``, the airmass comes
    #: from ``data["airmass"]`` (the header), else 1.0 with a warning.
    airmass: Optional[float] = None
    #: Precipitable water vapour (mm) used to rescale the reference H2O profile at
    #: setup. Falls back to the header PWV / IWV; ``None`` skips the scaling.
    pwv_mm: Optional[float] = None

    # ---- Instrument LSF ----
    #: Initial guess and prior mean of the resolving power, in units of 1e5
    #: (e.g. ``1.0`` = R 100 000).
    resolution: float = 1.0
    #: LSF shape, one of ``"gaussian"``, ``"voigt"`` (Lorentzian wings), or
    #: ``"custom"`` (set implicitly by ``lsf_kernel_file``).
    lsf_profile: str = "gaussian"
    #: Lorentzian HWHM as a fraction of the Gaussian σ (Voigt profile only).
    lsf_voigt_gamma_ratio: float = 0.01
    #: Odd pixel extent of the per-chunk LSF kernel on the model grid.
    lsf_kernel_width: int = 101
    #: Number of constant-σ chunks per order used to apply the variable-width LSF.
    lsf_n_chunks: int = 20
    #: Path to an empirical instrument LSF measured from real data, a whitespace-
    #: separated text file with two columns, velocity offset (km/s) and kernel
    #: amplitude (``#`` comment lines allowed). Setting it forces
    #: ``lsf_profile="custom"``: the kernel is resampled onto the velocity-uniform
    #: model grid and applied as-is, so the fitted R(λ) is bypassed and
    #: ``fit_resolution`` has no effect.
    lsf_kernel_file: Optional[str] = None

    # ---- Optimiser ----
    #: Hard cap on Adam steps (adaptive stopping usually ends the fit earlier).
    max_steps: int = 2000
    #: Adam learning rate.
    learning_rate: float = 3e-3
    #: Adaptive stopping tolerance. The fit stops when the relative improvement of
    #: the windowed mean loss stays below this for ``convergence_patience``
    #: checks. ``0`` disables it.
    convergence_ftol: float = 2e-4
    #: Consecutive below-``convergence_ftol`` checks required before stopping.
    convergence_patience: int = 5
    #: Evaluate the stopping criterion every this many steps.
    convergence_check_every: int = 50
    #: Never stop before this many steps.
    convergence_min_steps: int = 200
    #: Pixels whose normalised flux is below this value (saturated line cores) are
    #: masked out of the likelihood.
    saturation_mask_threshold: float = 0.15

    # ---- Reference atmosphere (T, P, H2O source; dry species always MIPAS) ----
    #: Source of the reference T, P and H2O (and O3) profiles. ``"mipas"`` (default)
    #: uses the bundled MIPAS ``.atm`` file. ``"gdas"`` uses the NOAA GDAS
    #: analysis column for the observing date and site, read from
    #: ``reference_profile_nc_path`` or auto-fetched from the observation MJD and
    #: ``observatory``. Dry species always come from MIPAS.
    reference_profile_source: str = "mipas"
    #: Path to a pre-fetched GDAS column NetCDF; ``None`` auto-fetches it.
    reference_profile_nc_path: Optional[str] = None

    # ---- Atmospheric winds (per-layer Doppler shift; requires GDAS source) ----
    #: Doppler-shift each layer's opacity by the GDAS line-of-sight wind before the
    #: column is summed (first-order approximation; see
    #: :mod:`enuma.model.forward`). Requires ``reference_profile_source="gdas"``.
    wind_enabled: bool = False

    #: Opacity-grid directory. ``None`` uses ``$ENUMA_DATA_DIR/opacities``, else
    #: ``data/opacities`` in the repository.
    opacity_dir: Optional[str] = None
    #: Cache directory for downloaded GDAS / stellar data. ``None`` uses
    #: ``$ENUMA_CACHE``, else ``~/.cache/enuma``.
    cache_dir: Optional[str] = None

    # ---- Stellar template ----
    #: Multiply the transmission by a normalised PHOENIX-NewEra stellar spectrum,
    #: selected by ``stellar_teff`` / ``stellar_logg`` / ``stellar_feh`` /
    #: ``stellar_alpha`` (fixed inputs, snapped to the nearest grid node).
    stellar_enabled: bool = False
    #: Stellar model grid; only ``"phoenix-newera"`` is available.
    stellar_grid: str = "phoenix-newera"
    #: Stellar effective temperature (K); required when ``stellar_enabled``.
    stellar_teff: Optional[float] = None
    #: Stellar surface gravity log g (cgs); required when ``stellar_enabled``.
    stellar_logg: Optional[float] = None
    #: Stellar metallicity [M/H].
    stellar_feh: float = 0.0
    #: Stellar alpha enhancement [α/Fe].
    stellar_alpha: float = 0.0
    #: Projected rotation velocity vsini (km/s), the initial guess and prior mean.
    vsini: float = 1.0
    #: Systemic radial velocity (km/s, barycentric frame), the initial guess and
    #: prior mean. The barycentric correction is added on top automatically.
    rv_kms: float = 0.0
    #: Fit the systemic RV (else fixed at ``rv_kms``).
    fit_rv: bool = True
    #: Fit vsini (else fixed at ``vsini``).
    fit_vsini: bool = True
    #: Half-width of the rotation kernel in model pixels; increase it for fast
    #: rotators so the kernel covers ±vsini.
    stellar_rot_half_width: int = 151
    #: Linear limb-darkening coefficient of the rotation kernel.
    stellar_epsilon: float = 0.6

    def __post_init__(self):
        """Canonicalise (lower-case names, tuple-convert sequences) and validate the
        config-only invariants in place, so every ``FitConfig`` is canonical once
        constructed. Model-layer choice membership (GP kernel, LSF profile,
        reference-profile source) is checked in ``setup_model``.
        """
        _set = object.__setattr__
        _set(self, "species", tuple(x.lower() for x in self.species))
        if self.fit_species is not None:
            _set(self, "fit_species", tuple(x.lower() for x in self.fit_species))
            unknown = set(self.fit_species) - set(self.species)
            if unknown:
                raise ValueError(f"fit_species {sorted(unknown)} not in species {list(self.species)}.")
        _set(self, "temp_gp_kernel", self.temp_gp_kernel.lower())
        _set(self, "h2o_gp_kernel", self.h2o_gp_kernel.lower())
        _set(self, "lsf_profile", self.lsf_profile.lower())
        # A kernel file selects the empirical LSF; the two stay consistent so
        # downstream code only has to test ``lsf_profile == "custom"``.
        if self.lsf_kernel_file is not None:
            _set(self, "lsf_profile", "custom")
        elif self.lsf_profile == "custom":
            raise ValueError("lsf_profile='custom' requires lsf_kernel_file (a 2-column "
                             "velocity[km/s], amplitude text file).")
        _set(self, "reference_profile_source", self.reference_profile_source.lower())
        _set(self, "stellar_grid", self.stellar_grid.lower())
        for name, value, low in (("n_layers", self.n_layers, 4),
                                 ("continuum_n_nodes", self.continuum_n_nodes, 4),
                                 ("n_wave_coeffs", self.n_wave_coeffs, 1),
                                 ("n_resolution_coeffs", self.n_resolution_coeffs, 1)):
            if value < low:
                raise ValueError(f"{name} must be >= {low}; got {value}.")
        if self.lsf_voigt_gamma_ratio < 0:
            raise ValueError("lsf_voigt_gamma_ratio must be non-negative.")
        if self.site_location is not None and len(self.site_location) != 3:
            raise ValueError("site_location must be (lat_deg, lon_deg, alt_m).")
