from dataclasses import dataclass
from typing import Optional, Tuple

__all__ = [
    "FitConfig",
]


@dataclass(frozen=True)
class FitConfig:

    species: Tuple[str, ...] = ("h2o", "co2", "n2o", "ch4", "co", "o3", "o2", "no")
    # Subset of ``species`` whose columns are FREE. None => fit every species;
    # empty tuple => fit no columns (all pinned at the reference VMR).
    fit_species: Optional[Tuple[str, ...]] = None

    fit_resolution: bool = True
    fit_wave_solution: bool = True
    fit_continuum: bool = True
    fit_temperature: bool = True

    # Optional list of (min_nm, max_nm) wavelength windows restricting the SVI fit
    # to the orders that overlap them. A second stage then freezes that fitted 
    # atmospheric state and fits the per-order continuum / LSF / wave solution 
    # for the REMAINING orders. None (default) => fit every loaded order in one stage. 
    fit_windows: Optional[Tuple[Tuple[float, float], ...]] = None

    # ---- Time-series only ----
    dry_vmr_per_exposure: bool = True
    temperature_per_exposure: bool = True
    h2o_per_exposure: bool = True
    resolution_per_exposure: bool = True
    wave_per_exposure: bool = True

    # ---- Polynomial degrees / sizes ----
    n_resolution_coeffs: int = 2   # 1 = constant R; 2 = linear in λ
    n_wave_coeffs: int = 3         # 1 = const shift; 2 = +linear; 3 = +quadratic
    continuum_n_nodes: int = 20

    # ---- GP priors on the per-layer T and H2O dex deviations ----
    temp_gp_kernel: str = "matern52"
    h2o_gp_kernel: str = "matern52"
    temp_gp_amplitude: float = 0.02        # relative T deviation σ
    temp_gp_length_scale_km: float = 1.0
    h2o_gp_amplitude: float = 0.5          # H2O dex σ
    h2o_gp_length_scale_km: float = 1.0
    # The spectrum has ~no sensitivity to the upper atmosphere so layers above these 
    # altitudes carry no free GP latent and stay at climatology. None = every layer free.
    temp_gp_cutoff_z_km: Optional[float] = 20.0 
    h2o_gp_cutoff_z_km: Optional[float] = 20.0    

    # ---- Atmosphere structure ----
    # Number of altitude-grid boundaries; layer cells/centres = n_layers - 1.
    n_layers: int = 50
    z_top_km: float = 80.0
    z_sampling_exponent: float = 2.0   # >1 packs layers near the surface
    # Observatory: an astropy site name (EarthLocation.of_site / get_site_names,
    # e.g. "keck", "paranal") or an EarthLocation instance. Supplies lat/lon/
    # altitude for the barycentric RV and the date/site atmospheric profile fetch.
    observatory: Optional[object] = None
    # Explicit (lat_deg, lon_deg, alt_m) fallback when astropy cannot resolve
    # ``observatory`` (unknown site name / offline).
    site_location: Optional[Tuple[float, float, float]] = None

    # ---- Observation overrides ----
    # Airmass: ``airmass`` if set, else ``data["airmass"]``, else 1.0 (warn).
    airmass: Optional[float] = None
    # PWV (mm) used to scale the reference H2O VMR at setup time:
    pwv_mm: Optional[float] = None

    # ---- Instrument LSF ----
    resolution: float = 1.0        # ×1e5
    lsf_profile: str = "gaussian"  # "gaussian", "voigt" (Lorentzian wings), or "custom"
    lsf_voigt_gamma_ratio: float = 0.01
    lsf_kernel_width: int = 101    # odd pixel extent of the per-chunk kernel
    lsf_n_chunks: int = 20         # constant-σ chunks per order
    # Empirical instrument LSF measured from real data. Path to a whitespace-
    # separated text file with two columns — velocity offset (km/s) and kernel
    # amplitude (``#`` comment lines allowed). Setting it forces
    # ``lsf_profile="custom"``: the measured kernel is resampled onto the (velocity-
    # uniform) model grid and convolved as-is, so the fitted resolution R(λ) is
    # bypassed and ``fit_resolution`` has no effect.
    lsf_kernel_file: Optional[str] = None

    # ---- Optimiser ----
    max_steps: int = 2000            # MAX Adam steps (hard cap; convergence stops earlier)
    learning_rate: float = 3e-3
    # Adaptive stopping: end the fit when the windowed loss plateaus rather than
    # at a fixed step count (generalises across datasets/inits). ftol=0 disables.
    convergence_ftol: float = 2e-4
    convergence_patience: int = 5        # consecutive sub-ftol checks before stopping
    convergence_check_every: int = 50    # evaluate the criterion every this many steps
    convergence_min_steps: int = 200     # never stop before this many steps
    saturation_mask_threshold: float = 0.15   # mask pixels at/below this flux

    # ---- Reference atmosphere (T, P, H2O source; dry species always MIPAS) ----
    # One of ``enuma.model.profile.REFERENCE_PROFILE_CHOICES``:
    #   "mipas" (default): bundled MIPAS .atm, no extra files.
    #   "gdas":  NOAA GDAS column from ``reference_profile_nc_path`` (or
    #            auto-fetched from the FITS date/site).
    reference_profile_source: str = "mipas"
    reference_profile_nc_path: Optional[str] = None

    # ---- Atmospheric winds (per-layer Doppler shift; requires GDAS source) ----
    # When True, each layer's telluric opacity is Doppler-shifted by the GDAS
    # line-of-sight wind before the air column is summed (first-order/derivative
    # form, see enuma.model.forward._optical_depth). Winds are read from the same
    # GDAS analysis as T/P/H2O, so this requires reference_profile_source="gdas".
    wind_enabled: bool = False

    # Opacity grid directory. None => $ENUMA_DATA_DIR/opacities
    opacity_dir: Optional[str] = None
    # Cache directory for fetched GDAS / stellar data. None => $ENUMA_CACHE or ~/.cache/enuma.
    cache_dir: Optional[str] = None

    # ---- Stellar template ----
    # When enabled, a normalised PHOENIX-NewEra spectrum for (Teff, logg, [M/H],
    # [alpha/Fe]) multiplies the transmission. Teff/logg/feh/alpha are FIXED
    # inputs selecting the grid node
    stellar_enabled: bool = False
    stellar_grid: str = "phoenix-newera"
    stellar_teff: Optional[float] = None
    stellar_logg: Optional[float] = None
    stellar_feh: float = 0.0
    stellar_alpha: float = 0.0
    vsini: float = 1.0             # km/s, projected rotation (init/prior mean)
    rv_kms: float = 0.0            # km/s, systemic RV (barycentric frame)
    fit_rv: bool = True
    fit_vsini: bool = True
    stellar_rot_half_width: int = 151   # rotation kernel half-width (model pixels)
    stellar_epsilon: float = 0.6        # linear limb-darkening coeff


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
