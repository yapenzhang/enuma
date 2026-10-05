__all__ = [
    "K_B",
    "G_EARTH",
    "AMU",
    "M_DRY_AIR",
    "M_H2O",
    "M_O3",
    "MMW_DRY_AIR",
    "MMW_H2O",
    "C_KMS",
    "OPACITY_GRID_R",
    "LSF_FWHM_TO_SIGMA",
    "MATMUL_PRECISION",
]

# ---------- Physical constants (cgs) ----------
K_B = 1.380649e-16        # Boltzmann constant (erg/K)
G_EARTH = 980.665         # Standard gravity (cm/s^2)
AMU = 1.66053906660e-24   # Atomic mass unit (g)

# ---------- Molar masses (g/mol) — single source for molecular masses ----------
# The radiative transfer needs per-molecule masses (``* AMU``, below); the
# specific-humidity -> VMR conversions (``standard_profile`` and the plotting
# profile overlays) need only dry-air/species ratios. Both derive from these
# molar masses, so the values live in exactly one place.
M_DRY_AIR = 28.9647       # mean molar mass of dry air (g/mol)
M_H2O = 18.01528          # molar mass of water vapour (g/mol)
M_O3 = 47.984745          # molar mass of ozone (g/mol)

# Per-molecule masses (g/molecule) for column-density / RT use.
MMW_DRY_AIR = M_DRY_AIR * AMU   # dry-air mean molecular mass (g/molecule)
MMW_H2O = M_H2O * AMU           # water-vapour molecular mass (g/molecule)

# ---------- Numerical / instrument constants ----------
# Speed of light in km/s for Doppler shifts and velocity grids.
C_KMS = 299792.458

# Resolution of the precomputed opacity grids. Its intrinsic convolution width
# is removed from the LSF sigma to avoid double counting; see ``forward.process_order``.
OPACITY_GRID_R = 5e5

# Gaussian FWHM -> sigma factor: 2*sqrt(2*ln2).
LSF_FWHM_TO_SIGMA = 2.35482

# Precision of every matmul/convolution in the forward model. JAX's GPU default
# (TF32 on Ampere, ~3 significant digits) is coarse enough to roughen the loss
# surface and stall SVI; "highest" is full float32 (a no-op on CPU).
MATMUL_PRECISION = "highest"
