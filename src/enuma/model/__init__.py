"""The differentiable forward model and its construction.

``forward`` evaluates the radiative transfer + instrument model (incl. the LSF
convolution); ``setup`` assembles the precomputed ``ModelContext`` /
``ModelConfig`` it consumes; ``voigt``/``stellar`` are the Voigt-profile and
stellar building blocks; ``opacity`` and ``profile`` load the reference data
(opacity grids, atmosphere) the setup consumes.
"""

from enuma.model.forward import (
    LSF_PROFILE_CHOICES,
    apply_custom_lsf,
    apply_variable_lsf,
    forward_model,
    forward_model_batched,
    get_layer_properties,
    telluric_only,
)
from enuma.model.opacity import OpacityGridInfo, get_opacities, load_pyrox_opacity
from enuma.model.profile import (
    GP_KERNEL_CHOICES,
    REFERENCE_PROFILE_CHOICES,
    fetch_gdas_profile,
    get_standard_atmosphere,
)
from enuma.model.setup import (
    baryrv_kms,
    build_context,
    get_observatory,
    setup_model,
    wind_los_kms,
)
from enuma.model.stellar import (
    STELLAR_GRID_CHOICES,
    apply_stellar,
    build_stellar_template,
    fetch_phoenix_spectrum,
    normalize_template,
)
from enuma.model.voigt import voigt_profile

__all__ = [
    "forward_model",
    "forward_model_batched",
    "get_layer_properties",
    "telluric_only",
    "LSF_PROFILE_CHOICES",
    "apply_custom_lsf",
    "apply_variable_lsf",
    "OpacityGridInfo",
    "get_opacities",
    "load_pyrox_opacity",
    "REFERENCE_PROFILE_CHOICES",
    "fetch_gdas_profile",
    "get_standard_atmosphere",
    "GP_KERNEL_CHOICES",
    "baryrv_kms",
    "build_context",
    "get_observatory",
    "setup_model",
    "wind_los_kms",
    "STELLAR_GRID_CHOICES",
    "apply_stellar",
    "build_stellar_template",
    "fetch_phoenix_spectrum",
    "normalize_template",
    "voigt_profile",
]
