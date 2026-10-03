"""enuma: JAX-based forward model and fitter for telluric absorption.

Public API (typical user workflow):

    from enuma import load_fits_spectra, FitConfig, fit_spectrum

    data = load_fits_spectra("path/to/spectrum.fits", wave_range=(2050, 2500))
    result = fit_spectrum(data, FitConfig(fit_species=("h2o", "co2")))
    result.plot()

Lower-level building blocks live in role-based subpackages and can be imported
directly: ``enuma.model`` (forward model + setup, opacities, atmosphere),
``enuma.inference`` (sites, NumPyro model, SVI runner, fit), ``enuma.io``
(loading + persistence), plus the top-level ``enuma.state``,
``enuma.config``, ``enuma.constants``, and ``enuma.plotting``.
"""

from enuma.backend import set_device, set_num_threads
from enuma.io.data_loader import (
    load_fits_spectra,
    load_h5_spectra,
)
from enuma.model.setup import build_context, get_observatory, setup_model
from enuma.config import FitConfig
from enuma.inference.fit import (
    FitResult,
    fit_spectrum,
    fit_timeseries,
)
from enuma.model.forward import forward_model, forward_model_batched
from enuma.io.results import (
    generate_best_fit_spectrum,
    load_best_fit_params_json,
    save_best_fit_params_json,
    save_best_fit_spectrum_txt,
)
from enuma.plotting import (
    plot_fit_results,
    plot_instrument_diagnostics,
    plot_loss_history,
    plot_residuals_diagnostic,
    plot_timeseries_diagnostics,
    plot_timeseries_residuals,
    plot_timeseries_parameters,
    plot_timeseries_profiles,
)
from enuma.state import Layout, ModelConfig, ModelContext, ModelParameters
from enuma.model.stellar import (
    build_stellar_template,
    fetch_phoenix_spectrum,
    normalize_template,
)

__all__ = [
    # High-level
    "set_device",
    "set_num_threads",
    "load_fits_spectra",
    "load_h5_spectra",
    "FitConfig",
    "FitResult",
    "fit_spectrum",
    "fit_timeseries",
    "save_best_fit_params_json",
    "load_best_fit_params_json",
    "save_best_fit_spectrum_txt",
    "generate_best_fit_spectrum",
    # Stellar templates
    "fetch_phoenix_spectrum",
    "normalize_template",
    "build_stellar_template",
    # Observatory resolution (astropy site registry)
    "get_observatory",
    # Lower-level
    "setup_model",
    "build_context",
    "forward_model",
    "forward_model_batched",
    "Layout",
    "ModelParameters",
    "ModelContext",
    "ModelConfig",
    "plot_fit_results",
    "plot_residuals_diagnostic",
    "plot_instrument_diagnostics",
    "plot_loss_history",
    "plot_timeseries_residuals",
    "plot_timeseries_profiles",
    "plot_timeseries_parameters",
    "plot_timeseries_diagnostics",
]
