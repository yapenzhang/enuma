"""Diagnostic plots for fits.

``spectrum`` holds the single-exposure spectrum/residual/instrument plots plus
the shared SVI loss/convergence diagnostics; ``timeseries`` holds the
multi-exposure (night) diagnostics; ``stellar`` the learned stellar template.
"""

from enuma.plotting.spectrum import (
    plot_fit_results,
    plot_instrument_diagnostics,
    plot_loss_history,
    plot_param_convergence,
    plot_residuals_diagnostic,
)
from enuma.plotting.stellar import (
    plot_static_residuals,
    plot_stellar_diagnostics,
    plot_stellar_history,
    plot_stellar_template,
)
from enuma.plotting.timeseries import (
    plot_timeseries_diagnostics,
    plot_timeseries_residuals,
    plot_timeseries_parameters,
    plot_timeseries_profiles,
)

__all__ = [
    "plot_fit_results",
    "plot_instrument_diagnostics",
    "plot_loss_history",
    "plot_param_convergence",
    "plot_residuals_diagnostic",
    "plot_static_residuals",
    "plot_stellar_diagnostics",
    "plot_stellar_history",
    "plot_stellar_template",
    "plot_timeseries_diagnostics",
    "plot_timeseries_residuals",
    "plot_timeseries_parameters",
    "plot_timeseries_profiles",
]
