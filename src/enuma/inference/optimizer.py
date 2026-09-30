"""AutoDelta SVI optimisation runner.

``run_svi`` drives a NumPyro generative model (the unified ``telluric_model``) with
an AutoDelta guide + Adam, with optional windowed-loss early stopping. It is
deliberately model-agnostic: it takes the model and its full positional argument
tuple, so the same runner serves the single-exposure and time-series fits.
"""

import logging
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import optax
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.infer.initialization import init_to_value
from tqdm import tqdm

__all__ = [
    "run_svi",
]

logger = logging.getLogger("enuma.inference.optimizer")


def _strip_auto_loc(sites: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """Drop the ``_auto_loc`` suffix NumPyro's AutoDelta appends to guide sites.

    Applied once, at the SVI boundary (here), so every downstream consumer — the
    params bridge, plotting — sees plain site names.
    """
    return {(k[:-len("_auto_loc")] if k.endswith("_auto_loc") else k): v
            for k, v in sites.items()}


class _ConvergenceTracker:
    """Windowed-loss plateau detector for SVI early stopping.

    Every ``check_every`` steps it compares the mean loss of the last window with
    the previous window's; once the relative improvement stays below ``ftol`` for
    ``patience`` consecutive windows (and at least ``min_steps`` have run),
    :meth:`should_stop` returns True. ``ftol <= 0`` disables it (it never stops).
    The knobs come straight from the ``convergence_*`` fields of
    :class:`enuma.config.FitConfig`.
    """

    def __init__(self, ftol: float, patience: int, check_every: int, min_steps: int):
        self.enabled = ftol > 0
        self.ftol = ftol
        self.patience = patience
        self.check_every = check_every
        self.min_steps = min_steps
        self._prev_window: Optional[float] = None
        self._stall = 0

    def should_stop(self, step: int, losses: List[float]) -> bool:
        """Call once per (0-based) step; True when the plateau criterion is met."""
        if (not self.enabled or step + 1 < self.min_steps
                or (step + 1) % self.check_every != 0):
            return False
        window = float(np.mean(losses[-self.check_every:]))
        stop = False
        if self._prev_window is not None:
            rel = (self._prev_window - window) / (abs(window) + 1e-12)
            self._stall = self._stall + 1 if rel < self.ftol else 0
            stop = self._stall >= self.patience
        self._prev_window = window
        return stop


def run_svi(model,
            model_kwargs,
            *,
            init_values: Optional[Dict[str, jnp.ndarray]] = None,
            max_steps: int = 1000,
            learning_rate: float = 3e-3,
            convergence_ftol: float = 0.0,
            convergence_patience: int = 5,
            convergence_check_every: int = 50,
            convergence_min_steps: int = 200,
            print_freq: int = 50,
            rng_seed: int = 42) -> Tuple[Dict[str, jnp.ndarray], List[float],
                                         Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """Run AutoDelta SVI on ``model``; return (optimised site dict, loss history,
    param history). Site keys are returned with AutoDelta's ``_auto_loc`` suffix
    already stripped, so the result feeds :func:`enuma.state.sites_to_params`
    directly.

    ``max_steps`` is the maximum number of Adam steps. When ``convergence_ftol > 0``
    optimisation stops early once the windowed-loss plateau criterion is met (the
    ``convergence_*`` knobs; see :class:`enuma.config.FitConfig`); otherwise it
    always runs the full ``max_steps``.

    The param history is ``(recorded_steps, {site_name: trace})`` where each
    trace has shape ``(n_recorded, *param_shape)``; it feeds the convergence plot.
    """
    init_values = init_values or {}

    guide = AutoDelta(model, init_loc_fn=init_to_value(values=init_values))
    optimizer = numpyro.optim.optax_to_numpyro(optax.adam(learning_rate))
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    svi_state = svi.init(jax.random.PRNGKey(rng_seed), **model_kwargs)

    @jax.jit
    def step(state):
        return svi.update(state, **model_kwargs)

    losses: List[float] = []
    # Snapshot the MAP point every ~max_steps/200 iterations for the convergence
    # plot (flat trace = converged; a trace that keeps drifting flags a
    # poorly-constrained direction). get_params just unpacks the optimiser state.
    trace_every = max(1, max_steps // 200)
    rec_steps: List[int] = []
    snaps: List[Dict[str, np.ndarray]] = []

    def snapshot(i):
        rec_steps.append(i)
        snaps.append({k: np.asarray(v)
                      for k, v in _strip_auto_loc(svi.get_params(svi_state)).items()})

    tracker = _ConvergenceTracker(convergence_ftol, convergence_patience,
                                  convergence_check_every, convergence_min_steps)

    logger.info("NumPyro SVI optimisation (max %d steps%s)...", max_steps,
                ", adaptive stop" if tracker.enabled else "")
    with tqdm(total=max_steps) as pbar:
        for i in range(max_steps):
            svi_state, loss = step(svi_state)
            losses.append(float(loss))
            last = (i == max_steps - 1)
            if i % trace_every == 0 or last:
                snapshot(i)
            if i % print_freq == 0:
                pbar.set_description(f"ELBO: {loss:.2f}")
            pbar.update(1)

            if tracker.should_stop(i, losses):
                if rec_steps[-1] != i:
                    snapshot(i)
                break

    param_history = (
        np.asarray(rec_steps),
        {k: np.stack([s[k] for s in snaps]) for k in snaps[0]} if snaps else {},
    )
    # Strip AutoDelta's ``_auto_loc`` suffix once here so every downstream consumer
    # (the params bridge, plotting) sees plain site names.
    return _strip_auto_loc(svi.get_params(svi_state)), losses, param_history
