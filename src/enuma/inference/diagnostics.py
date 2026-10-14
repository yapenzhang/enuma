"""Post-fit report on the instrument terms and the continuum scales.

For every order it gives the static wavelength-shift and resolution terms in
physical units, the range of the per-exposure drifts and continuum scales, the terms
pressed against their bounds, and how much of the order's constraint on a constant
shift and R level comes from telluric lines and how much from stellar lines.
"""

import logging
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from enuma.model.forward import (
    _exposure_inputs,
    _exposure_slice,
    _transmission,
    centre_night_terms,
    continuum_model,
    instrument_terms,
)

__all__ = [
    "PRESSED_FRACTION",
    "instrument_report",
]

logger = logging.getLogger("enuma.inference.diagnostics")

#: A term beyond this fraction of its bound is reported as pressed against it.
PRESSED_FRACTION = 0.75


@partial(jax.jit, static_argnames=("config",))
def _shift_r_jacobian(params, ctx, config):
    """``(O, P, 2)`` derivative of the transmission w.r.t. the fitted drift values: a
    constant shift (pixels) of each order and a constant ln R offset, before the bound."""
    n_orders = ctx.obs_wave.shape[0]
    ud = jnp.broadcast_to(params.wave_drift, (n_orders,))
    ug = jnp.asarray(params.resolution_drift)

    def f(theta):
        p = params._replace(wave_drift=ud + theta[0],
                            resolution_drift=ug + theta[1])
        return _transmission(p, ctx, config)

    return jax.jacfwd(f)(jnp.zeros(2, ctx.obs_wave_off.dtype))


def _exposure(result, i: int):
    """``(params, context, flux, err, mask)`` of exposure ``i`` (or the single one)."""
    data, mask = result.data, np.asarray(result.obs_mask)
    flux, err = np.asarray(data["flux"]), np.asarray(data["err"])
    if not result.layout.n_exp:
        return result.params, result.context, flux, err, mask
    ex = jax.tree_util.tree_map(lambda a: a[i], _exposure_inputs(
        result.params, result.context, result.config, result.layout))
    params, ctx = _exposure_slice(result.params, result.context, result.config, ex)
    return params, ctx, flux[i], err[i], mask[i]


def _information(result, i: int):
    """Fisher information of exposure ``i`` on a constant shift (pixels) and a constant
    ln R in each order, from the telluric lines and from the stellar lines alone:
    ``(telluric (O, 2), stellar (O, 2) or None)``."""
    config = result.config
    params, ctx, flux, err, mask = _exposure(result, i)
    ok = mask & np.isfinite(flux) & np.isfinite(err) & (err > 0)
    w = np.where(ok, 1.0 / np.where(ok, err, 1.0) ** 2, 0.0)
    cont = np.asarray(continuum_model(params, ctx, config))
    # Fitted value → bounded term: the slope of B·tanh(u/B) at the current values.
    c = config.instrument_bound_sigmas
    ud = np.broadcast_to(np.asarray(params.wave_drift), cont.shape[:1])
    ug = float(np.asarray(params.resolution_drift))
    bd, bg = c * config.wave_drift_prior_pix, c * config.resolution_drift_prior
    slope = np.stack([1.0 / np.cosh(ud / bd) ** 2,
                      np.full(ud.shape, 1.0 / np.cosh(ug / bg) ** 2)], axis=-1)   # (O, 2)

    def info(p_ctx, cfg):
        J = np.asarray(_shift_r_jacobian(params, p_ctx, cfg)) * cont[..., None]
        return np.einsum("op,opk->ok", w, J ** 2) / slope ** 2

    telluric = info(ctx, config._replace(stellar_enabled=False))
    stellar = (info(ctx._replace(airmass=jnp.zeros_like(ctx.airmass)), config)
               if config.stellar_enabled else None)
    return telluric, stellar


def instrument_report(result, *, information: bool = True,
                      exposure: Optional[int] = None) -> dict:
    """Log and return the instrument report of a fit.

    Args:
        result: a :class:`~enuma.inference.fit.FitResult` (one exposure or a night).
        information: also compute the telluric and stellar information on each
            order's shift and R level (one forward-mode Jacobian of one exposure).
        exposure: the exposure used for the information; the middle one by default.

    Returns:
        ``{"orders": [per-order dict], "flags": [str]}``. Each order dict holds the
        static shift terms (``shift_px``, Legendre, pixels), the R level and slope
        (``R_level``, ``log_r_slope``), the largest per-exposure drift
        (``max_drift_px``) and continuum-scale range (``scale_range``), and, with
        ``information``, the night's telluric and stellar information on a constant
        shift and ln R (``info_telluric``, ``info_stellar``; ``None`` without a star).
    """
    config, layout = result.config, result.layout
    params = centre_night_terms(result.params, layout)
    t = jax.tree_util.tree_map(np.asarray, instrument_terms(params, config))
    c = config.instrument_bound_sigmas
    wave_bound = c * np.asarray(config.wave_prior_pix)
    log_r_bound = c * np.asarray(config.resolution_prior)
    drift_bound = c * config.wave_drift_prior_pix
    g_bound = c * config.resolution_drift_prior
    r0 = 1e5 * config.resolution_init
    n_orders = t.wave.shape[0]
    n_exp = max(layout.n_exp, 1)
    wave = np.asarray(result.context.obs_wave)

    drift = np.broadcast_to(t.wave_drift, (n_exp, n_orders))
    log_scale = np.broadcast_to(np.asarray(params.continuum_log_scale), (n_exp, n_orders))
    if information:
        i = layout.n_exp // 2 if exposure is None else exposure
        info_t, info_s = _information(result, i)
        info_t, info_s = info_t * n_exp, (None if info_s is None else info_s * n_exp)

    rows, flags = [], []
    for o in range(n_orders):
        pressed = []
        for k, (a, b) in enumerate(zip(t.wave[o], wave_bound)):
            if abs(a) > PRESSED_FRACTION * b:
                pressed.append(f"shift P{k} = {a:+.2f} px (bound {b:.2f})")
        if abs(t.log_r[o, 0]) > PRESSED_FRACTION * log_r_bound[0]:
            pressed.append(f"R level {np.exp(t.log_r[o, 0]):.2f} R0 (bound "
                           f"{np.exp(np.sign(t.log_r[o, 0]) * log_r_bound[0]):.2f} R0)")
        for k in range(1, t.log_r.shape[1]):
            if abs(t.log_r[o, k]) > PRESSED_FRACTION * log_r_bound[k]:
                pressed.append(f"ln R P{k} = {t.log_r[o, k]:+.3f} (bound {log_r_bound[k]:.3f})")
        max_drift = float(np.max(np.abs(drift[:, o])))
        if max_drift > PRESSED_FRACTION * drift_bound:
            pressed.append(f"wavelength drift up to {max_drift:.2f} px (bound {drift_bound:.2f})")
        row = dict(order=o, wave_nm=(float(wave[o].min()), float(wave[o].max())),
                   shift_px=t.wave[o].tolist(), R_level=float(r0 * np.exp(t.log_r[o, 0])),
                   log_r_slope=(float(t.log_r[o, 1]) if t.log_r.shape[1] > 1 else 0.0),
                   max_drift_px=max_drift,
                   scale_range=(float(np.exp(log_scale[:, o].min())),
                                float(np.exp(log_scale[:, o].max()))),
                   pressed=pressed)
        if information:
            row["info_telluric"] = info_t[o].tolist()
            row["info_stellar"] = None if info_s is None else info_s[o].tolist()
        rows.append(row)
        if pressed:
            msg = f"Order {o} ({wave[o].min():.0f}-{wave[o].max():.0f} nm): " + "; ".join(pressed)
            if information and info_s is not None:
                ratio = info_s[o] / np.maximum(info_t[o], 1e-30)
                source = "stellar" if max(ratio) > 1.0 else "telluric"
                msg += (f". Star : telluric information {ratio[0]:.1f} : 1 (shift), "
                        f"{ratio[1]:.1f} : 1 (R), so its instrument is set mainly by the "
                        f"{source} lines")
            flags.append(msg + ".")

    if layout.n_exp:
        g = np.broadcast_to(t.log_r_drift, (n_exp,))
        if np.max(np.abs(g)) > PRESSED_FRACTION * g_bound:
            flags.append(f"The per-exposure R factor reaches {np.exp(g[np.argmax(np.abs(g))]):.2f} "
                         f"(bound {np.exp(g_bound):.2f}).")

    _log(rows, layout.n_exp, r0, information, info_s is not None if information else False)
    for msg in flags:
        logger.warning(msg)
    return dict(orders=rows, flags=flags)


def _log(rows, n_exp, r0, information, stellar):
    """One INFO line per order."""
    head = (f"Instrument (R0 = {r0 / 1e3:.0f}k): order, nm, shift P0/P1/... (px), "
            f"R level (k), ln R slope" + (", max |drift| (px), continuum scale" if n_exp else ""))
    if information:
        head += ", telluric sigma(shift px, ln R)" + (", star : telluric info" if stellar else "")
    logger.info(head)
    for r in rows:
        line = (f"  {r['order']:3d} {r['wave_nm'][0]:7.1f}-{r['wave_nm'][1]:7.1f}  "
                + " ".join(f"{a:+.2f}" for a in r["shift_px"])
                + f"  {r['R_level'] / 1e3:6.1f}  {r['log_r_slope']:+.3f}")
        if n_exp:
            line += f"  {r['max_drift_px']:.2f}  {r['scale_range'][0]:.3f}-{r['scale_range'][1]:.3f}"
        if information:
            sig = [1.0 / np.sqrt(max(x, 1e-30)) for x in r["info_telluric"]]
            line += f"  {sig[0]:.4f} {sig[1]:.4f}"
            if stellar:
                ratio = [s / max(tl, 1e-30) for s, tl in zip(r["info_stellar"], r["info_telluric"])]
                line += f"  {ratio[0]:.1f} {ratio[1]:.1f}"
        if r["pressed"]:
            line += "  <- at bound"
        logger.info(line)
