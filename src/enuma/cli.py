"""Command-line interface for enuma.

Subcommands:
    fit    Run a multi-order SVI fit on a FITS spectrum.
    plot   Regenerate diagnostic plots from a saved best-fit JSON.
    bench  Time forward_model on the input spectrum.

Configuration: command-line flags override values from an optional YAML file
whose keys mirror `FitConfig` fields exactly. Example invocation:

    python -m enuma fit spectrum.fits --species h2o,co2 --steps 200
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "cmd_fit",
    "cmd_plot",
    "cmd_bench",
    "build_parser",
    "main",
]

logger = logging.getLogger("enuma.cli")


def _setup_logging(verbose: int) -> None:
    """-v → INFO, -vv → DEBUG. Default WARNING."""
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level,
                        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")


def _load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required for --config; install with `pip install pyyaml`.") from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_fitconfig(args: argparse.Namespace):
    """Compose FitConfig from an optional YAML file + explicit CLI overrides."""
    from enuma import FitConfig

    base: dict = _load_yaml(args.config) if args.config else {}
    if args.species:
        base["species"] = tuple(args.species.split(","))
    if args.fit_species:
        base["fit_species"] = tuple(args.fit_species.split(","))
    if args.steps is not None:
        base["max_steps"] = int(args.steps)
    if args.lr is not None:
        base["learning_rate"] = float(args.lr)
    if args.airmass is not None:
        base["airmass"] = float(args.airmass)
    if getattr(args, "lsf_kernel_file", None):
        base["lsf_kernel_file"] = args.lsf_kernel_file
    if getattr(args, "observatory", None):
        base["observatory"] = args.observatory
    if getattr(args, "fit_windows", None):
        base["fit_windows"] = _resolve_fit_windows(args.fit_windows)

    # Stellar template: --stellar-teff enables it; the rest override the node /
    # broadening. Flags take precedence over YAML.
    if getattr(args, "stellar_teff", None) is not None:
        base["stellar_enabled"] = True
        base["stellar_teff"] = float(args.stellar_teff)
    if getattr(args, "stellar_logg", None) is not None:
        base["stellar_logg"] = float(args.stellar_logg)
    if getattr(args, "stellar_feh", None) is not None:
        base["stellar_feh"] = float(args.stellar_feh)
    if getattr(args, "stellar_grid", None):
        base["stellar_grid"] = args.stellar_grid
    if getattr(args, "vsini", None) is not None:
        base["vsini"] = float(args.vsini)
    if getattr(args, "rv", None) is not None:
        base["rv_kms"] = float(args.rv)
    if getattr(args, "no_fit_rv", False):
        base["fit_rv"] = False
    if getattr(args, "no_fit_vsini", False):
        base["fit_vsini"] = False

    # YAML lists → tuples (FitConfig is frozen).
    for key in ("species", "fit_species"):
        if isinstance(base.get(key), list):
            base[key] = tuple(base[key])
    # ``fit_windows`` from YAML is a list of [min, max] lists → tuple of tuples.
    if isinstance(base.get("fit_windows"), list):
        base["fit_windows"] = tuple(tuple(w) for w in base["fit_windows"])

    valid = {f.name for f in dataclasses.fields(FitConfig)}
    unknown = set(base) - valid
    if unknown:
        raise SystemExit(f"Unknown FitConfig field(s): {sorted(unknown)}. Valid: {sorted(valid)}")
    return FitConfig(**base)


def _add_common_fit_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("fits", help="Input FITS file (multi-order spectrum).")
    p.add_argument("--config", help="Optional YAML config file mirroring FitConfig.")
    p.add_argument("--species", help="Comma-separated species list (overrides --config).")
    p.add_argument("--fit-species", dest="fit_species",
                   help="Comma-separated species columns to FIT (subset of --species).")
    p.add_argument("--steps", type=int, help="SVI steps (overrides --config).")
    p.add_argument("--lr", type=float, help="Adam learning rate.")
    p.add_argument("--airmass", type=float,
                   help="Override airmass (default: FITS header, fall back to 1.0).")
    p.add_argument("--lsf-kernel", dest="lsf_kernel_file",
                   help="Empirical LSF kernel file (2 columns: velocity[km/s], amplitude); "
                        "selects the 'custom' LSF profile.")
    p.add_argument("--observatory",
                   help="Astropy site name (e.g. 'paranal', 'keck') supplying the "
                        "observatory altitude and barycentric RV; required to build the model.")
    p.add_argument("--wave-range", dest="wave_range",
                   help="Wavelength range MIN,MAX in nm (e.g. 2050,2500).")
    p.add_argument("--fit-windows", dest="fit_windows",
                   help="Restrict the SVI fit to the orders overlapping these "
                        "wavelength windows (the correction still spans the full "
                        "range). Semicolon-separated MIN,MAX pairs in nm "
                        "(e.g. '2050,2080;2150,2180').")
    p.add_argument("--device", choices=("cpu", "gpu"), help="Force JAX device.")
    p.add_argument("--out-dir", dest="out_dir", default=".",
                   help="Output directory for fit artefacts (default: cwd).")
    # Stellar template (M dwarfs etc.). Passing --stellar-teff enables it.
    p.add_argument("--stellar-teff", dest="stellar_teff", type=float,
                   help="Stellar effective temperature (K); enables the stellar template.")
    p.add_argument("--stellar-logg", dest="stellar_logg", type=float, help="Stellar log g (cgs).")
    p.add_argument("--stellar-feh", dest="stellar_feh", type=float, help="Stellar [M/H].")
    p.add_argument("--stellar-grid", dest="stellar_grid", help="Stellar grid (default phoenix-newera).")
    p.add_argument("--vsini", type=float, help="Projected rotation vsini (km/s).")
    p.add_argument("--rv", type=float, help="Systemic RV (km/s, barycentric frame).")
    p.add_argument("--no-fit-rv", dest="no_fit_rv", action="store_true", help="Fix rv (don't fit).")
    p.add_argument("--no-fit-vsini", dest="no_fit_vsini", action="store_true", help="Fix vsini (don't fit).")


def _resolve_wave_range(arg: Optional[str]):
    if not arg:
        return None
    parts = arg.split(",")
    if len(parts) != 2:
        raise SystemExit(f"--wave-range expects MIN,MAX nm; got '{arg}'.")
    return float(parts[0]), float(parts[1])


def _resolve_fit_windows(arg: Optional[str]):
    """Parse '--fit-windows MIN,MAX;MIN,MAX' into a tuple of (min, max) pairs."""
    if not arg:
        return None
    windows = []
    for chunk in arg.split(";"):
        parts = chunk.split(",")
        if len(parts) != 2:
            raise SystemExit(
                f"--fit-windows expects ';'-separated MIN,MAX nm pairs; got '{arg}'.")
        windows.append((float(parts[0]), float(parts[1])))
    return tuple(windows)


def cmd_fit(args: argparse.Namespace) -> int:
    if args.device:
        from enuma import set_device
        set_device(args.device)

    from enuma import fit_spectrum, load_fits_spectra

    fit_config = _build_fitconfig(args)
    data = load_fits_spectra(args.fits, wave_range=_resolve_wave_range(args.wave_range))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = fit_spectrum(
        data, fit_config,
        save_params_json_path=str(out_dir / "best_fit.json"),
        save_spectrum_txt_path=str(out_dir / "best_fit.txt"),
    )
    result.plot(str(out_dir))
    logger.info("Fit artefacts written to %s", out_dir.resolve())
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    """Regenerate plots from a saved best-fit JSON + the original FITS."""
    import jax.numpy as jnp

    from enuma import FitResult, build_context, load_best_fit_params_json, load_fits_spectra
    from enuma.inference.sites import dry_species_of
    from enuma.state import params_to_sites

    fit_config = _build_fitconfig(args)
    data = load_fits_spectra(args.fits, wave_range=_resolve_wave_range(args.wave_range))
    context, config = build_context(data, fit_config)
    optimized = load_best_fit_params_json(args.params, species=fit_config.species)
    dry_species = dry_species_of(context, tuple(fit_config.species))

    obs_mask = jnp.asarray(np.asarray(data["mask"]) & (data["flux"] >= fit_config.saturation_mask_threshold))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = FitResult(
        params=optimized, sites=params_to_sites(optimized), losses=[], data=data,
        obs_mask=obs_mask, context=context, config=config, fit_config=fit_config,
        dry_species=dry_species,
    )
    result.plot(str(out_dir))
    logger.info("Plots written to %s", out_dir.resolve())
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    if args.device:
        from enuma import set_device
        set_device(args.device)

    from enuma import build_context, load_fits_spectra
    from enuma.model.forward import forward_model
    from enuma.inference.sites import default_params, dry_species_of

    fit_config = _build_fitconfig(args)
    data = load_fits_spectra(args.fits, wave_range=_resolve_wave_range(args.wave_range))
    context, config = build_context(data, fit_config)
    dry_species = dry_species_of(context, tuple(fit_config.species))
    params = default_params(context, config, dry_species)

    forward_model(params, context, config).block_until_ready()  # warm-up / compile
    n = max(1, args.iters)
    t0 = time.perf_counter()
    for _ in range(n):
        out = forward_model(params, context, config)
    out.block_until_ready()
    dt_ms = (time.perf_counter() - t0) / n * 1000
    print(f"forward_model: {dt_ms:.3f} ms/call ({n} iters)")
    print(f"orders={context.obs_wave.shape[0]} order_size={config.order_size} species={list(context.opacity_grids)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="enuma", description="JAX-based telluric forward-model fitter.")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="-v INFO, -vv DEBUG (default WARNING).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    fit_p = sub.add_parser("fit", help="Run a multi-order SVI fit.")
    _add_common_fit_flags(fit_p)
    fit_p.set_defaults(func=cmd_fit)

    plot_p = sub.add_parser("plot", help="Regenerate plots from a saved best-fit JSON.")
    _add_common_fit_flags(plot_p)
    plot_p.add_argument("--params", required=True, help="Path to best_fit.json.")
    plot_p.set_defaults(func=cmd_plot)

    bench_p = sub.add_parser("bench", help="Time forward_model on the input spectrum.")
    _add_common_fit_flags(bench_p)
    bench_p.add_argument("--iters", type=int, default=50, help="Forward-model iterations.")
    bench_p.set_defaults(func=cmd_bench)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
