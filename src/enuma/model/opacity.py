import logging
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Tuple

import h5py
import jax.numpy as jnp
import numpy as np
from jax import device_put

from enuma.model._paths import ENV_VAR, data_root

__all__ = [
    "OpacityGridInfo",
    "load_pyrox_opacity",
    "get_opacities",
]

logger = logging.getLogger("enuma.model.opacity")


class OpacityGridInfo(NamedTuple):
    """Grid-axis metadata for the (P, T) bilinear σ-interpolator.

    The cross-section table lives on a mesh that is uniform in T but may be
    non-uniform in pressure; `forward_model` uses these fields to locate each
    (p_centers, t_centers) layer in the mesh.
    """
    T_min: float
    T_step: float
    logP_nodes: np.ndarray   # (N_P,) ascending log10 P [dyne/cm^2]


def load_pyrox_opacity(
    filepath: str,
    species_name: str,
    wave_min: float = None,
    wave_max: float = None,
) -> Tuple[jnp.ndarray, OpacityGridInfo, jnp.ndarray]:
    """Load one pyROX opacity grid, slicing the wavelength axis on disk."""
    with h5py.File(filepath, 'r') as f:
        T_grid = f['T'][:]
        P_grid = f['P'][:] * 10.0          # Pa -> dyne/cm^2
        wave_grid_full = f['wave'][:] * 1e9  # m -> nm (1D, small; load fully)

        # Slice the (N_lambda, N_P, N_T) cross-section on disk so only the
        # requested wavelength chunk is read into RAM. +4 converts log10 sigma
        # from SI (m²) to cgs (cm²).
        if wave_min is not None and wave_max is not None:
            idx_start = np.searchsorted(wave_grid_full, wave_min)
            idx_end = np.searchsorted(wave_grid_full, wave_max, side='right')
            if idx_start >= idx_end:
                raise ValueError(f"No opacity data in range {wave_min}-{wave_max} nm")
            wave_grid = wave_grid_full[idx_start:idx_end]
            sigma_log10_cgs = f['log10(xsec)'][idx_start:idx_end, :, :] + 4.0
        else:
            wave_grid = wave_grid_full
            sigma_log10_cgs = f['log10(xsec)'][:] + 4.0

    # (N_lambda, N_P, N_T) -> (N_P, N_T, N_lambda), then to linear cgs in
    sigma_log10_cgs = np.transpose(sigma_log10_cgs, (1, 2, 0))
    sigma_linear_cgs = np.power(10.0, sigma_log10_cgs, dtype=np.float32)

    # Pressure nodes may be non-uniform but must be ascending for the lookup.
    p_order = np.argsort(P_grid)
    P_grid, sigma_linear_cgs = P_grid[p_order], sigma_linear_cgs[p_order]

    T_step = float(T_grid[1] - T_grid[0])
    if not np.allclose(np.diff(T_grid), T_step):
        raise ValueError(f"Temperature grid of {species_name} is not uniformly spaced.")

    grid_info = OpacityGridInfo(
        T_min=float(T_grid[0]),
        T_step=T_step,
        logP_nodes=np.log10(P_grid),
    )

    grid_values = device_put(jnp.array(sigma_linear_cgs, dtype=jnp.float32))
    wavelength_grid = np.asarray(wave_grid, dtype=np.float64)
    return grid_values, grid_info, wavelength_grid


def _same_pt_axes(a: OpacityGridInfo, b: OpacityGridInfo) -> bool:
    """Whether two grids share (P, T) axes; ``forward_model`` uses one set for all species."""
    return (a.logP_nodes.shape == b.logP_nodes.shape
            and np.allclose(a.logP_nodes, b.logP_nodes)
            and np.isclose(a.T_min, b.T_min) and np.isclose(a.T_step, b.T_step))


def get_opacities(species_list: list, wave_range: tuple,
                  opacity_dir: Optional[str] = None) -> Tuple[Dict, jnp.ndarray]:
    """Load several species and verify they share one wavelength grid.
    Returns ``({species: (grid_values, OpacityGridInfo)}, common_wave_grid)``.
    """
    opac_dir = Path(opacity_dir) if opacity_dir else data_root() / "opacities"

    opacity_grids = {}
    common_wave_grid = None
    for species in species_list:
        path = opac_dir / f"{species}.hdf5"
        if not path.exists():
            raise FileNotFoundError(
                f"Opacity grid for '{species}' not found at {path}. "
                f"Set ${ENV_VAR} or FitConfig.opacity_dir to the data directory.")
        try:
            grid, info, wave = load_pyrox_opacity(str(path), species, *wave_range)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", species, exc)
            continue

        if common_wave_grid is None:
            common_wave_grid, common_info = wave, info
        elif not jnp.allclose(wave, common_wave_grid, atol=1e-5):
            raise ValueError(f"Wavelength grid for {species} does not match previous species!")
        elif not _same_pt_axes(info, common_info):
            raise ValueError(f"(P, T) grid for {species} does not match previous species!")
        opacity_grids[species] = (grid, info)

    return opacity_grids, common_wave_grid
