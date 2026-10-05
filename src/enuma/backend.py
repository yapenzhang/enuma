import logging
import os

__all__ = [
    "set_device",
    "set_num_threads",
]

_THREADS_ENV_VAR = "ENUMA_NUM_THREADS"
_BLAS_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS")


def _export_thread_limit(n: int):
    # XLA sizes its CPU thread pool from NPROC when the backend initializes;
    # BLAS reads its variables when numpy is first imported.
    os.environ["NPROC"] = str(n)
    for var in _BLAS_THREAD_VARS:
        os.environ.setdefault(var, str(n))


# Applied before jax/numpy are imported so the BLAS limits bind as well.
if os.environ.get(_THREADS_ENV_VAR):
    _export_thread_limit(int(os.environ[_THREADS_ENV_VAR]))

import jax  # noqa: E402
import numpyro  # noqa: E402

# Backend discovery logs a TPU/GPU probe failure at INFO on every CPU-only run.
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)


def set_device(device: str = "cpu"):
    """Globally configure JAX and NumPyro to use 'cpu' or 'cuda'.

    Must be called BEFORE any JAX arrays are created or compiled.
    """
    device = device.lower()
    if device not in ('cpu', 'cuda'):
        raise ValueError("Device must be 'cpu' or 'cuda'")

    jax.config.update("jax_platform_name", device)
    numpyro.set_platform(device)
    print(f"Configured to use platform: {jax.default_backend().upper()}")


def set_num_threads(n: int):
    """Cap the number of CPU threads JAX/XLA uses (the ``top`` CPU% ceiling is ``100*n``).

    Must be called BEFORE the first JAX computation; importing ``enuma`` is fine.
    Alternatively export ``ENUMA_NUM_THREADS=n`` before starting Python, which
    also caps numpy's BLAS threads.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    from jax._src import xla_bridge
    if getattr(xla_bridge, "_backends", None):
        raise RuntimeError(
            "set_num_threads() must be called before the first JAX computation; "
            f"export {_THREADS_ENV_VAR}={n} before starting Python instead.")
    _export_thread_limit(n)
