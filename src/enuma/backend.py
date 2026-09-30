import jax
import numpyro

__all__ = [
    "set_device",
]


def set_device(device: str = "cpu"):
    """Globally configure JAX and NumPyro to use 'cpu' or 'gpu'.

    Must be called BEFORE any JAX arrays are created or compiled.
    """
    device = device.lower()
    if device not in ('cpu', 'gpu'):
        raise ValueError("Device must be 'cpu' or 'gpu'")

    jax.config.update("jax_platform_name", device)
    numpyro.set_platform(device)
    print(f"Configured to use platform: {jax.default_backend().upper()}")
