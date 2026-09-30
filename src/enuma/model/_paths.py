"""Resolution of filesystem paths: the bundled ``data/`` dir + the writable cache.
"""

import os
from pathlib import Path

ENV_VAR = "ENUMA_DATA_DIR"


def data_root() -> Path:
    """The ``data/`` directory: ``$ENUMA_DATA_DIR`` if set, else repo-root ``data/``."""
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parents[3] / "data"


CACHE_ENV_VAR = "ENUMA_CACHE"


def cache_root(cache_dir: str = None, subdir: str = None) -> Path:
    """Writable cache for fetched reanalysis / stellar data (created if absent).

    ``cache_dir`` wins; else ``$ENUMA_CACHE``; else ``~/.cache/enuma``. ``subdir``
    (e.g. ``"phoenix-newera"``) is appended.
    """
    if cache_dir:
        root = Path(cache_dir)
    else:
        env = os.environ.get(CACHE_ENV_VAR)
        root = Path(env).expanduser() if env else Path.home() / ".cache" / "enuma"
    if subdir:
        root = root / subdir
    root.mkdir(parents=True, exist_ok=True)
    return root
