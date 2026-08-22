"""
src/config_loader.py

Role
----
The single, shared way every module in SENTRIX reads configuration.
No module opens config.yaml directly — they all call load_config().

Why this exists as its own file
--------------------------------
If ten modules each wrote their own `open("config.yaml")` logic, changing
the config file's location or format would mean editing ten files. One
loader means one place to change.
"""

from pathlib import Path
from typing import Any, Dict
import yaml

# Resolved relative to the project root regardless of where a script is run from
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"

_cached_config: Dict[str, Any] | None = None


def load_config(path: str | Path = _DEFAULT_CONFIG_PATH, force_reload: bool = False) -> Dict[str, Any]:
    """
    Load config.yaml into a plain dict.

    Parameters
    ----------
    path : str | Path
        Location of the YAML file. Defaults to <project_root>/config/config.yaml.
    force_reload : bool
        If False (default), returns a cached copy after the first call so
        repeated imports across modules don't re-read the file from disk.

    Returns
    -------
    dict
        The parsed configuration, e.g. config["mysql"]["host"].
    """
    global _cached_config

    if _cached_config is not None and not force_reload:
        return _cached_config

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path}")

    with open(path, "r") as f:
        _cached_config = yaml.safe_load(f)

    return _cached_config


def get_project_root() -> Path:
    """Absolute path to the project root — useful for building file paths safely."""
    return _PROJECT_ROOT


if __name__ == "__main__":
    cfg = load_config()
    print("Config loaded successfully. Top-level keys:", list(cfg.keys()))
