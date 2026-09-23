"""
load_config(): the one way every module reads config/config.yaml.
"""

from pathlib import Path
from typing import Any, Dict
import yaml

# Resolved from the project root, wherever the script is run from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"

_cached_config: Dict[str, Any] | None = None


def load_config(path: str | Path = _DEFAULT_CONFIG_PATH, force_reload: bool = False) -> Dict[str, Any]:
    """
    Load config.yaml into a dict. Cached after the first call unless
    force_reload=True.
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
    """Absolute path to the project root."""
    return _PROJECT_ROOT


if __name__ == "__main__":
    cfg = load_config()
    print("Config loaded successfully. Top-level keys:", list(cfg.keys()))
