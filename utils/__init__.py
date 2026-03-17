"""Utility package for DEM repair.

Keep package-level imports lightweight so preprocessing scripts do not
implicitly require the full training stack.
"""

from .config_utils import get_repo_root, load_yaml, resolve_path

__all__ = ["get_repo_root", "load_yaml", "resolve_path"]
