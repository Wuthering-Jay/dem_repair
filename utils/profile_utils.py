"""Terrain profile export placeholders."""

from __future__ import annotations

from typing import Any


def extract_profile(dem: Any, start: tuple[int, int], end: tuple[int, int]) -> dict[str, Any]:
    """Return a placeholder terrain profile description."""
    return {"dem": dem, "start": start, "end": end, "status": "not implemented"}
