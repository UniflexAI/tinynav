"""Helpers for pois.json entries."""
from __future__ import annotations


def poi_not_display(poi: dict) -> bool:
    """True when this POI should be hidden from the frontend UI."""
    return bool(poi.get('not_display', False))


def visible_pois(pois: dict) -> list[dict]:
    return [
        poi for poi in pois.values()
        if isinstance(poi, dict) and not poi_not_display(poi)
    ]
