from __future__ import annotations

from typing import Optional


def geotag_detection(
    lat: float,
    lon: float,
    heading_deg: float,
    signed_cross_track_m: float,
) -> tuple[Optional[float], Optional[float]]:
    """Project a cross-track sonar contact from sensor position to WGS84.

    Positive signed range is starboard; negative is port. This is a first-order
    geotag and assumes the supplied platform position is already the sonar/towfish
    position (or has been layback-corrected).
    """
    try:
        from pyproj import Geod
    except ImportError:
        return None, None

    bearing = (float(heading_deg) + (90.0 if signed_cross_track_m >= 0 else -90.0)) % 360.0
    geod = Geod(ellps="WGS84")
    out_lon, out_lat, _ = geod.fwd(float(lon), float(lat), bearing, abs(float(signed_cross_track_m)))
    return float(out_lat), float(out_lon)
