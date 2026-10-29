from __future__ import annotations

import math
from datetime import date, datetime, timedelta


def solar_zenith_deg(lat: float, lon: float, moment_utc: datetime) -> float:
    """NOAA solar-position approximation. `moment_utc` must be UTC."""
    doy = moment_utc.timetuple().tm_yday
    hour = moment_utc.hour + moment_utc.minute / 60.0 + moment_utc.second / 3600.0
    gamma = 2.0 * math.pi / 365.0 * (doy - 1 + (hour - 12.0) / 24.0)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2.0 * gamma)
        - 0.040849 * math.sin(2.0 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2.0 * gamma)
        + 0.000907 * math.sin(2.0 * gamma)
        - 0.002697 * math.cos(3.0 * gamma)
        + 0.00148 * math.sin(3.0 * gamma)
    )
    time_offset = eqtime + 4.0 * lon
    true_solar_time = hour * 60.0 + time_offset
    hour_angle = math.radians(true_solar_time / 4.0 - 180.0)
    lat_r = math.radians(lat)
    cos_zenith = math.sin(lat_r) * math.sin(decl) + math.cos(lat_r) * math.cos(decl) * math.cos(hour_angle)
    cos_zenith = min(1.0, max(-1.0, cos_zenith))
    return math.degrees(math.acos(cos_zenith))


def assumed_utc(day: date, local_hour: int, local_minute: int, utc_offset_hours: float) -> datetime:
    local = datetime(day.year, day.month, day.day, local_hour, local_minute)
    return local - timedelta(hours=utc_offset_hours)
