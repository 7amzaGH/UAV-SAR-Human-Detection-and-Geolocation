"""
srt_reader.py
-------------
Reads DJI .SRT telemetry files and returns per-frame GPS data.

Real DJI Air 3S SRT format per subtitle block:
    FrameCnt: 1, DiffTime: 33ms
    2024-06-10 11:35:37.000
    [iso: 100] [shutter: 1/500.0] [fnum: 280] [ev: 0]
    [latitude: 50.290200] [longtitude: 18.664700]     <- DJI misspells longitude
    [rel_alt: 28.300 abs_alt: 312.412]                <- rel_alt = above ground (what we need)

IMPORTANT: DJI SRT does NOT store heading.
We default heading to 0 (North). For accurate geolocation,
check your DJI Fly app flight record or fly with Video Caption ON.
"""

import re


def load_srt(srt_path):
    """
    Parse a DJI .SRT file and return a list of telemetry dicts, one per frame.

    Each dict contains:
        lat      — latitude in decimal degrees
        lon      — longitude in decimal degrees
        altitude — relative altitude above ground in meters (rel_alt)
        heading  — always 0.0 (not stored in DJI SRT files)
    """
    with open(srt_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    telemetry = []

    # Match latitude — standard spelling
    # Match longitude — DJI misspells it as "longtitude" (extra t), handle both
    # Match rel_alt — relative altitude above ground (more useful than abs_alt)
    pattern = re.compile(
        r"\[latitude\s*:\s*([\-\d.]+)\]"
        r".*?"
        r"\[lon[g]?titude\s*:\s*([\-\d.]+)\]"   # matches both longitude and longtitude
        r".*?"
        r"\[rel_alt\s*:\s*([\-\d.]+)",            # relative altitude above takeoff point
        re.DOTALL
    )

    for match in pattern.finditer(content):
        telemetry.append({
            "lat":      float(match.group(1)),
            "lon":      float(match.group(2)),
            "altitude": float(match.group(3)),
            "heading":  0.0,    # not available in SRT — assumed North
        })

    return telemetry


def get_frame_telemetry(telemetry, frame_id):
    """
    Get telemetry for a specific frame index.
    Clamps to last entry if frame_id exceeds SRT length.
    Returns None if telemetry list is empty (SRT not found or wrong format).
    """
    if not telemetry:
        return None
    index = min(frame_id, len(telemetry) - 1)
    return telemetry[index]
