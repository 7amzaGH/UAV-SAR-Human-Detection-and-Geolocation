"""
srt_reader.py
-------------
Parses DJI .SRT telemetry files into per-frame GPS dictionaries.

DJI SRT format (Air 3S example):
    FrameCnt: 1, DiffTime: 33ms
    2024-06-10 11:35:37.000
    [iso: 100] [shutter: 1/500.0] [fnum: 280] [ev: 0]
    [latitude: 50.2902] [longtitude: 18.6647]   ← DJI typo: "longtitude"
    [rel_alt: 28.300 abs_alt: 312.412]

Note: heading is NOT stored in DJI SRT files.
      It defaults to 0.0 (North). For accurate geolocation provide
      heading from your flight logs or a compass/IMU source.
"""

import re


def load_srt(srt_path):
    """
    Parse a DJI .SRT file.

    Returns
    -------
    List of dicts with keys: lat, lon, altitude (rel_alt), heading (0.0).
    Returns an empty list if no telemetry blocks are found.
    """
    with open(srt_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Handles both "longitude" and DJI's misspelling "longtitude"
    pattern = re.compile(
        r"\[latitude\s*:\s*([\-\d.]+)\]"
        r".*?"
        r"\[lon[g]?titude\s*:\s*([\-\d.]+)\]"
        r".*?"
        r"\[rel_alt\s*:\s*([\-\d.]+)",
        re.DOTALL,
    )

    telemetry = [
        {
            "lat":      float(m.group(1)),
            "lon":      float(m.group(2)),
            "altitude": float(m.group(3)),
            "heading":  0.0,
        }
        for m in pattern.finditer(content)
    ]

    if not telemetry:
        print(
            "[srt_reader] WARNING: No telemetry found.\n"
            "             Make sure 'Video Caption' is ON in DJI camera settings."
        )

    return telemetry


def get_frame_telemetry(telemetry, frame_id):
    """
    Return telemetry for a given logical frame index.
    Clamps to the last entry if frame_id exceeds the SRT length.
    Returns None if telemetry is empty.
    """
    if not telemetry:
        return None
    return telemetry[min(frame_id, len(telemetry) - 1)]