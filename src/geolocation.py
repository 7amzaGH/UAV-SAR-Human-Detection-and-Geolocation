"""
geolocation.py
--------------
Converts a bounding-box pixel position into real-world GPS coordinates
using drone telemetry (position, heading, altitude) and camera intrinsics.
"""

import math

EARTH_CIRCUMFERENCE = 40_075_000  # metres


def get_real_coords(bbox_center, drone_position, drone_heading,
                    gimbal_pitch, camera_config):
    """
    Project a detected pixel position to GPS coordinates.

    Parameters
    ----------
    bbox_center    : (cx, cy) — pixel centre of the detection
    drone_position : (lat, lon, alt_m) — drone GPS + altitude AGL
    drone_heading  : float — compass heading in degrees (0 = North)
    gimbal_pitch   : float — 0 for nadir, negative for oblique below horizon
                     e.g. pass -45 for a 45-degree oblique shot
    camera_config  : dict with keys fov_h, fov_v, image_width, image_height

    Returns
    -------
    (lat, lon) — estimated GPS coordinates of the detected object
    """
    drone_lat, drone_lon, alt_m = drone_position
    cx, cy = bbox_center
    W = camera_config["image_width"]
    H = camera_config["image_height"]

    # Ground sample distance (metres per pixel) at nadir
    gsd_x = (2 * alt_m * math.tan(math.radians(camera_config["fov_h"] / 2))) / W
    gsd_y = (2 * alt_m * math.tan(math.radians(camera_config["fov_v"] / 2))) / H

    # Pixel offset from image centre → real-world offset in camera frame
    # dy is negated: image y grows downward, world y grows upward
    dx_m = (cx - W / 2) * gsd_x
    dy_m = -(cy - H / 2) * gsd_y

    # Oblique gimbal correction: shift the ground footprint forward
    if gimbal_pitch != 0:
        dy_m += alt_m * math.tan(math.radians(-gimbal_pitch))

    # Rotate camera frame offsets into world North/East frame
    heading_rad = math.radians(drone_heading)
    north_m = -dx_m * math.sin(heading_rad) + dy_m * math.cos(heading_rad)
    east_m  =  dx_m * math.cos(heading_rad) + dy_m * math.sin(heading_rad)

    # Convert metres offset to degrees
    metres_per_deg_lat = EARTH_CIRCUMFERENCE / 360
    metres_per_deg_lon = metres_per_deg_lat * math.cos(math.radians(drone_lat))

    real_lat = drone_lat + north_m / metres_per_deg_lat
    real_lon = drone_lon + east_m  / metres_per_deg_lon

    return real_lat, real_lon


def haversine(lat1, lon1, lat2, lon2):
    """
    Great-circle distance between two GPS points in metres.
    """
    R = EARTH_CIRCUMFERENCE / (2 * math.pi)
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(d_lon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))