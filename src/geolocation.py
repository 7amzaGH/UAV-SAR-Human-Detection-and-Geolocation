import math

# Earth circumference in meters
EARTH_CIRCUMFERENCE = 40_075_000

def get_real_coords(bbox_center, drone_position, drone_heading, gimbal_pitch, camera_config):
    """
    Convert a bounding box pixel position to GPS coordinates.
    
    bbox_center:    (cx, cy) pixel coordinates of the detection
    drone_position: (latitude, longitude, altitude) of the drone
    drone_heading:  compass heading in degrees (0 = North)
    gimbal_pitch:   angle from horizon (e.g., -90 for Nadir, -45 for oblique)
                    Note: Adjust mapping based on your specific gimbal data.
    camera_config:  dict with fov_h, fov_v, image_width, image_height
    """
    drone_lat, drone_lon, alt_m = drone_position
    cx, cy = bbox_center

    W = camera_config["image_width"]
    H = camera_config["image_height"]

    # 1. Calculate Ground Sample Distance (GSD) per pixel
    gsd_x = (2 * alt_m * math.tan(math.radians(camera_config["fov_h"]) / 2)) / W
    gsd_y = (2 * alt_m * math.tan(math.radians(camera_config["fov_v"]) / 2)) / H

    # 2. Pixel offsets from image center
    # delta_y is negated to flip image coordinates to Cartesian (Up is Positive)
    delta_x = cx - W / 2
    delta_y = -(cy - H / 2)

    # 3. Convert pixel offsets to real-world meters in the image frame
    dx_m = delta_x * gsd_x
    dy_m = delta_y * gsd_y

    # 4. Apply Gimbal Pitch Correction
    # If gimbal_pitch follows DJI standard (0=horizon, -90=down), 
    # we convert it to an offset forward from the drone.
    # We use -pitch because our tangent logic expects a positive angle from Nadir.
    # Map your 90->0 and 45->-45 logic here if needed.
    if gimbal_pitch != 0:
        dy_m += alt_m * math.tan(math.radians(-gimbal_pitch))

    # 5. Rotate by heading to align with North/East (World Frame)
    heading_rad = math.radians(drone_heading)
    north_m = dx_m * (-math.sin(heading_rad)) + dy_m * math.cos(heading_rad)
    east_m  = dx_m * math.cos(heading_rad) + dy_m * math.sin(heading_rad)

    # 6. Convert North/East meters to Latitude/Longitude degrees
    meters_per_deg_lat = EARTH_CIRCUMFERENCE / 360
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(drone_lat))

    real_lat = drone_lat + (north_m / meters_per_deg_lat)
    real_lon = drone_lon + (east_m / meters_per_deg_lon)

    return real_lat, real_lon
