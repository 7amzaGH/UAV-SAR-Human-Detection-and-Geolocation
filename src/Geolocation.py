import math

# Earth circumference in meters
EARTH_CIRCUMFERENCE = 40_075_000

def get_scale_x(altitude, fov_h, image_width):
    return (altitude * math.tan(math.radians(fov_h / 2))) / (image_width / 2)

def get_scale_y(altitude, fov_v, image_height):
    return (altitude * math.tan(math.radians(fov_v / 2))) / (image_height / 2)

def rotate(dist_x, dist_y, heading):
    theta = math.radians(heading)
    x_rotated = math.cos(theta) * dist_x - math.sin(theta) * dist_y
    y_rotated = math.sin(theta) * dist_x + math.cos(theta) * dist_y
    return x_rotated, y_rotated

def get_real_coords(bbox_center, drone_position, drone_heading, camera_config):
    """
    Convert a bounding box pixel position to GPS coordinates.
    
    bbox_center:    (cx, cy) pixel coordinates of the detected person
    drone_position: (latitude, longitude, altitude) of the drone
    drone_heading:  compass heading in degrees (0 = North)
    camera_config:  dict with fov_h, fov_v, image_width, image_height
    """
    drone_lat, drone_lon, altitude = drone_position
    cx, cy = bbox_center

    image_width  = camera_config["image_width"]
    image_height = camera_config["image_height"]

    # Pixel offset from the center of the image
    offset_x = cx - image_width  / 2
    offset_y = cy - image_height / 2

    # Convert pixel offset to meters
    scale_x = get_scale_x(altitude, camera_config["fov_h"], image_width)
    scale_y = get_scale_y(altitude, camera_config["fov_v"], image_height)

    dist_x = offset_x * scale_x
    dist_y = offset_y * scale_y

    # Rotate to align with North based on drone heading
    dist_x, dist_y = rotate(dist_x, dist_y, drone_heading)

    # Convert meters to latitude/longitude degrees
    delta_lat = (dist_y / EARTH_CIRCUMFERENCE) * 360
    delta_lon = (dist_x / (EARTH_CIRCUMFERENCE * math.cos(math.radians(drone_lat)))) * 360

    real_lat = drone_lat + delta_lat
    real_lon = drone_lon + delta_lon

    return real_lat, real_lon
