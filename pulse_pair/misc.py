import datetime
import numpy as np

def round_to_hour(dt):
    dt_start_of_hour = dt.replace(minute=0, second=0, microsecond=0)
    dt_half_hour = dt.replace(minute=30, second=0, microsecond=0)

    if dt >= dt_half_hour:
        # round up
        dt = dt_start_of_hour + datetime.timedelta(hours=1)
    else:
        # round down
        dt = dt_start_of_hour

    return dt

def normalize_angle(angle):
    """
    Normalize the angle to be between -180 and 180 degrees.
    """
    return (angle + 180) % 360 - 180

def angular_difference(a, b):
    """
    Calculate the angular difference between angles a and b.
    """
    normalized_a = normalize_angle(a + 90)  # add 90 degrees as radar is right looking
    normalized_b = normalize_angle(b + 180) # add 180 degrees to go from blowing towards, to blowing from

    # Calculate the angular difference
    diff = (normalized_b - normalized_a + 180) % 360 - 180

    return diff

def calculate_distance(x, y, x0=0 , y0=0):
    return np.sqrt((x - x0) ** 2 + (y - y0 ) ** 2)