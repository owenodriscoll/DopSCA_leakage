import numpy as np


def convert_to_0_360(longitude):
    return (longitude + 360) % 360


def angular_projection_factor(inc_original, inc_new=90) -> float:
    """
    Computes multiplication factor to convert vector from one incidence to new one, e.g. from slant range to horizontal w.r.t. to the surface (if inc_new = 90)

    Input
    -----
    inc_original: float, array-like
        incidence angle w.r.t. horizontal of vector, in degrees
    inc_new: float,array-like
        new incidence angle in degrees. Defaults to 0 degrees (horizontal)

    Returns
    -------
    factor with which to multiply original vector to find projected vector's magnitude
    """
    return np.sin(np.deg2rad(inc_new)) / np.sin(np.deg2rad(inc_original))


def phase2vel(phase, wavenumber, T):
    """
    Converts a phase change within time T for a specific wavenumber to a velocity
    """
    return phase / 2 / wavenumber / T


def dop2vel(Doppler, Lambda, angle_incidence, angle_azimuth, degrees=True):
    """
    Computes velocity corresponding to Doppler shift based on eq. 4.34 from Digital Procesing of Synthetic Aperture Radar Data by Ian G. Cumming

    Input
    -----
    doppler: float,
        relative frequency shift in Hz of surface or object w.r.t to the other
    Lambda: float,
        Wavelength of radio wave, in m
    angle_incidence: float,
        incidence angle, in of wave with surface, degrees or radians
    angle_azimuth: float,
        azimuthal angle with respect to boresight (0 for right looking system)
    degrees: bool,
        whether input angles are provided in degrees or radians

    Return
    -------
    velocity: float,
        relative velocity in m/s of surface or object w.r.t to the other
    """

    if degrees:
        angle_azimuth, angle_incidence = [
            np.deg2rad(i) for i in [angle_azimuth, angle_incidence]
        ]

    return Lambda / 2 * Doppler / (np.sin(angle_azimuth) * np.sin(angle_incidence))


def vel2dop(velocity, Lambda, angle_incidence, angle_azimuth, degrees=True):
    """
    Computes Doppler shift corresponding to velocity based on eq. 4.34 from Digital Procesing of Synthetic Aperture Radar Data by Ian G. Cumming

    Input
    -----
    velocity: float,
        relative velocity in m/s of surface or object w.r.t to the other
    Lambda: float,
        Wavelength of radio wave, in m
    angle_incidence: float,
        incidence angle, in of wave with surface, degrees or radians
    angle_azimuth: float,
        azimuthal angle with respect to boresight (0 for right looking system)
    degrees: bool,
        whether input angles are provided in degrees or radians

    Returns
    -------
    Doppler: float,
        frequency shift corresponding to input geometry and velocity, in Hz
    """

    if degrees:
        angle_azimuth, angle_incidence = [
            np.deg2rad(i) for i in [angle_azimuth, angle_incidence]
        ]

    return 2 / Lambda * velocity * np.sin(angle_azimuth) * np.sin(angle_incidence)


def slant2ground(
    spacing_slant_range: float | int,
    height: float | int,
    ground_range_max: float | int,
    ground_range_min: float | int,
) -> float:
    """
    Converts a slant range pixel spacing to that projected onto the ground (assuming flat earth)

    Input
    -----
    spacing_slant_range:  float|int,
        slant range grid size, in meters
    height: float | int,
        height of platform, in meters
    ground_range_max: float|int,
        ground range projected maximum distance from satellite
    ground_range_min: float|int,
        ground range projected minimum distance from satellite

    Returns
    -------
    new_grg_pixel: float,
        new ground range pixel spacing, in meters
    """
    current_distance = ground_range_max
    new_grg_pixel = []

    # iteratively compute new pixel spacing starting from the maximum extend
    while current_distance > ground_range_min:
        new_grg_pixel.append(current_distance)
        new_incidence = np.arctan(current_distance / height)
        current_distance -= spacing_slant_range / np.sin(new_incidence)

    # reverse order to convert from decreasing to increasing ground ranges
    new_grg_pixel.reverse()

    return new_grg_pixel
