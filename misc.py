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


def era5_wind_monthly(year, month, directory = ''):
    import cdsapi
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                'temperature', '10m_u_component_of_wind', '10m_v_component_of_wind',
            ],
            'year': str(year),
            'month': str(month),
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30', '31',
            ],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
        },
        f'{directory}era5_wind_{str(year)+str(month)}.nc')
    return
    

def era5_wind_point(year, month, day, time, lat, lon, filename):
    """
    add trailing '/' to directory
    """
    import cdsapi
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                'temperature', '10m_u_component_of_wind', '10m_v_component_of_wind',
            ],
            'year': str(year),
            'month': str(month),
            'day': str(day),
            'time': str(time),
            'area': [lat, lon, lat, lon],
        },
        f'{filename}')
    return

def era5_wind_area(year, month, day, time, latmax, lonmin, latmin, lonmax, filename):
    """
    add trailing '/' to directory

    Downloads ERA5 data for an area given a specific time and extent
    """
    import cdsapi
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                'temperature', '10m_u_component_of_wind', '10m_v_component_of_wind',
            ],
            'year': str(year),
            'month': str(month),
            'day': str(day),
            'time': str(time),
            'area': [latmax, lonmin, latmin, lonmax],
        },
        f'{filename}')
    return
