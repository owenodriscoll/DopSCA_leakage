import os
import sys
import glob
import pyproj
import cdsapi
import types
import xarray as xr
import numpy as np

from .misc import round_to_hour, angular_difference
from .conversions import convert_to_0_360


"""
run file using command line as 

python era5_download.py {year} {month} {directory}

where "year" and "month" are obligatory and input types should preferably be strings

"""


def era5_wind_area(year, month, day, time, latmax, lonmin, latmin, lonmax, filename):
    """
    add trailing '/' to directory

    Downloads ERA5 data for an area given a specific time and extent
    """

    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": [
                "temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
            ],
            "year": str(year),
            "month": str(month),
            "day": str(day),
            "time": str(time),
            "area": [latmax, lonmin, latmin, lonmax],
        },
        f"{filename}",
    )
    return


def era5_wind_monthly(year, month, directory=""):
    import cdsapi

    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": [
                "temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
            ],
            "year": str(year),
            "month": str(month),
            "day": [
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "08",
                "09",
                "10",
                "11",
                "12",
                "13",
                "14",
                "15",
                "16",
                "17",
                "18",
                "19",
                "20",
                "21",
                "22",
                "23",
                "24",
                "25",
                "26",
                "27",
                "28",
                "29",
                "30",
                "31",
            ],
            "time": [
                "00:00",
                "01:00",
                "02:00",
                "03:00",
                "04:00",
                "05:00",
                "06:00",
                "07:00",
                "08:00",
                "09:00",
                "10:00",
                "11:00",
                "12:00",
                "13:00",
                "14:00",
                "15:00",
                "16:00",
                "17:00",
                "18:00",
                "19:00",
                "20:00",
                "21:00",
                "22:00",
                "23:00",
            ],
        },
        f"{directory}era5_wind_{str(year)+str(month)}.nc",
    )
    return


def querry_era5(
    date,
    latitudes,
    longitudes,
    grid_spacing,
    era5_file: str = None,
    directory: str = "",
    era5_smoothing_window: int = None,
    era5_undersample_factor: int = 10,
):
    """
    Opens or downloads relevant ERA5 data to find the wind direction w.r.t. to the radar.
    1. Will first attempt to load a specific file (if provided).
    2. Otherwise will check if previously downloaded file covers area of interest.
    3. Lastely, will submit new download request.

    This method can be skipped by manually providing a "wdir_wrt_sensor" attribute to "".
    Either an integer/float as an average over the area of interest, or an array, in which case the
    array must be the same shape as the backscatter array in the S1_file.

    We do some ERA5 interpolating to the NRCS grid with some arbitrary choiced, these don't really affect results since ERA5 is coarse to begin with
    """

    # set smoothing window of era5 based on grid size of loaded S1 data
    # a bit arbitrary values, but they dont really matter
    if type(era5_smoothing_window) == types.NoneType:
        era5_smoothing_window = int((200 / grid_spacing) * 15)

    # NOTE this is not nice because different dimension order will cause it to break
    var_azi, var_grg = list(latitudes.sizes)
    assert list(latitudes.sizes) == list(longitudes.sizes)

    date_rounded = round_to_hour(date)
    yy, mm, dd, hh = (
        date_rounded.year,
        date_rounded.month,
        date_rounded.day,
        date_rounded.hour,
    )
    time = f"{hh:02}00"

    latmean, lonmean = latitudes.mean().values * 1, longitudes.mean().values * 1
    latmin, latmax = latitudes.min().values * 1, latitudes.max().values * 1
    lonmin, lonmax = longitudes.min().values * 1, longitudes.max().values * 1
    lonmin, lonmax = [
        convert_to_0_360(i) for i in [lonmin, lonmax]
    ]  # NOTE correction for fact that ERA5 goes between 0 - 360

    if type(era5_file) == str:
        era5 = xr.open_dataset(era5_file)
    else:
        sub_str = str(yy) + str(mm)
        try:
            # try to find if monthly data file exists which to load
            era5_filename = [
                s for s in glob.glob(f"{directory}*") if sub_str + ".nc" in s
            ][0]
        except:
            #  if not, try to find single estimate hour ERA5 wind file
            era5_filename = (
                f"era5_{yy}{mm:02d}{dd:02d}h{time}_lat{latmean:.1f}_lon{lonmean:.1f}.nc"
            )
            era5_filename = era5_filename.replace(".", "_", 2)
            if not directory is None:
                era5_filename = os.path.join(directory, era5_filename)

            # if neither monthly file nor hourly single estimate file exist, download new single estimate hour
            if not os.path.isfile(era5_filename):
                era5_wind_area(
                    year=yy,
                    month=mm,
                    day=dd,
                    time=time,
                    lonmin=lonmin,
                    lonmax=lonmax,
                    latmin=latmin,
                    latmax=latmax,
                    filename=era5_filename,
                )

        print(
            f"Loading nearest ERA5 point w.r.t. observation from ERA5 file: {era5_filename}"
        )
        era5 = xr.open_dataset(era5_filename)

    era5_subset_time = era5.sel(
        time=np.datetime64(date_rounded, "ns"), method="nearest"
    )

    era5_subset = era5_subset_time.sel(
        longitude=convert_to_0_360(longitudes),
        latitude=latitudes,
        method="nearest",
    )

    # ERA5 data is subsampled
    # this should not affect resolution much as, for example, the resolution of 1/4 deg ERA5 is approx 50km,
    # The resampled grid size should still be ~50km (S1 resolution * era5_smoothing_window * era5_undersample_factor ~ 50km)
    resolution_condition = (
        grid_spacing * era5_undersample_factor * (1 * era5_smoothing_window)
    )  # NOTE * 1 because only mean window operations considered

    # checks whether resampling is unacceptable
    # resolution is ideally twice the grid spacing and 1 degree is approximately 100km
    resolution_era5_deg = 2 * min(
        [
            abs(era5.latitude.diff(dim="latitude")).min(),
            abs(era5.longitude.diff(dim="longitude")).min(),
        ]
    )
    resolution_era5_m = 100e3 * resolution_era5_deg
    message = (
        lambda x: f"Warning: interpolated ERA5 data may be {x[0]}. \nConsider plotting fields in era5 for inspection and/or {x[1]} the undersample_factor or the smoothing_window size."
    )

    if (
        resolution_condition >= 1.33 * resolution_era5_m
    ):  # arbitrary threshold at which ERA5 resolution may be degraded
        print(message(["over-smoothed", "decreasing"]))
    elif (
        resolution_condition <= 0.50 * resolution_era5_m
    ):  # arbitrary threshold at which grainy ERA5 is expected (maybe cause artifacts)
        print(message(["under-smoothed", "increasing"]))

    # create a placeholder dataset to avoid having to subsample/oversample on datetime vectors
    azimuth_time_placeholder = np.arange(era5_subset.sizes[var_azi])
    ground_range_placeholder = np.arange(era5_subset.sizes[var_grg])
    era5_placeholder = era5_subset.assign_coords(
        {var_azi: azimuth_time_placeholder, var_grg: ground_range_placeholder}
    )

    # Subsample the dataset by interpolation
    new_azimuth_time = (
        np.arange(era5_subset.sizes[var_azi] / era5_undersample_factor)
        * era5_undersample_factor
    )
    new_ground_range = (
        np.arange(era5_subset.sizes[var_grg] / era5_undersample_factor)
        * era5_undersample_factor
    )
    era5_subsamp = era5_placeholder.interp(
        {var_azi: new_azimuth_time, var_grg: new_ground_range}, method="linear"
    )

    # first perform median filter (performs better if anomalies are present),then mean filter to smooth edges
    era5_smoothed = era5_subsamp.rolling(
        {var_azi: era5_smoothing_window, var_grg: era5_smoothing_window},
        center=True,
        min_periods=2,
    ).median()
    era5_smoothed = era5_smoothed.rolling(
        {var_azi: era5_smoothing_window, var_grg: era5_smoothing_window},
        center=True,
        min_periods=2,
    ).mean()

    # re-interpolate to the native resolution and add to object
    era5_interp = era5_smoothed.interp(
        {var_azi: azimuth_time_placeholder, var_grg: ground_range_placeholder},
        method="linear",
    )
    era5_data = era5_interp.assign_coords(
        {var_azi: era5_subset[var_azi], var_grg: era5_subset[var_grg]}
    )
    return era5_data


def wdir_from_era5(era5_data, corner_coords):
    """
    Calculates the wind direction with respect to the sensor using the ground geometry and era5 wind vectors
    """

    wdir_era5 = np.rad2deg(np.arctan2(era5_data.u10, era5_data.v10))

    # compute ground footprint direction
    geodesic = pyproj.Geod(ellps="WGS84")
    ground_dir, _, _ = geodesic.inv(*corner_coords)

    # compute directional difference between satelite and era5 wind direction
    wdir_wrt_sensor = angular_difference(ground_dir, wdir_era5)

    return wdir_wrt_sensor


if __name__ == "__main__":
    from .misc import era5_wind_monthly

    year = sys.argv[1]
    month = sys.argv[2]

    try:
        directory = sys.argv[3]
    except:
        directory = ""

    era5_wind_monthly(
        year=year,
        month=month,
        directory=directory,
    )
