import datetime
import numpy as np
import xarray as xr
import pandas as pd
import joblib
import sys
import io
import warnings
import xrft


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
    normalized_b = normalize_angle(
        b + 180
    )  # add 180 degrees to go from blowing towards, to blowing from

    # Calculate the angular difference
    diff = (normalized_b - normalized_a + 180) % 360 - 180

    return diff


def calculate_distance(x, y, x0=0, y0=0):
    return np.sqrt((x - x0) ** 2 + (y - y0) ** 2)


def warning_catcher_ML(function, *args):
    """
    Function to catch all the numerous warnings the ML scripts yield...
    """
    with warnings.catch_warnings(record=True) as caught_warnings:
        output_buffer = io.StringIO()
        stdout_save = sys.stdout
        sys.stdout = output_buffer

        try:
            result = function(*args)
        except Exception as e:
            sys.stdout = stdout_save
            print(e)

        sys.stdout = stdout_save

    return result


def prediction_ML(
    filename: str, model_ML: str, field_data: str = "nrcs_scat_subscene_w_noise"
):

    data = xr.open_dataset(filename)
    model = joblib.load(model_ML)

    # slow time can not be indexed for concatonation step, remove indexing if present
    try:
        data = data.reset_index("slow_time")
    except:
        pass

    # select samples along azimuth to fill azimuth beam footpint cutoff
    n = int(data.az_mask_cutoff / data.stride / 2)

    # select spaced data
    X_data = xr.concat(
        [
            data[field_data]
            .isel(slow_time=slice(i - n, 1 + i + n))
            .drop_vars("slow_time")
            for i in range(n, len(data[field_data]["slow_time"]) - n)
        ],
        dim="placeholder",
    )

    # reshape data and turn in M * (2n + 1) dataframe
    X_test_pre = X_data.values.reshape(-1, 2 * n + 1)
    df_data = pd.DataFrame(X_test_pre)

    # exclude indexes containing NaN's, these cannot be fed to ML algorithm
    indexes_non_nan = df_data[~df_data.isna().any(axis=1)].index.values
    df_data.dropna(inplace=True)

    # perform predict using a warning catcher
    pred = warning_catcher_ML(model.predict, df_data)

    # add prediction to empty field, accounting for indexes
    empty_field = np.nan * np.ones_like(data[field_data]).ravel()
    empty_field[indexes_non_nan] = pred

    # reconstruct original shape
    result = empty_field.reshape(*data[field_data].T.shape)

    # account for missing "n" rows at the bottom
    result = np.roll(result, shift=n, axis=0)

    # add to data
    data["estimate"] = (["grg", "slow_time"], result.T)
    return data


def power_spectrum_custom(
    da,
    dim=None,
    scaling="spectrum",
    detrend="constant",
    window="hann",
    window_correction="True",
):
    condition_fill = np.isfinite(da)
    da_filled = xr.where(condition_fill, da, 0)

    p2 = xrft.power_spectrum(
        da=da_filled,
        dim=dim,
        scaling=scaling,
        detrend=detrend,
        window=window,
        window_correction=window_correction,
    )
    return p2
