import xrft
import xarray as xr
import numpy as np
from typing import Callable
from scipy.signal import firwin
from .utils import is_chunked_checker, add_delayed_assert


def design_low_pass_filter_2D(
    da_shape: tuple[int, int],
    cutoff_frequency: float,
    fs_x: float,
    fs_y: float,
    window: str = "hann",
):
    """
    Design 2D window in time domain given sampling in x- and y-direction and desired cutoff frequency

    Input
    -----
    da_shape:  tuple[int, int],
       tuple containing shape of x- and y-dimension like (x_shape, y_shape)
    cutoff_frequency: float,
        threshold frequnecy, greater frequencies are filtered out
    fs_x: float,
        sampling along first DataArray dimension in tuple
    fs_y: float,
        sampling along first DataArray dimension in tuple
    window: str,
        window string from scipy.signal.get_window

    Returns
    -------
    filter_response: Array[float],
        2D array with time domain filter response
    """

    taps_x = firwin(
        numtaps=da_shape[0],
        cutoff=cutoff_frequency,
        fs=fs_x,
        pass_zero=True,
        window=window,
    )
    taps_y = firwin(
        numtaps=da_shape[1],
        cutoff=cutoff_frequency,
        fs=fs_y,
        pass_zero=True,
        window=window,
    )

    # generate 2D field
    filter_response = np.outer(taps_x, taps_y)

    return filter_response

def low_pass_filter(
    da: xr.DataArray,
    cutoff_frequency: float,
    fs: float,
    window: str = "hann",
    fill_nans: bool = False,
    return_complex: bool = False,
) -> xr.DataArray:
    """
    Low pass filtering an xarray dataArray in the Fourier domain using xrft, scipy.signal.windows and np.fft

    Input:
    ------
    da: xr.DataArray,
        Data to be filtered
    cutoff_frequency: float,
        threshold frequnecy, greater frequencies are filtered out
    fs: float,
        sampling along first DataArray dimension
    window: str,
        window string from scipy.signal.get_window
    fill_nans: bool,
        Whether to replace non-finite values (e.g. nans and inifinities) with 0
    return_complex: bool,
        Whether the imaginary part should be returned or not

    Returns:
    --------
    da_filt: xr.DataArray,
        The real part of low-pass filtered data
    """

    if fill_nans:
        condition_fill = np.isfinite(da)
        da_filled = xr.where(condition_fill, da, 0)
    else:
        da_filled = da

    dim = da_filled.dims[0]

    # data is rechunked because fourier transform cannot be performed over chunked dimension
    # shift set to false to prevent clashing fftshifts between np.fft and xrft.fft
    if is_chunked_checker(da_filled):
        da_spec = xrft.fft(
            da_filled.chunk({**da_filled.sizes}),
            dim = dim,
            chunks_to_segmentsbool=False,
            shift=False,
        )
    else:
        da_spec = xrft.fft(
            da_filled, 
            dim = dim,
            shift=False)

    # design time-domain filter
    numtaps = da_filled.sizes[dim]
    filter_response = firwin(
        numtaps=numtaps,
        cutoff=cutoff_frequency,
        fs=fs,
        pass_zero=True,
        window=window,
    )

    # convert window to fourier domain and multiply with spectrum, then convert back (i.e. same as convolving filter with input image)
    filter_response_fourier = np.fft.fft(filter_response)
    dim_freq = "freq_"+dim
    da_spec_filt = da_spec * filter_response_fourier[:, None]
    lag = da_spec_filt[dim_freq].attrs.get("direct_lag", 0.0)
    da_filt = xrft.ifft(da_spec_filt, 
                        dim=dim_freq,
                        shift=False, 
                        lag=lag)

    if not return_complex:
        # assert that magnitude of complex part is very small
        assert_message = f"potentially significant complex component detected. Consider zero-padding input to low-pass filter"
        assert_condition = lambda x: np.isclose(abs(x.imag).max(), 0, atol=1e-10)
        da_filt = add_delayed_assert(
            x=da_filt,
            condition_function=assert_condition,
            error_msg=assert_message,
        )
        da_filt = da_filt.real

    if fill_nans:
        da_filt = da_filt.where(condition_fill.data, np.nan)

    # ensure dimensions after fft match those of input (sometimes rounding errors can occur)
    # NOTE with shift set to False and no manual fftshift the coordinates appear incorrectly not to match (but is good, I hope)
    dimensions = [*da.sizes]
    for dimension in dimensions:
        da_filt[dimension] = da[dimension]

    if da.name == None:
        new_name = None
    else:
        new_name = f"{da.name} low-pass filtered"
    da_filt = da_filt.rename(new_name)

    return da_filt

def low_pass_filter_2D(
    da: xr.DataArray,
    cutoff_frequency: float,
    fs_x: float,
    fs_y: float,
    window: str = "hann",
    fill_nans: bool = False,
    return_complex: bool = False,
) -> xr.DataArray:
    """
    Low pass filtering an xarray dataArray in the Fourier domain using xrft, scipy.signal.windows and np.fft

    Assumes both x and y have same dimensions

    Input:
    ------
    da: xr.DataArray,
        Data to be filtered
    cutoff_frequency: float,
        threshold frequnecy, greater frequencies are filtered out
    fs_x: float,
        sampling along first DataArray dimension in tuple
    fs_y: float,
        sampling along first DataArray dimension in tuple
    window: str,
        window string from scipy.signal.get_window
    fill_nans: bool,
        Whether to replace non-finite values (e.g. nans and inifinities) with 0
    return_complex: bool,
        Whether the imaginary part should be returned or not

    Returns:
    --------
    da_filt: xr.DataArray,
        The real part of low-pass filtered data
    """

    if fill_nans:
        condition_fill = np.isfinite(da)
        da_filled = xr.where(condition_fill, da, 0)
    else:
        da_filled = da

    # data is rechunked because fourier transform cannot be performed over chunked dimension
    # shift set to false to prevent clashing fftshifts between np.fft and xrft.fft
    if is_chunked_checker(da_filled):
        da_spec = xrft.fft(
            da_filled.chunk({**da_filled.sizes}),
            chunks_to_segmentsbool=False,
            shift=False,
        )
    else:
        da_spec = xrft.fft(da_filled, shift=False)

    # design time-domain filter
    filter_response = design_low_pass_filter_2D(
        da_spec.shape, cutoff_frequency, fs_x=fs_x, fs_y=fs_y, window=window
    )

    # convert window to fourier domain and multiply with spectrum, then convert back (i.e. same as convolving filter with input image)
    filter_response_fourier = np.fft.fft2(filter_response)
    da_spec_filt = da_spec * filter_response_fourier
    lag = [da_spec_filt[d].attrs.get("direct_lag", 0.0) for d in da_spec_filt.dims]
    da_filt = xrft.ifft(da_spec_filt, shift=False, lag=lag)

    if not return_complex:
        # assert that magnitude of complex part is very small
        assert_message = f"potentially significant complex component detected. Consider zero-padding input to low-pass filter"
        assert_condition = lambda x: np.isclose(abs(x.imag).max(), 0, atol=1e-10)
        da_filt = add_delayed_assert(
            x=da_filt,
            condition_function=assert_condition,
            error_msg=assert_message,
        )
        da_filt = da_filt.real

    if fill_nans:
        da_filt = da_filt.where(condition_fill.data, np.nan)

    # ensure dimensions after fft match those of input (sometimes rounding errors can occur)
    # NOTE with shift set to False and no manual fftshift the coordinates appear incorrectly not to match (but is good, I hope)
    dimensions = [*da.sizes]
    for dimension in dimensions:
        da_filt[dimension] = da[dimension]

    if da.name == None:
        new_name = None
    else:
        new_name = f"{da.name} low-pass filtered"
    da_filt = da_filt.rename(new_name)

    return da_filt


def low_pass_filter_2D_dataset(
    ds: xr.Dataset,
    cutoff_frequency: float,
    fs_x: float,
    fs_y: float,
    window: str = "hann",
    fill_nans: bool = False,
    return_complex: bool = False,
) -> xr.Dataset:
    """
    Wrapper of low_pass_filter_2D for each dataArray in dataset

    Assumes both x and y have same dimensions

    Input:
    ------
    ds: xr.Dataset,
        Data to be filtered
    cutoff_frequency: float,
        threshold frequnecy, greater frequencies are filtered out
    fs_x: float,
        sampling along first DataArray dimension in tuple
    fs_y: float,
        sampling along first DataArray dimension in tuple
    window: str,
        window string from scipy.signal.get_window
    fill_nans: bool,
        Whether to replace non-finite values (e.g. nans and inifinities) with 0
    return_complex: bool,
        Whether the imaginary part should be returned or not

    Returns:
    --------
    ds_filt: xr.Dataset,
        The real part of low-pass filtered data
    """

    ds_filt = xr.Dataset(
        {
            var: low_pass_filter_2D(
                ds[var],
                cutoff_frequency=cutoff_frequency,
                fs_x=fs_x,
                fs_y=fs_y,
                window=window,
                fill_nans=fill_nans,
                return_complex=return_complex,
            )
            for var in ds
        }
    )

    return ds_filt
