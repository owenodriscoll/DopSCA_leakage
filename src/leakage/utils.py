import xarray as xr
from typing import Callable


def mean_along_azimuth(
    x: xr.DataArray | xr.Dataset, azimuth_dim: str = "az_idx", skipna: bool = False
) -> xr.DataArray | xr.Dataset:
    """
    integrates input array/dataset along azimuthal beam pattern

    Input
    -----
    x: xr.DataArray | xr.Dataset,
        array or dataset of arrays which to integrate over beam
    azimuth_dim: str,
        Name of azimuthal dimension over which to integrate beam, default is 'az_idx'
    skipna: bool,
        whether to skip (ignore) nans, default is False

    Returns
    ------_
    integrated_beam: xr.DataArray | xr.Dataset,
        input features integrated along the beam's azimuth
    """

    integrated_beam = x.mean(dim=azimuth_dim, skipna=skipna)
    return integrated_beam


def is_chunked_checker(da: xr.DataArray) -> bool:
    """checks whether inoput datarray is chunked"""
    return da.chunks is not None and any(da.chunks)


def add_delayed_assert(
    x: xr.DataArray | xr.Dataset,
    condition_function: Callable,
    error_msg: str = "Undesired condition met!",
    raise_error: bool = True,
) -> xr.DataArray | xr.Dataset:
    """
    Raise an error if any element in input do not meet conditions.
    Assertion is performed in a lazy-computation-friendly manner

    Parameters
    ----------
    x : xr.DataArray|xr.Dataset
        Input array on which condition will be checked
    condition_function : Callable
        Function to be applied on x
    error_msg : str
        Optional error message to return
    raise_error : bool
        Whether to raise an error or to print a warning

    Returns
    -------
    xr.DataArray|xr.Dataset
        Original input data with delayed assert added to task grah

    Raises
    ------
    ValueError
        If the input x does not meet condition

    Examples
    --------
    >>> x = xr.DataArray(
            np.ones((4, 3)) +
            1j * np.ones((4, 3)) / 1e6
            )
    >>> condition_function = lambda x: abs(x.imag) >= (0.01 * abs(x.real))
    >>> x = add_delayed_assert(x=x, condition_function=condition_function)
    >>> x.compute()
    ValueError
    """

    def delayed_raise(x, condition_function, error_msg, raise_error):
        y = condition_function(x)
        if not y.any():
            if raise_error:
                raise ValueError(error_msg)
            else:
                print("Warning! " + error_msg)
        return x

    kwargs = dict(
        condition_function=condition_function,
        error_msg=error_msg,
        raise_error=raise_error,
    )

    return x.map_blocks(delayed_raise, kwargs=kwargs, template=x)
