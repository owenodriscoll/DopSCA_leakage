import xarray as xr


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

    Return
    ------
    integrated_beam: xr.DataArray | xr.Dataset,
        input features integrated along the beam's azimuth
    """

    integrated_beam = x.mean(dim=azimuth_dim, skipna=skipna)
    return integrated_beam


def is_chunked_checker(da: xr.DataArray) -> bool:
    """checks whether inoput datarray is chunked"""
    return da.chunks is not None and any(da.chunks)