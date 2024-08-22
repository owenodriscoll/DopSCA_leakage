import xarray as xr
import numpy as np
from .utils import is_chunked_checker
import xrft


def padding_fourier(
    da: xr.DataArray, padding: int | tuple, dimension: str, pad_value=complex(0, 0)
) -> xr.DataArray:
    """
    Interpolate data by zero-padding in Fourier domain along dimension
    """
    if is_chunked_checker(da):
        da = da.chunk({**da.sizes})

    da_spec = xrft.fft(
        da, true_amplitude=False
    )  # set to false for same behaviour as np.fft
    da_spec_padded = xrft.padding.pad(
        da_spec, {"freq_" + dimension: padding}, constant_values=pad_value
    )

    # data is rechunked because Fourier transform cannot be performed over chunked dimension
    if is_chunked_checker(da_spec_padded):
        da_spec_padded = da_spec_padded.chunk({**da_spec_padded.sizes})

    da_spec_padded *= (
        da_spec_padded.sizes["freq_" + dimension] / da.sizes[dimension]
    )  # multiply times factor to compensate for more samples in Fourier domain
    lag = [da_spec_padded[d].attrs.get("direct_lag", 0.0) for d in da_spec_padded.dims]
    da_padded = xrft.ifft(
        da_spec_padded, true_amplitude=False, lag=lag
    )  # set to false for same behaviour as np.fft

    return da_padded


def da_integer_oversample_like(
    da: xr.DataArray, dim_to_resample: int = 0, new_samples_per_original_sample: int = 2
) -> xr.DataArray:
    """
    Creates a new datarray filled with ones whose shape corresponds to the number of independent samples, rather than (oversampled) real samples
    """

    dim_interp = "dim_" + str(dim_to_resample)
    a = da.shape
    b = np.ones(a)
    c = xr.DataArray(b)
    d = c.coarsen({dim_interp: new_samples_per_original_sample}, boundary="trim").mean()
    return d


def compute_padding_1D(length_desired: int, length_current: int) -> tuple[int, int]:
    """
    Computes how much to pad on both sides of 1D dimension to obtain desired length
    """
    assert length_desired > length_current
    pad_total = length_desired - length_current
    pad = int(np.ceil(pad_total / 2))
    return pad
