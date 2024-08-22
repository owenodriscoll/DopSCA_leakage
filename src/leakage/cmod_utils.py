from stereoid.oceans.GMF.cmod5n import cmod5n_inverse, cmod5n_forward
import xarray

def wrapper_wind2nrcs(ds, 
                      dimensions=["az_idx", "grg", "slow_time"], 
                      str_windfield="windfield",
                      str_wdir_wrt_sensor="wdir_wrt_sensor",
                      str_incidence_angle="inc_scatt_eqv_cube",
                      output_name="nrcs_scat_eqv"):
    """
    Wrapper of CMOD5.n to enable dask's lazy computations

    input
    -----
    ds: xr.Dataset, dataset containing the a windfield, wind direction and incidence angle field
    dimensions: list, list of strings containing the dimensions for which the nrcs is calculated per the equivalent scatterometer incidence

    output
    ------
    ds: xr.Dataset, dataset containing a new variable {output_name}
    """

    nrcs_scatterometer_equivalent = cmod5n_forward(
        ds[str_windfield].data,
        ds[str_wdir_wrt_sensor].data,
        ds[str_incidence_angle].data,
    )
    ds[output_name] = (
        dimensions,
        nrcs_scatterometer_equivalent,
        {"units": "sigma nought"},
    )
    return ds

def wrapper_nrcs2wind(ds,
                      dimensions=["az", "grg"], 
                      str_nrcs="nrcs",
                      str_wdir_wrt_sensor="wdir_wrt_sensor",
                      str_incidence_angle="inc",
                      output_name="windfield"):
    """
    Wrapper of CMOD5.n to enable dask's lazy computations

    input
    -----
    ds: xr.Dataset, dataset containing the a windfield, wind direction and incidence angle field
    dimensions: list, list of strings containing the dimensions for which the nrcs is calculated per the equivalent scatterometer incidence

    output
    ------
    ds: xr.Dataset, dataset containing a new variable {output_name}
    """

    nrcs_scatterometer_equivalent = cmod5n_inverse(
        ds[str_nrcs].data,
        ds[str_wdir_wrt_sensor].data,
        ds[str_incidence_angle].data,
    )
    ds[output_name] = (
        dimensions,
        nrcs_scatterometer_equivalent,
        {"units": "m/s"},
    )
    return ds
