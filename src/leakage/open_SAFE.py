import xsar
import io
import os
import sys
import warnings
import xarray as xr
import numpy as np
from . import constants


def create_storage_name(filename, grid_spacing):
    """
    Function to create a storage name for given Sentinel-1 .SAFE file(s)
    """
    if isinstance(filename, str):
        ref_file = filename
        unique_ids = find_unique_id(filename)

    elif isinstance(filename, list):
        ref_file = filename[0]
        unique_ids = [find_unique_id(file) for file in filename]
        unique_ids.sort()
        unique_ids = "_".join(unique_ids)

    id_end = ref_file.rfind("/") + 1
    storage_dir = ref_file[:id_end]
    return storage_dir + unique_ids + f"_res{grid_spacing}.nc"


def find_unique_id(filename, unique_id_length=4):
    """
    Function to find the last four digits (unique ID) of Sentinel-1 naming convention
    """
    id_start = filename.rfind("_") + 1
    return filename[id_start : id_start + unique_id_length]


def compute_S1_ground_range(ds: xr.Dataset) -> xr.DataArray:
    """
    Computes approximate ground range corresponding to elevation angle and slant range travel time
    """
    slant_range_distance = (ds.slant_range_time * constants.c) / 2
    ground_range_approx = np.sin(np.deg2rad(ds.elevation)) * slant_range_distance
    return ground_range_approx


def open_new_file(
    filename, grid_spacing, dim_concat: str = "line", var_sortby: str = "time"
):
    """
    Loads Sentinel-1 .SAFE file(s) and, if multiple, concatenate along azimuth (sorted by time)
    """

    # Surpress some warnings
    output_buffer = io.StringIO()
    stdout_save = sys.stdout
    sys.stdout = output_buffer

    # Use the catch_warnings context manager to temporarily catch warnings
    with warnings.catch_warnings(record=True) as caught_warnings:
        successful_files = []
        if isinstance(filename, str):
            S1_file = xsar.open_dataset(filename, resolution=grid_spacing)
            successful_files.append(filename)
        elif isinstance(filename, list):
            _file_contents = []
            for i, file in enumerate(filename):
                try:
                    content_partial = xsar.open_dataset(file, resolution=grid_spacing)
                    _file_contents.append(content_partial)
                    successful_files.append(file)
                except Exception as e:
                    # temporarily stop surpressing warnings
                    sys.stdout = stdout_save
                    print(
                        f"File {i} did not load properly. File in question: {file} \n Error: {e}"
                    )
                    sys.stdout = output_buffer

            # concatenate data ensuring that it is sorted by time.
            data_concatened = xr.concat(_file_contents, dim_concat)
            data_sorted = data_concatened.sortby(var_sortby)

            # redifine coordinate labels of concatened dimension
            line_step = data_sorted.line.diff(dim=dim_concat).max()
            S1_file = data_sorted.assign_coords(
                line=line_step.data * np.arange(data_sorted[dim_concat].size)
            )

    # Reset system output to saved version
    sys.stdout = stdout_save
    return S1_file, successful_files


def open_S1(
    filename: str,
    grid_spacing: int,
    attrs_to_str: list[str] = [
        "start_date",
        "stop_date",
        "footprint",
        "multidataset",
    ],
    vars_to_keep: list[str] = [
        "ground_heading",
        "time",
        "incidence",
        "latitude",
        "longitude",
        "sigma0",
        "ground_range_approx",
    ],
    coords_to_drop: str = "spatial_ref",
    pol: str = "VV",
):
    """
    Open data from a file or a list of files. First check if file can be reloaded
    """

    storage_name = create_storage_name(filename, grid_spacing)

    # reload file if possible
    if os.path.isfile(storage_name):
        print(f"Associated file found and reloaded: {storage_name}")
        S1_file = xr.open_dataset(storage_name)

    # else open .SAFE files, process and save as .nc
    else:
        S1_file, successful_files = open_new_file(filename, grid_spacing)
        storage_name = create_storage_name(successful_files, grid_spacing)

        # convert to attributes to ones storeable as .nc
        for attr in attrs_to_str:
            S1_file.attrs[attr] = str(S1_file.attrs[attr])

        # drop coordinate with a lot of attributes
        S1_file = S1_file.drop_vars(coords_to_drop)

        # keep only VV polarisation
        S1_file = S1_file.sel(pol=pol)

        # compute approximate ground range
        S1_file["ground_range_approx"] = compute_S1_ground_range(ds=S1_file)

        # store as .nc file ...
        S1_file[vars_to_keep].to_netcdf(storage_name)

        # ... and reload
        S1_file = xr.open_dataset(storage_name)

        print(f"No pre-saved file found, instead saved loaded file as: {storage_name}")

    return S1_file
