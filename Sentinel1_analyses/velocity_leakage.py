import os
import sys 
import glob
import pyproj
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import xarray as xr
import xarray_sentinel
import drama.utils as drtls
import s1sea.geo_plot as gplt
from s1sea.cmod5n import cmod5n_inverse, cmod5n_forward
from s1sea.get_era5 import getera5
from s1sea.s1_preprocess import grd_to_nrcs, nrcs_to_roughness
from drama.performance.sar.antenna_patterns import sinc_bp

# FIXME fix this
# importing from one directory  up
sys.path.insert(0, "../" )
from misc import round_to_hour, angular_difference, calculate_distance  

from dataclasses import dataclass
from typing import Callable, Union, List, Dict, Any


# --------- TODO LIST
# fix ugly import from directory up
# add dask chunking




# constants
c = 3E8

import warnings
import io
import sys

@dataclass
class S1DopplerLeakage:
    filename: Union[str, list]
    f0 = 5.3E9
    z0 = 700E3
    length_antenna = 3.2
    height_antenna = 0.3
    incidence_angle_scat = 40
    incidence_angle_scat_boresight = 45
    vx_sat = 6800 
    PRF = 4
    az_mask_cutoff = 200_000 # m
    resolution_spatial = 340 # m
    scene_size = 25_000

    _denoise = True


    def __post_init__(self):

        self.Lambda = c / self.f0 # m
        self.stride = self.vx_sat / self.PRF
        self.az_mask_pixels_cutoff = int(self.az_mask_cutoff/2//self.resolution_spatial) 
        self.grg_N = int(self.scene_size // self.resolution_spatial)           # number of fast-time samples to average to scene size
        self.slow_time_N = int(self.scene_size // self.stride)                 # number of slow-time samples to average to scene size
    

    # TODO add chunking
    def open_data(self):
        """
        Open data from a file or a list of files.
        """
        # Redirect standard output to a variable
        output_buffer = io.StringIO()
        sys.stdout = output_buffer

        # Use the catch_warnings context manager to temporarily catch warnings
        with warnings.catch_warnings(record=True) as caught_warnings:
            if isinstance(self.filename, str):
                self.S1_file = grd_to_nrcs(self.filename, prod_res=self.resolution_spatial, denoise=self._denoise)
            elif isinstance(self.filename, list):
                self._file_contents = []
                for i, file in enumerate(self.filename):
                    try:
                        content_partial = grd_to_nrcs(file, prod_res=self.resolution_spatial, denoise=self._denoise)
                        self._file_contents.append(content_partial)
                    except Exception as e:
                        sys.stdout = sys.__stdout__
                        print(f'File {i} did not load properly. \nConsider manually adding file content to self._file_contents. File in question: {file} \n Error: {e}')
                        sys.stdout = output_buffer

                self.S1_file = xr.merge(self._file_contents)

        # Reset standard output to its original value
        sys.stdout = sys.__stdout__

        # # Optionally print the caught warnings
        # for warning in caught_warnings:
            # print(warning.message)

        return


    def querry_era5(self):
    
    
        return


    def _wdir_from_era5(self):


        return


    def create_dataset(self):


        return
    

    def to_scatt_eqv_backscatter(self):


        return
    
    def create_beampattern(self):


        return
    
    def create_beam_mask(self):


        return
        
    def compute_Doppler_leakage(self):


        return

