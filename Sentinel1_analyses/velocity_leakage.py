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
from s1sea.s1_preprocess import grd_to_nrcs
from drama.performance.sar.antenna_patterns import sinc_bp, phased_array

# FIXME fix this
# importing from one directory  up
sys.path.insert(0, "../" )
from misc import round_to_hour, angular_difference, calculate_distance  

from dataclasses import dataclass
from typing import Callable, Union, List, Dict, Any


# --------- TODO LIST ------------
# FIXME ambiguity with beam pattern one-or two-way tapering
# FIXME unfortunate combinates of vx_sat, PRF and resolution_spatial can lead to artifacts

# TODO include for and aft viewing geometry in addition to mid, to obtain mutliple velocity vectors
# TODO ugly import from directory up
# TODO add dask chunking
# TODO querry era5 per pixel rather than per dataset
# TODO add docstrings
# TODO create a second xarray dataset object after removing objects outside beam pattern

# NOTE weight is linearly scaled with relative nrcs (e.g. a nrcs of twice the average will yield relative weight of 2.0)
# NOTE nrcs weight is calculated per azimuth line in slow time, not per slow time (i.e. not the average over grg and az per slow time)


# constants
c = 3E8

import warnings
import io
import sys

@dataclass
class S1DopplerLeakage:
    """

    """

    filename: Union[str, list]
    f0: float = 5.3E9
    z0: float = 700E3
    length_antenna: float = 3.2
    height_antenna: float = 0.3
    beam_pattern: str = "sinc" # ["sinc", "phased_array"]
    beam_weight_in_scene: float = 0.9995 # fraction of the beam weigths within scene
    incidence_angle_scat: float = 40
    incidence_angle_scat_boresight: float = 45
    vx_sat: int = 6800 
    PRF: int = 4
    az_mask_cutoff: int = 80_000 # m # two sided
    resolution_spatial: int = 340 # m # 
    scene_size: int = 25_000
    era5_nc: Union[bool, str] = False # filename of era5 to load
    path_era5: str = None
    _denoise: bool = True


    def __post_init__(self):

        self.Lambda = c / self.f0 # m
        self.stride = self.vx_sat / self.PRF
        self.az_mask_pixels_cutoff = int(self.az_mask_cutoff/2//self.resolution_spatial) 
        self.grg_N = int(self.scene_size // self.resolution_spatial)           # number of fast-time samples to average to scene size
        self.slow_time_N = int(self.scene_size // self.stride)                 # number of slow-time samples to average to scene size
    
    @staticmethod
    def windfield_over_slow_time(ds, dims):
        """

        input
        -----
        ds: xr.Dataset, dataset containing the fields 'windfield', 'inc_scatt_eqv' and the attribute 'wdir_wrt_sensor'
        dims: list, list of strings containing the dimensions for which the nrcs is calculated per the equivalent scatterometer incidence

        output
        ------
        ds: xr.Dataset, dataset containing a new variable 'nrcs_scat_eqv'
        """

        nrcs_scatterometer_equivalent = cmod5n_forward(ds['windfield'].values, ds.attrs['wdir_wrt_sensor'], ds['inc_scatt_eqv'].values)
        ds['nrcs_scat_eqv'] = (dims, nrcs_scatterometer_equivalent, {'units': 'm/s'}) 
        
        return ds
    

    @staticmethod
    def remove_outside_beampattern(ds, dim_filter: str, dim_new: str, dim_new_res: Union[int, float] = 1):
        """
        Function to remove data along specific coordinates which corresponding to a mask array in the ds

        input
        -----
        ds: xr.Dataset, dataset that contains a 'mask' field and has an attribute 'resolution_spatial'
        dim_filter: str, name of dimension to filter 
        dim_new: str, new name of filtered dimension
        dim_new_res: Union[int, float], resolution along new dimension

        output
        -------
        ds: xr.Dataset, dataset with old dimension replaced
        """

        ds = ds.where(ds.mask, drop = True)
        ds = ds.assign_coords({dim_new: (dim_filter, dim_new_res*np.arange(ds.dims[dim_filter]))})
        ds = ds.swap_dims({dim_filter:dim_new})
        ds = ds.reset_coords(names=dim_filter)

        return ds

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
        """

        """

        if type(self.era5_nc) == str:
            self.era5 = xr.open_dataset(self.era5_nc)

        else:
            date = self.S1_file.azimuth_time.min().values.astype('datetime64[m]').astype(object)
            date_rounded = round_to_hour(date)

            yy, mm, dd, hh = date_rounded.year, date_rounded.month, date_rounded.day, date_rounded.hour
            latmin = latmax = self.S1_file.latitude.mean().data*1
            lonmin = lonmax = self.S1_file.longitude.mean().data*1

            era5_filename = getera5(latmin, latmax, lonmin, lonmax, yy, mm, dd, hh, path=self.path_era5, retrieve=True)

            self.era5 = xr.open_dataset(era5_filename)

        return


    def wdir_from_era5(self):
        """

        """

        # extract wind vectors
        u10, v10 = np.ravel(self.era5.u10.values*1)[0], np.ravel(self.era5.v10.values*1)[0]
        wdir_era5 = np.rad2deg(np.arctan2(u10, v10))

        # Compute orientation of observation
        lats, lons = self.S1_file.latitude.values, self.S1_file.longitude.values
        geodesic = pyproj.Geod(ellps='WGS84')
        ground_dir, _, _ = geodesic.inv(lons[0, 0], lats[0, 0], lons[-1,0], lats[-1,0])

        # compute directional difference between satelite and era5 wind direction
        self.wdir_wrt_sensor = angular_difference(ground_dir, wdir_era5)

        return


    def create_dataset(self):
        """

        """  

        # calculate new ground range and azimuth range belonging to observation with scatterometer viewing geometry
        grg_offset = np.tan(np.deg2rad(self.incidence_angle_scat)) * self.z0
        grg = np.arange(self.S1_file.NRCS_VV.data.shape[1]) * self.resolution_spatial + grg_offset
        az = (np.arange(self.S1_file.NRCS_VV.data.shape[0]) - self.S1_file.NRCS_VV.data.shape[0]//2) * self.resolution_spatial
        x_sat = np.arange(az.min(), az.max(), self.stride)

        # create new dataset 
        data = xr.Dataset(
            data_vars=dict(
                nrcs = (["az", "grg"], self.S1_file.NRCS_VV.data, {'units': 'm2/m2'}),
                inc = (["az", "grg"], self.S1_file.inc.data, {'units': 'Degrees'}),
            ),
            coords=dict(
                az = (["az"], az, {'units': 'm'}),
                grg = (["grg"], grg, {'units': 'm'}),
            ),
            attrs=dict(
                wdir_wrt_sensor = self.wdir_wrt_sensor,
                resolution_spatial= self.resolution_spatial),
        )

        # add windfield
        windfield = cmod5n_inverse(data["nrcs"].values, data.wdir_wrt_sensor, data["inc"].values)

        # for some reason cmod still returns a value even when input is nan, here these are removed
        self.windfield = xr.where(data.nrcs.isnull(), np.nan, windfield)
        data["windfield"] = self.windfield.assign_attrs(units= 'm/s', description = 'CMOD5n Windfield for Sentinel-1 backscatter')

        # add another dimension for later use
        x_sat = np.arange(data.az.min(), data.az.max(), self.stride)
        data['x_sat'] = (["slow_time"], x_sat)

        self.data = data

        return
    
    def create_beam_mask(self):
        """
        
        """

        # compute beam mask around 
        beam_center = abs(self.data['x_sat'] - self.data['az']).argmin(dim=['az'])['az'].values 

        masks = []
        for i in beam_center:
            mask = np.nan*np.zeros_like(self.data['az'])  # create a mask
            lower_limit = np.where(i-self.az_mask_pixels_cutoff < 0, 0, i-self.az_mask_pixels_cutoff)
            mask[lower_limit:i+self.az_mask_pixels_cutoff+1] = 1
            masks.append(mask)

        ds_mask = xr.Dataset(
            data_vars=dict(
                mask_az_st = (["az","slow_time"], np.array(masks).T), 
                mask_grg = (["grg"], np.ones_like(self.data.grg))),
            coords=dict(
                az=(["az"], self.data['az'].values),
                grg=(["grg"], self.data['grg'].values),
                time_slow=(["slow_time"], self.data['slow_time'].values),),
        )

        self.data['beam_mask'] = ds_mask.mask_az_st * ds_mask.mask_grg
        self.data['mask'] = ~self.data.beam_mask.isnull()
        self.data = self.data.groupby('slow_time').map(self.remove_outside_beampattern, args = ["az", "az_idx", self.data.attrs['resolution_spatial']])


        return

    def compute_scatt_eqv_backscatter(self):
        """

        """

        # calculate surface distance between sensor and point on surface as well as equivalent incidence angle for sensor
        self.data['distance_ground'] = calculate_distance(x = self.data["az"], y = self.data["grg"], x0 = self.data["x_sat"]) 
        self.data['inc_scatt_eqv'] = np.rad2deg(np.arctan(self.data['distance_ground']/self.z0))
        self.data = self.data.groupby('slow_time').map(self.windfield_over_slow_time, dims = ['az_idx', 'grg']).transpose('az_idx', 'grg', 'slow_time')

        return
    
    def compute_beam_pattern(self, N: int = 10, w: float =0.5):
        """
        calculate beam patterns

        input
        -----
        N = int, Number of phased array elements (if phased_array, default = 10)
        w = float, weighting of phased array elements (if phased_array, default = 0.5)

        """

        self.data['distance_slant_range'] = np.sqrt(self.data['distance_ground']**2 + self.z0**2)
        self.data['az_angle_wrt_boresight'] = np.arcsin((- self.data['x_sat'] + self.data['az'] )/self.data['distance_slant_range']) # incidence from boresight
        self.data['grg_angle_wrt_boresight'] = np.deg2rad(self.data['inc_scatt_eqv'] - self.incidence_angle_scat_boresight)
        self.data = self.data.transpose('az_idx', 'grg', 'slow_time')

        if self.beam_pattern == "sinc":
            beam_rg = sinc_bp(np.sin(self.data['az_angle_wrt_boresight']), L=self.length_antenna, f0=self.f0)**2
            beam_az = sinc_bp(np.sin(self.data['grg_angle_wrt_boresight']), L=self.height_antenna, f0=self.f0)**2

        # FIXME when using phased array, is tapering only applied on receive?
        elif self.beam_pattern == "phased_array":
            beam_rg = np.squeeze(phased_array(np.sin(self.data['az_angle_wrt_boresight']), L=self.length_antenna, f0=self.f0, N=N, w=w)**2)
            beam_az = np.squeeze(phased_array(np.sin(self.data['grg_angle_wrt_boresight']), L=self.height_antenna, f0=self.f0, N=N, w=w)**2)

        self.data['beam_grg'] = (['az_idx', 'grg', 'slow_time'], beam_rg)
        self.data['beam_az'] = (['az_idx', 'grg', 'slow_time'], beam_az)
        self.data['beam_grg_az'] = self.data['beam_grg'] * self.data['beam_az']

        return
    
        
    def compute_Doppler_leakage(self, beam_weight_in_scene: float  = 0.9995):
        """

        """
        
        # cheesy
        data = self.data

        # compute geometrical doppler, beam pattern and nrcs weigths
        data['dop_geom'] = (2 * self.vx_sat * np.sin(data['az_angle_wrt_boresight']) / self.Lambda) # eq. 4.34 from Digital Procesing of Synthetic Aperture Radar Data by Ian G. Cummin
        data['beam'] = data['beam_grg_az'] * data['beam_mask'] 
        data['nrcs_weight'] = (data.nrcs_scat_eqv / data.nrcs_scat_eqv.mean(dim=['az_idx',])) # NOTE weight calculated per azimuth line

        # compute weighted received Doppler and resulting apparent LOS velocity
        data['dop_beam_weighted'] = data['dop_geom'] * data['beam']* data['nrcs_weight']
        data['V_leakage'] = self.Lambda * data['dop_beam_weighted'] / (2 * np.sin(np.deg2rad(data['inc_scatt_eqv']))) # using the equivalent scatterometer incidence angle

        # calculate scatt equivalent nrcs
        data['nrcs_scat'] = ((data['nrcs'] * data['beam']).sum(dim='az_idx') / data['beam'].sum(dim='az_idx'))

        # sum over azimuth to receive range-slow_time results
        weight_rg = (data['beam'] * data['nrcs_weight']).sum(dim='az_idx', skipna=False)
        receive_rg = data[['dop_beam_weighted', 'V_leakage']].sum(dim='az_idx', skipna=False)
        data[['doppler_pulse_rg', 'V_leakage_pulse_rg']] = receive_rg / weight_rg

        # set data to nan for which too much of the beam weights fall outside the scene (ramping up/down)
        data['beam_cutoff'] = data['beam'].sum(dim = ['az_idx', 'grg'])/ data['beam'].sum(dim = ['az_idx', 'grg']).max() < beam_weight_in_scene
        data[['doppler_pulse_rg', 'V_leakage_pulse_rg', 'nrcs_scat']][dict(slow_time=data['beam_cutoff'])] = np.nan

        # add attributes and coarsen data to resolution of subscenes
        self.data['V_leakage_pulse_rg'] = data['V_leakage_pulse_rg'].assign_attrs(units= 'm/s', description = 'Line of sight velocity ')
        self.subscenes = data[['doppler_pulse_rg', 'V_leakage_pulse_rg']].coarsen(grg=self.grg_N, slow_time=self.slow_time_N, boundary='trim').mean(skipna=False) 

        return

    def apply(self, **kwargs):
        self.open_data()
        self.querry_era5()
        self.wdir_from_era5()
        self.create_dataset()
        self.create_beam_mask()
        self.compute_scatt_eqv_backscatter()
        self.compute_beam_pattern(**kwargs)
        self.compute_Doppler_leakage()

        return
