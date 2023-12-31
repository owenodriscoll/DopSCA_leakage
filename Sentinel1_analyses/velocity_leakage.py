import os
import io
import sys 
import glob
import dask
import copy
import pyproj
import warnings
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import xarray as xr
import xarray_sentinel
import dask.array as da
import drama.utils as drtls
# import s1sea.geo_plot as gplt
from s1sea.cmod5n import cmod5n_inverse, cmod5n_forward
# from s1sea.get_era5 import getera5
from s1sea.s1_preprocess import grd_to_nrcs
from drama.performance.sar.antenna_patterns import sinc_bp, phased_array

# FIXME fix this
# importing from one directory  up
sys.path.insert(0, "../" )
from misc import round_to_hour, angular_difference, calculate_distance, era5_wind_single_time_loc

from dataclasses import dataclass
from typing import Callable, Union, List, Dict, Any


# --------- TODO LIST ------------
# FIXME Find correct beam pattern (tapering/pointing?) for receive and transmit, as well as correct N sensor elements
# FIXME unfortunate combinates of vx_sat, PRF and resolution_spatial can lead to artifacts, maybe interpolation?

# TODO add option to include geophysical Doppler
# TODO add land mask filter
# TODO ugly import from directory up
# TODO add dask chunking
# TODO querry era5 per pixel rather than per dataset
    # NOTE currently the mean ERA5 value is chosen over the entire dataset --> USE CONTINUOUS SCENES ONLY!
        # TODO assess lekage velovity estimation performance assuming 0-360 reference wind direction 
    # TODO if high-res era5 wdir is used, wdir_wrt_sensor should be calculated across slow time
# TODO add docstrings
    # TODO add kwargs to input for phased beam pattern in create_beampattern
# TODO currently inversion interpolates scatterometer grid size to S1 grid, change to first apply inversion and then interpolate 

# NOTE loading .SAFE often does not work. If not, try using dual pol data only (still no guarantee)
# NOTE Range cell migration not included
# NOTE weight is linearly scaled with relative nrcs (e.g. a nrcs of twice the average will yield relative weight of 2.0)
# NOTE nrcs weight is calculated per azimuth line in slow time, not per slow time (i.e. not the average over grg and az per slow time)


# constants
c = 3E8




@dataclass
class S1DopplerLeakage:
    """
    Input:
    ------
    filename: str, list; filename or list of filenames of Sentinel-1 .SAFE
    f0: float; radiowave frequency in hz
    z0: float; average satellite orbit height in meters
    length_antenna: float;
    height_antenna: float;
    beam_pattern: str; choose from ["sinc", "phased_array"], determines the beam width and side-lobe sensitivity
    era5_directory: str; directory containing era5 file. Will look for file containing "{yyyy}{mm}.nc" or "era5{yy}{mm}{dd}.nc" to load
    era5_file: str; specific file to load

    Output:
    -------

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
    era5_directory: str = "" # directory name containing era5 files to load
    era5_file: Union[bool, str] = False # file name of era5 to load
    random_state: int = 42
    _denoise: bool = True


    def __post_init__(self):

        self.Lambda = c / self.f0 # m
        self.stride = self.vx_sat / self.PRF
        self.az_mask_pixels_cutoff = int(self.az_mask_cutoff/2//self.resolution_spatial) 
        self.grg_N = int(self.scene_size // self.resolution_spatial)           # number of fast-time samples to average to scene size
        self.slow_time_N = int(self.scene_size // self.stride)                 # number of slow-time samples to average to scene size
        attributes_to_store = copy.deepcopy(self.__dict__)

        # if input values are None or Booleans, convert them to string type
        attributes_to_store_updated = {key: value if value is not None and type(value) is not bool else str(value) for key, value in attributes_to_store.items()}
        self.attributes_to_store = attributes_to_store_updated

        # warn if aliasing might occur due to combination of spatial and sampling resolutions 
        if self.stride % self.resolution_spatial != 0:
            warnings.warn("Combination of vx_sat, PRF and resolution_spatial may lead to aliasing: (vx_sat / PRF) % resolution_spatial != 0")

    @staticmethod
    def convert_to_0_360(longitude):
        return (longitude + 360) % 360
    
    @staticmethod
    def speckle_noise(noise_shape: tuple, random_state: int = 42):
        """
        Generates multiplicative speckle noise with mean value of 1
        """
        np.random.seed(random_state)
        noise_real = np.random.randn(*noise_shape)
        noise_imag = np.random.randn(*noise_shape)
        noise = np.array([complex(a,b) for a, b in zip(noise_real.ravel(), noise_imag.ravel())])
        noise = (abs(noise)**2)/2

        return noise.reshape(noise_shape)


    # TODO add chunking
    def open_data(self):
        """
        Open data from a file or a list of files. First check if file can be reloaded
        """

        def create_storage_name(filename, resolution_spatial):
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
                unique_ids = '_'.join(unique_ids)
                
            id_end = ref_file.rfind('/') + 1
            storage_dir = ref_file[:id_end]
            return storage_dir + unique_ids + f'_res{resolution_spatial}.nc' 


        def find_unique_id(filename, unique_id_length = 4):
            """
            Function to find the last four digits (unique ID) of Sentinel-1 naming convention
            """
            id_start = filename.rfind('_') + 1 
            return filename[id_start:id_start+unique_id_length]


        def open_new_file(filename, resolution_spatial, denoise):
            """
            Loads Sentinel-1 .SAFE file(s) and, if multiple, merges them 
            """
            # Surpress some warnings
            output_buffer = io.StringIO()
            stdout_save = sys.stdout
            sys.stdout = output_buffer

            # Use the catch_warnings context manager to temporarily catch warnings
            with warnings.catch_warnings(record=True) as caught_warnings:
                self._successful_files = []
                if isinstance(filename, str):
                    S1_file = grd_to_nrcs(filename, prod_res=resolution_spatial, denoise=denoise)
                    self._successful_files.append(file)
                elif isinstance(filename, list):
                    _file_contents = []
                    for i, file in enumerate(filename):
                        try:
                            content_partial = grd_to_nrcs(file, prod_res=resolution_spatial, denoise=denoise)
                            _file_contents.append(content_partial)
                            self._successful_files.append(file)
                        except Exception as e:
                            # temporarily stop surpressing warnings
                            sys.stdout = stdout_save
                            print(f'File {i} did not load properly. \nConsider manually adding file content to _file_contents. File in question: {file} \n Error: {e}')
                            sys.stdout = output_buffer

                    S1_file = xr.merge(_file_contents)

            # Reset system output to saved version
            sys.stdout = stdout_save
            return S1_file


        storage_name = create_storage_name(self.filename, self.resolution_spatial)

        # reload file if possible
        if os.path.isfile(storage_name):
            print(f"Associated file found and reloaded: {storage_name}")
            self.S1_file = xr.open_dataset(storage_name)

        # else open .SAFE files, process and save as .nc
        else:
            self.S1_file = open_new_file(self.filename, self.resolution_spatial, self._denoise)
            storage_name = create_storage_name(self._successful_files, self.resolution_spatial)
            self.S1_file.to_netcdf(storage_name)
            print(f"No pre-saved file found, instead saved loaded file as: {storage_name}")
        return


    def querry_era5(self):
        """

        """

        date = self.S1_file.azimuth_time.min().values.astype('datetime64[m]').astype(object)
        date_rounded = round_to_hour(date)
        yy, mm, dd, hh = date_rounded.year, date_rounded.month, date_rounded.day, date_rounded.hour
        time = f"{date_rounded.hour:02}00"

        # NOTE Currently the mean latitude and longitude are chosen
        latmin = latmax = self.S1_file.latitude.mean().data*1
        lonmin = lonmax = self.convert_to_0_360(self.S1_file.longitude).mean().data*1 # NOTE correction for fact that ERA5 goes between 0 - 360

        if type(self.era5_file) == str:
            era5 = xr.open_dataset(self.era5_file)
        else:
            sub_str = str(yy) + str(mm)
            try:
                # try to find if monthly data file exists which to load 
                era5_filename = [s for s in glob.glob(f"{self.era5_directory}*") if sub_str + '.nc' in s][0]
            except:
                #  if not, try to find single estimate hour ERA5 wind file
                # era5_filename = f"era5{yy}{mm:02d}{dd:02d}.nc"
                era5_filename = f"era5_{yy}{mm:02d}{dd:02d}h{time}_lat{latmax:.2f}_lon{lonmax:.2f}.nc"
                era5_filename = era5_filename.replace('.', '_',  2)
                if not self.era5_directory is None:
                    era5_filename = os.path.join(self.era5_directory, era5_filename)

                # if neither monthly file nor hourly single estimate file exist, download new single estimate hour
                if not os.path.isfile(era5_filename):
                    era5_wind_single_time_loc(year=yy,
                          month=mm,
                          day=dd,
                          time=time,
                          lat=latmin,
                          lon=lonmin,
                          filename=era5_filename,
                          )
                    # era5_filename = getera5(latmin, latmax, lonmin, lonmax, yy, mm, dd, hh, path=self.era5_directory, retrieve=True)
            
            print(f"Loading nearest ERA5 point w.r.t. observation from ERA5 file: {era5_filename}")
            era5 = xr.open_dataset(era5_filename)

        # TODO add check that nearest is not too far of spatially or temporally
        era5 = era5.sel(
                longitude= lonmin,
                latitude = latmin,
                time = np.datetime64(date_rounded, 'ns'),
                tolerance=0.5,
                method = 'nearest')
        self.era5 = era5
        return


    def wdir_from_era5(self):
        """

        """

        # extract wind vectors
        u10, v10 = np.ravel(self.era5.u10.values*1)[0], np.ravel(self.era5.v10.values*1)[0]
        wdir_era5 = np.rad2deg(np.arctan2(u10, v10))

        # Compute orientation of observation
        # FIXME do not load all values, only load exact indixes needed 
        lats, lons = self.S1_file.latitude.values, self.S1_file.longitude.values  # FIXME
        geodesic = pyproj.Geod(ellps='WGS84')
        ground_dir, _, _ = geodesic.inv(lons[0, 0], lats[0, 0], lons[-1,0], lats[-1,0])

        # compute directional difference between satelite and era5 wind direction
        self.wdir_wrt_sensor = angular_difference(ground_dir, wdir_era5)
        return


    def create_dataset(self, var_nrcs: str = "NRCS_VV", var_inc: str = "inc"):
        """

        """  

        # calculate new ground range and azimuth range belonging to observation with scatterometer viewing geometry
        grg_offset = np.tan(np.deg2rad(self.incidence_angle_scat)) * self.z0
        grg = np.arange(self.S1_file[var_nrcs].data.shape[1]) * self.resolution_spatial + grg_offset
        az = (np.arange(self.S1_file[var_nrcs].data.shape[0]) - self.S1_file[var_nrcs].data.shape[0]//2) * self.resolution_spatial
        x_sat = np.arange(az.min(), az.max(), self.stride)

        # create new dataset 
        data = xr.Dataset(
            data_vars=dict(
                nrcs = (["az", "grg"], self.S1_file[var_nrcs].data, {'units': 'm2/m2'}),
                inc = (["az", "grg"], self.S1_file[var_inc].data, {'units': 'Degrees'}),
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
        x_sat = da.arange(data.az.min(), data.az.max(), self.stride)
        slow_time = self.stride * da.arange(x_sat.shape[0])
        x_sat = xr.DataArray(x_sat, dims='slow_time', coords={'slow_time': slow_time})
        data = data.assign(x_sat=x_sat)

        # update with previously stored data
        data.attrs.update(self.attributes_to_store)

        self.data = data
        return
    

    def create_beam_mask(self): 
        """

        """
        # find indexes along azimuth with beam beam center NOTE does not work squinted
        beam_center = abs(self.data['x_sat'] - self.data['az']).argmin(dim=['az'])['az'].values 

        # find indxes within allowed beam with over slow time
        masks = []
        lengths = []
        for i in beam_center:
            mask = np.zeros_like(self.data['az'])  # create a mask
            lower_limit = np.where(i-self.az_mask_pixels_cutoff < 0, 0, i-self.az_mask_pixels_cutoff)
            mask[lower_limit:i+self.az_mask_pixels_cutoff+1] = 1
            idx_valid = np.argwhere(mask).squeeze()
            lengths.append(idx_valid.shape[0])
            masks.append(idx_valid)

        # determine which slow-time observations occur at the edges of the scenes, to filter out
        l = np.array(lengths)
        l_max = l.max()
        idx_slow_time = np.argwhere(l==l_max).squeeze()
        self.idx_slow_time = idx_slow_time

        # prepare clipping indixes outside beam pattern
        _data = []
        dim_filter = "az"
        dim_new = "az_idx"
        dim_new_res = self.data.attrs['resolution_spatial']

        # array with azimuth indexes to select over slow time
        idx_az = np.array([masks[i] for i in idx_slow_time])
        self.idx_az = idx_az

        # prepare data by chunking and conversion to lower bit
        self.data = self.data.astype('float32').chunk('auto')
        for i, st in enumerate(idx_slow_time):

            a = self.data.isel(slow_time = st, az = idx_az[i])
            a = a.assign_coords({dim_new: (dim_filter, dim_new_res*np.arange(a.dims[dim_filter]))})
            a = a.swap_dims({dim_filter:dim_new})
            a = a.reset_coords(names=dim_filter)

            _data.append(a)

        self.data = xr.concat(_data, dim = 'slow_time')
        return 
    

    def compute_scatt_eqv_backscatter(self):
        """

        """

        self.data['distance_az'] = (self.data["az"] - self.data["x_sat"]).isel(slow_time = 0)
        self.data['distance_ground'] = calculate_distance(x = self.data['distance_az'], y = self.data["grg"]) 
        self.data['inc_scatt_eqv'] = np.rad2deg(np.arctan(self.data['distance_ground']/self.z0))
        slow_time_vector = self.data.slow_time
        self.data['inc_scatt_eqv_cube'] = self.data['inc_scatt_eqv'].expand_dims(dim={"slow_time": slow_time_vector})

        self.data = self.data.transpose('az_idx', 'grg', 'slow_time')
        self.data = self.data.astype('float32').unify_chunks()

        def windfield_over_slow_time(ds, dimensions = ['az_idx', 'grg', 'slow_time']):
            """

            input
            -----
            ds: xr.Dataset, dataset containing the fields 'windfield', 'inc_scatt_eqv' and the attribute 'wdir_wrt_sensor'
            dims: list, list of strings containing the dimensions for which the nrcs is calculated per the equivalent scatterometer incidence

            output
            ------
            ds: xr.Dataset, dataset containing a new variable 'nrcs_scat_eqv'
            """

            nrcs_scatterometer_equivalent = cmod5n_forward(ds['windfield'].data, ds.attrs['wdir_wrt_sensor'], ds['inc_scatt_eqv_cube'].data)
            ds['nrcs_scat_eqv'] = (dimensions, nrcs_scatterometer_equivalent, {'units': 'm/s'}) 
            return ds

        self.data = self.data.map_blocks(windfield_over_slow_time)
        self.data = self.data.astype('float32')
        return

    def compute_beam_pattern(self):
        """

        """
        self.data['distance_slant_range'] = np.sqrt(self.data['distance_ground']**2 + self.z0**2)
        self.data['az_angle_wrt_boresight'] = np.arcsin((self.data['distance_az'])/self.data['distance_slant_range']) # incidence from boresight
        self.data['grg_angle_wrt_boresight'] = np.deg2rad(self.data['inc_scatt_eqv'] - self.incidence_angle_scat_boresight)
        self.data = self.data.transpose('az_idx', 'grg', 'slow_time')

        # NOTE the following computations are directly computed, a delayed lazy computation may be better
        N = 10 # number of antenna elements
        w = 0.5 # complex weighting of elements
        beam_az_tx = sinc_bp(sin_angle=self.data.az_angle_wrt_boresight, L = self.length_antenna, f0 = self.f0)

        if self.beam_pattern == "sinc":
            beam_az = beam_az_tx ** 2
        elif self.beam_pattern == "phased_array":
            beam_az_rx = phased_array(sin_angle=self.data.az_angle_wrt_boresight, L = self.length_antenna, f0 = self.f0, N = N, w = w).squeeze()
            beam_az = beam_az_tx * beam_az_rx

        beam_grg_tx = sinc_bp(sin_angle=self.data.grg_angle_wrt_boresight, L = self.height_antenna, f0 = self.f0)
        beam_grg_rx = beam_grg_tx
        beam_grg = beam_grg_tx * beam_grg_rx
        beam = beam_az * beam_grg
        self.data['beam'] = ([*self.data.az_angle_wrt_boresight.dims], beam)

        self.data = self.data.astype('float32')

        return

    def compute_leakage_velocity(self):
        """

        """
        # compute geometrical doppler, beam pattern and nrcs weigths
        self.data['dop_geom'] = (2 * self.vx_sat * np.sin(self.data['az_angle_wrt_boresight']) / self.Lambda) # eq. 4.34 from Digital Procesing of Synthetic Aperture Radar Data by Ian G. Cummin 
        self.data['nrcs_weight'] = (self.data['nrcs_scat_eqv'] / self.data['nrcs_scat_eqv'].mean(dim=['az_idx',])) # NOTE weight calculated per azimuth line

        # compute weighted received Doppler and resulting apparent LOS velocity
        self.data['dop_beam_weighted'] = self.data['dop_geom'] * self.data['beam']* self.data['nrcs_weight']
        self.data['V_leakage'] = self.Lambda * self.data['dop_beam_weighted'] / (2 * np.sin(np.deg2rad(self.data['inc_scatt_eqv']))) # using the equivalent scatterometer incidence angle

        # calculate scatt equivalent nrcs
        self.data['nrcs_scat'] = ((self.data['nrcs'] * self.data['beam']).sum(dim='az_idx') / self.data['beam'].sum(dim='az_idx'))

        # sum over azimuth to receive range-slow_time results
        weight_rg = (self.data['beam'] * self.data['nrcs_weight']).sum(dim='az_idx', skipna=False)
        receive_rg = self.data[['dop_beam_weighted', 'V_leakage']].sum(dim='az_idx', skipna=False)
        self.data[['doppler_pulse_rg', 'V_leakage_pulse_rg']] = receive_rg / weight_rg
        
        # add attribute
        self.data['V_leakage_pulse_rg'] = self.data['V_leakage_pulse_rg'].assign_attrs(units= 'm/s', description = 'Line of sight velocity ')

        # low-pass filter scatterometer data to subscene resolution
        data_4subscene = ['doppler_pulse_rg', 'V_leakage_pulse_rg', 'nrcs_scat']
        data_subscene = [name + '_subscene' for name in data_4subscene]
        self.data[data_subscene] = self.data[data_4subscene].rolling(grg=self.grg_N, slow_time=self.slow_time_N, center=True).mean()
        return


    def compute_leakage_velocity_estimate(self, speckle_noise: bool = True):
        """
        Method that estimates the leakage velocity using the scatterometer backscatter field. 
        Can be done more efficeintly

        Input
        -----
        speckle_noise: bool; whether to apply multiplicative speckle noise to scatterometer estimated nrcs
        """
        # find indexes of S1 scene that were cropped (outside full beam pattern)
        idx_start = self.idx_az[0][self.az_mask_pixels_cutoff]
        idx_end = self.idx_az[-1][self.az_mask_pixels_cutoff]

        # create placeholder S1 data (pre-cropped)
        new_nrcs = np.nan * np.ones_like(self.S1_file.NRCS_VV)
        new_inc = np.nan * np.ones_like(self.S1_file.NRCS_VV)

        # add speckle noise assuming a single look
        if speckle_noise:
            noise_multiplier = self.speckle_noise(self.data.nrcs_scat.shape, random_state = self.random_state)
        elif not speckle_noise:
            noise_multiplier = 1

        single_look_nrcs = noise_multiplier * self.data.nrcs_scat

        # interpolate estimated scatterometer data back to S1 grid size
        slow_time_upsamp = np.linspace(self.data.slow_time[0], self.data.slow_time[-1], idx_end - idx_start) 
        nrcs_scat_upsamp = single_look_nrcs.T.interp(slow_time = slow_time_upsamp)
        inc_scat_upsamp = self.data.inc_scatt_eqv_cube.mean(dim='az_idx').T.interp(slow_time = slow_time_upsamp)

        # apply cropping 
        new_nrcs[idx_start: idx_end, :] = nrcs_scat_upsamp
        new_inc[idx_start: idx_end, :] = inc_scat_upsamp

        # copy existing object to avoid overwritting
        self_copy = copy.deepcopy(self)

        # replace real S1 data with scatterometer data interpolated to S1
        self_copy.S1_file['NRCS_VV'] = (['azimuth_time', 'ground_range'], new_nrcs)
        self_copy.S1_file['inc'] = (['azimuth_time', 'ground_range'], new_inc)

        # define names of variables to consider and return
        data_to_return = ['doppler_pulse_rg', 'doppler_pulse_rg_subscene', 'V_leakage_pulse_rg', 'V_leakage_pulse_rg_subscene', 'nrcs_scat', 'nrcs_scat_subscene']
        data_to_return_new_names = [name + '_inverted' for name in data_to_return[:-2]] + ['nrcs_scat_w_noise', 'nrcs_scat_subscene_w_noise']
        
        # repeat the  previous chain of computations NOTE this could be done more efficiently
        self_copy.create_dataset()
        self_copy.create_beam_mask()
        self_copy.compute_scatt_eqv_backscatter()
        self_copy.compute_beam_pattern()
        self_copy.compute_leakage_velocity()

        # add estimated leakage velocity back to original object
        self.data[data_to_return_new_names] = self_copy.data[data_to_return]
        return 


    def apply(self, 
              data_to_return: list[str] = 
                                ['doppler_pulse_rg', 'V_leakage_pulse_rg', 'nrcs_scat', 
                                 'doppler_pulse_rg_subscene', 'doppler_pulse_rg_subscene_inverted',
                                 'V_leakage_pulse_rg_subscene', 'V_leakage_pulse_rg_subscene_inverted',
                                 'nrcs_scat_subscene', 'doppler_pulse_rg_inverted',
                                 'V_leakage_pulse_rg_inverted', 'nrcs_scat_w_noise',
                                 'nrcs_scat_subscene_w_noise'],
                **kwargs):
        """

        """

        self.open_data()
        self.querry_era5()
        self.wdir_from_era5()
        self.create_dataset()
        self.create_beam_mask()
        self.compute_scatt_eqv_backscatter()
        self.compute_beam_pattern(**kwargs)
        self.compute_leakage_velocity(**kwargs)
        self.compute_leakage_velocity_estimate(**kwargs)

        self.data[data_to_return] = self.data[data_to_return].chunk('auto').compute()
        return
    









    # NOTE this function uses less RAM but takes significantly longer
    def _compute_scatt_eqv_backscatter(self):
        self.data['distance_az'] = (self.data["az"] - self.data["x_sat"]).isel(slow_time = 0)
        self.data['distance_ground'] = calculate_distance(x = self.data['distance_az'], y = self.data["grg"]) 
        self.data['inc_scatt_eqv'] = np.rad2deg(np.arctan(self.data['distance_ground']/self.z0))

        # self.data = self.data.unify_chunks()

        @dask.delayed
        def cmod_delayed(coord_value):
            slice_data = self.data.isel(slow_time=coord_value)

            result = cmod5n_forward(slice_data.windfield.values, self.data.attrs['wdir_wrt_sensor'] , self.data.inc_scatt_eqv.values)

            return  result

        delayed_result = [da.from_delayed(cmod_delayed(coord), 
                                    shape=self.data.inc_scatt_eqv.shape, 
                                    dtype=self.data.inc_scatt_eqv.dtype
                                    ) for coord in range(self.data.dims['slow_time'])]
        delayed_result = da.stack(delayed_result, axis = 0)
        self.data['nrcs_scat_eqv'] = (['slow_time', 'az_idx', 'grg'], delayed_result)

        return
    

    def _create_beam_mask(self):
        """
        
        """
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


    def __compute_scatt_eqv_backscatter(self):
        """

        """
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
        
        # calculate surface distance between sensor and point on surface as well as equivalent incidence angle for sensor
        self.data['distance_ground'] = calculate_distance(x = self.data["az"], y = self.data["grg"], x0 = self.data["x_sat"]) 
        self.data['inc_scatt_eqv'] = np.rad2deg(np.arctan(self.data['distance_ground']/self.z0))
        self.data = self.data.groupby('slow_time').map(windfield_over_slow_time, dims = ['az_idx', 'grg']).transpose('az_idx', 'grg', 'slow_time')
        return
    

    def _compute_beam_pattern(self, N: int = 10, w: float = 0.5):
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

        # chunk data to leverage dask's parallelization
        da_az_angle_wrt_boresight = da.from_array(x=self.data['az_angle_wrt_boresight'], chunks='auto')
        da_grg_angle_wrt_boresight = da.from_array(x=self.data['grg_angle_wrt_boresight'], chunks='auto')

        # Create a dictionary with kwargs for use in respective beam pattern
        kwargs_rg_tx = dict(L = self.height_antenna, f0 = self.f0)
        kwargs_rg_rx = dict(L = self.height_antenna, f0 = self.f0, N = N, w = w)
        kwargs_az_tx = dict(L = self.length_antenna, f0 = self.f0)
        kwargs_az_rx = dict(L = self.length_antenna, f0 = self.f0, N = N, w = w)

        beam_rg_tx = da_grg_angle_wrt_boresight.map_blocks(sinc_bp,  dtype='float', **kwargs_rg_tx)
        beam_az_tx = da_az_angle_wrt_boresight.map_blocks(sinc_bp,  dtype='float', **kwargs_az_tx)

        print("Parallizing beam-pattern computation...")
        if self.beam_pattern == "phased_array":
            # NOTE when using phased array, tapering is only applied on receive
            beam_az_rx = da_az_angle_wrt_boresight.map_blocks(phased_array,  dtype='float', new_axis = 0, **kwargs_az_rx).squeeze()
            beam_az = beam_az_tx * beam_az_rx

        else:
            beam_az = beam_az_tx**2
        beam_rg = beam_rg_tx**2
        print("Beam pattern computed")

        beam_grg_az = (beam_az * beam_rg).compute()
        self.data['beam_grg_az'] = (['az_idx', 'grg', 'slow_time'], beam_grg_az) 
        return
    
        
    def _compute_Doppler_leakage(self):
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
        data['beam_cutoff'] = data['beam'].sum(dim = ['az_idx', 'grg'])/ data['beam'].sum(dim = ['az_idx', 'grg']).max() < self.beam_weight_in_scene
        data[['doppler_pulse_rg', 'V_leakage_pulse_rg', 'nrcs_scat']][dict(slow_time=data['beam_cutoff'])] = np.nan

        # add attributes and coarsen data to resolution of subscenes
        self.data['V_leakage_pulse_rg'] = data['V_leakage_pulse_rg'].assign_attrs(units= 'm/s', description = 'Line of sight velocity ')
        self.subscenes = data[['doppler_pulse_rg', 'V_leakage_pulse_rg']].coarsen(grg=self.grg_N, slow_time=self.slow_time_N, boundary='trim').mean(skipna=False) 
        return

