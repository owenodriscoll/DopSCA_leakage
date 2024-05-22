import os
import io
import sys 
import glob
import warnings
import xsar
import dask
import copy
import pyproj
import xrft
import bottleneck # implicitely loaded somewhere
import numpy as np
import xarray as xr
import dask.array as da
from scipy.signal import firwin
from stereoid.oceans.GMF.cmod5n import cmod5n_inverse, cmod5n_forward
from drama.performance.sar.antenna_patterns import sinc_bp, phased_array

from .misc import round_to_hour, angular_difference, calculate_distance, era5_wind_point, era5_wind_area
from .add_dca import DCA_helper

from dataclasses import dataclass
import types
from typing import Callable, Union, List, Dict, Any

# --------- TODO LIST ------------
# FIXME Find correct beam pattern (tapering/pointing?) for receive and transmit, as well as correct N sensor elements
# FIXME unfortunate combinates of vx_sat, PRF and grid_spacing can lead to artifacts, maybe interpolation?
# FIXME consider gain compensation in range (weighting beam backscatter weight)
# FIXME maybe not use linear for interpolation from slant range to ground range?


# TODO add slant to ground range beyond boresight (non squinted), now only exactly on boresight
# TODO add land mask filter
# TODO add dask chunking
# TODO add docstrings
    # TODO add kwargs to input for phased beam pattern in create_beampattern
# TODO currently inversion interpolates scatterometer grid size to S1 grid, change to first apply inversion and then interpolate 


# NOTE Calculations assume 
    # NOTE a rectilinear geometry
    # NOTE perfect right-looking radar with no pitch, yaw and roll
    # NOTE range cell migration already corrected for
    # NOTE neglecting any Earth-rotation effects. 
    # NOTE same beam pattern on transmit and receive
    # NOTE two nrcs observations per estiamte such that speckle is reduced by sqrt(2)
# NOTE Weight is linearly scaled with relative nrcs (e.g. a nrcs of twice the average will yield relative weight of 2.0)
# NOTE Nrcs weight is calculated per azimuth line in slow time, not per slow time (i.e. not the average over grg and az per slow time)
# NOTE Assumes square Sentinel-1 pixels. The following processes rely on square pixels
    # NOTE ERA5 data is resampled to Sentinel-1 grid size and then smoothed
    # NOTE calculation of az_mask_pixels_cutoff, grg_N, slow_time_N
# NOTE currently number of antenna elements in azimuth and range are considered equal
# NOTE Assumes no squint in:
    # NOTE beam mask construction
    # NOTE velocity variance calculation from coherence loss
    # NOTE slant to ground range


# constants
c = 3E8


def mean_along_azimuth(x:xr.DataArray|xr.Dataset, azimuth_dim: str = 'az_idx', skipna: bool = False) -> xr.DataArray | xr.Dataset:
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

def angular_projection_factor(inc_original, inc_new = 90) -> float:
    """
    Computes multiplication factor to convert vector from one incidence to new one, e.g. from slant range to horizontal w.r.t. to the surface (if inc_new = 90)

    Input
    -----
    inc_original: float, array-like 
        incidence angle w.r.t. horizontal of vector, in degrees
    inc_new: float,array-like 
        new incidence angle in degrees. Defaults to 0 degrees (horizontal)

    Returns
    -------
    factor with which to multiply original vector to find projected vector's magnitude
    """
    return np.sin(np.deg2rad(inc_new))/np.sin(np.deg2rad(inc_original))

def dop2vel(Doppler, Lambda, angle_incidence, angle_azimuth, degrees = True):
    """
    Computes velocity corresponding to Doppler shift based on eq. 4.34 from Digital Procesing of Synthetic Aperture Radar Data by Ian G. Cumming 

    Input
    -----
    doppler: float, 
        relative frequency shift in Hz of surface or object w.r.t to the other
    Lambda: float,
        Wavelength of radio wave, in m
    angle_incidence: float, 
        incidence angle, in of wave with surface, degrees or radians
    angle_azimuth: float, 
        azimuthal angle with respect to boresight (0 for right looking system)
    degrees: bool,
        whether input angles are provided in degrees or radians

    Return
    -------
    velocity: float,
        relative velocity in m/s of surface or object w.r.t to the other
    """

    if degrees:
        angle_azimuth, angle_incidence = [np.deg2rad(i) for i in [angle_azimuth, angle_incidence]]

    return Lambda / 2 * Doppler / ( np.sin(angle_azimuth) * np.sin(angle_incidence))

def vel2dop(velocity, Lambda, angle_incidence, angle_azimuth, degrees = True):
    """
    Computes Doppler shift corresponding to velocity based on eq. 4.34 from Digital Procesing of Synthetic Aperture Radar Data by Ian G. Cumming 

    Input
    -----
    velocity: float, 
        relative velocity in m/s of surface or object w.r.t to the other
    Lambda: float,
        Wavelength of radio wave, in m
    angle_incidence: float, 
        incidence angle, in of wave with surface, degrees or radians
    angle_azimuth: float, 
        azimuthal angle with respect to boresight (0 for right looking system)
    degrees: bool,
        whether input angles are provided in degrees or radians

    Returns
    -------
    Doppler: float,
        frequency shift corresponding to input geometry and velocity, in Hz
    """

    if degrees:
        angle_azimuth, angle_incidence = [np.deg2rad(i) for i in [angle_azimuth, angle_incidence]]

    return 2 / Lambda * velocity * np.sin(angle_azimuth) * np.sin(angle_incidence)

def slant2ground(spacing_slant_range: float|int, height: float|int, ground_range_max: float|int, ground_range_min: float|int) -> float:
    """
    Converts a slant range pixel spacing to that projected onto the ground (assuming flat earth)

    Input
    -----
    spacing_slant_range:  float|int,
        slant range grid size, in meters
    height: float | int,
        height of platform, in meters
    ground_range_max: float|int,
        ground range projected maximum distance from satellite
    ground_range_min: float|int,
        ground range projected minimum distance from satellite  
    
    Returns
    -------
    new_grg_pixel: float,
        new ground range pixel spacing, in meters
    """
    current_distance = ground_range_max
    new_grg_pixel = []

    # iteratively compute new pixel spacing starting from the maximum extend
    while current_distance > ground_range_min:
        new_grg_pixel.append(current_distance)
        new_incidence = np.arctan(current_distance / height)
        current_distance -= (spacing_slant_range / np.sin(new_incidence))

    # reverse order to convert from decreasing to increasing ground ranges
    new_grg_pixel.reverse()

    return new_grg_pixel


def design_low_pass_filter_2D(da_shape: tuple[int, int], cutoff_frequency: float, fs_x: float, fs_y: float, window: str = 'hann'): 
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
    
    taps_x = firwin(numtaps = da_shape[0], cutoff=cutoff_frequency, fs=fs_x, pass_zero=True, window=window)
    taps_y = firwin(numtaps = da_shape[1], cutoff=cutoff_frequency, fs=fs_y, pass_zero=True, window=window)

    #generate 2D field
    filter_response = np.outer(taps_x, taps_y)
    
    return filter_response
    

def low_pass_filter_2D(da: xr.DataArray, cutoff_frequency: float, fs_x: float, fs_y: float, window: str = 'hann', fill_nans: bool = False, return_complex: bool = False) -> xr.DataArray:
    """
    Low pass filtering am xarray dataset in the Fourier domain using xrft, scipy.signal.windows and np.fft

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
        da_spec = xrft.fft(da_filled.chunk({**da_filled.sizes}), chunks_to_segmentsbool = False, shift = False)
    else:
        da_spec = xrft.fft(da_filled, shift = False)

    # design time-domain filter
    filter_response = design_low_pass_filter_2D(da_spec.shape, cutoff_frequency, fs_x=fs_x, fs_y=fs_y, window=window)

    # convert to fourier domain and multiply with spectrum (i.e. same as convolving filter with input image)
    filter_response_fourier = np.fft.fft2(filter_response)
    da_spec_filt = da_spec * filter_response_fourier
    
    if not return_complex:
        da_filt = da_filt.real
    if fill_nans:
        da_filt = da_filt.where(condition_fill.data, np.nan)

    return da_filt

def low_pass_filter_2D_dataset(ds: xr.Dataset, cutoff_frequency: float, fs_x: float, fs_y: float, window: str = 'hann', fill_nans: bool = False, return_complex: bool = False) -> xr.Dataset:
    """
    Wrapper of low_pass_filter_2D for each array in dataset

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

    ds_filt = xr.Dataset({
        var + '_subscene': low_pass_filter_2D(ds[var], 
                                              cutoff_frequency = cutoff_frequency, 
                                              fs_x=fs_x,
                                              fs_y=fs_y,
                                              window= window,
                                              fill_nans = fill_nans,
                                              return_complex = return_complex,
                                              ) for var in ds
        })

    # ensure dimensions of fft match those of input (sometimes rounding errors can occur)
    dimensions = [*ds.sizes]
    for dimension in dimensions:
        ds_filt[dimension] = ds[dimension]

    return ds_filt

def complex_speckle_noise(noise_shape: tuple, random_state: int = 42):
    """
    Generates complex multiplicative speckle noise 
    
    Intensity is obtained by (abs(speckle_complex)**2)/2, which has a mean and variance of 1

    Assumes Gaussian noise for real and imaginary components and uniform phase
    """
    np.random.seed(random_state)
    noise_real = np.random.randn(*noise_shape)
    noise_imag = np.random.randn(*noise_shape)
    speckle = np.array([complex(a,b) for a, b in zip(noise_real.ravel(), noise_imag.ravel())])

    return speckle.reshape(noise_shape)

def is_chunked_checker(da: xr.DataArray) -> bool:
    """checks whether inoput datarray is chunked"""
    return da.chunks is not None and any(da.chunks)

def padding_fourier(da: xr.DataArray, padding: int|tuple, dimension: str) -> xr.DataArray:
    """
    Interpolate data by zero-padding in Fourier domain along dimension
    """
    if is_chunked_checker(da):
        da = da.chunk({**da.sizes})

    da_spec = xrft.fft(da, true_amplitude=False) # set to false for same behaviour as np.fft
    da_spec_padded = xrft.padding.pad(da_spec, {'freq_' + dimension: padding}, constant_values = complex(0,0))

    # data is rechunked because Fourier transform cannot be performed over chunked dimension 
    if is_chunked_checker(da_spec_padded):
        da_spec_padded = da_spec_padded.chunk({**da_spec_padded.sizes})
    
    da_spec_padded *= da_spec_padded.sizes['freq_' + dimension]/da.sizes[dimension] # multiply times factor to compensate for more samples in Fourier domain
    da_padded = xrft.ifft(da_spec_padded, true_amplitude=False) # set to false for same behaviour as np.fft

    return da_padded

def da_ones_independent_samples(da: xr.DataArray, dim_to_resample: int = 0, samples_per_indepent_sample: int = 2) -> xr.DataArray:
    """
    Creates a new datarray filled with ones whose shape corresponds to the number of independent samples, rather than (oversampled) real samples
    """

    dim_interp = 'dim_'+str(dim_to_resample)
    a = da.shape
    b = np.ones(a)
    c = xr.DataArray(b)
    d = c.coarsen({dim_interp: samples_per_indepent_sample} ,boundary='trim').mean()
    return d

def compute_padding_1D(length_desired: int, length_current: int) -> tuple[int, int]:
    """
    Computes how much to pad on both sides of 1D dimension to obtain desired length
    """
    assert length_desired > length_current
    pad_total = length_desired - length_current
    pad = int(np.ceil(pad_total/2))
    return pad

@dataclass
class S1DopplerLeakage:
    """
    Parameters
    ----------
    filename: str, list
        filename or list of filenames of Sentinel-1 .SAFE
    f0: float
        radiowave frequency in hz
    z0: float
        average satellite orbit height in meters
    antenna_length: float

    antenna_height: float

    antenna_elements: int
        number of elements in azimuth and range (currently must be equal) used for phased array tapering on receive
    antenna_weighting: float
        element weighting in azimuth and range (currently must be equal) used for phased array tapering on receive
    beam_pattern: str
        "sinc" or "phased_array". Determines the beam width and side-lobe sensitivity
    swath_start_incidence_angle_scat: float
        incidence angle of scatterometer at first ground range (determines ground range distance)
    boresight_elevation_angle_scat: float
        center elevation angle of scatterometer beam center
    vx_sat: int
        along-azimuthal velocity of satellite, in meters per second
    PRF: int
        pulse repetition frequency
    az_footprint_cutoff: int
        along azimuth beam footprint to consider per pulse (outside is clipped off), in meters
    grid_spacing: int
        pixel spacing in meters to which Sentinel-1 SAFE files are coarsened
    resolution_product: int
        spatial resolution to which subscene results are averaged, e.g. if resolution_product = 25 km, moving average window is 12.5 km (pixel size permitting)
    product_averaging_window: str, 'hann'
        window with which to low pass towards product resolution. Window must be available from scipy.signal.get_window.
    era5_directory: str
        directory containing era5 file. Will look for file containing "{yyyy}{mm}.nc" or "era5{yy}{mm}{dd}h{hh}00_lat{}_lon{}.nc" to load
    era5_file: str
        specific file to load
    era5_undersample_factor: int
        factor by which to coarsen era5 resolution (after interpolating to grid_spacing)
    era5_smoothing_window: int
        window size for smoothing coarsened interpolated era5 data
    fill_nan_limit: int,
            Number of continuous missing pixels to fill along azimuth in loaded Sentinel-1 file. Select:
                - 0 for no filling, 
                - 1 for filling spurious missing pixels
                - None for no limit on filling (everything filled)
    random_state: int
        fixed random state seed for reproducibility of stochastic processes
    _pulsepair_noise: bool
        whether to add artifical noise from coherence loss due to pulse pair mechanism following Cramer Roa lower bound (recommended)
    _speckle_noise: bool
        whether to add artifical speckle to synthesized scatterometer product (recommended)
    _interpolator: str
        xarray interp1d complient interpolator

    Sources
    -------
        "Fois, F., Hoogeboom, P., Le Chevalier, F., & Stoffelen, A. (2015, July). DOPSCAT: A mission concept for a Doppler wind-scatterometer. 
            In 2015 IEEE International Geoscience and Remote Sensing Symposium (IGARSS) (pp. 2572-2575). IEEE."
        "Hoogeboom, P., Stoffelen, A., & Lopez-Dekker, P. (2018, October). DopSCA, Scatterometer-based Simultaneous Ocean Vector Current and 
            Wind Estimation. In 2018 Doppler Oceanography from Space (DOfS) (pp. 1-9). IEEE."
        "Rostan, F., Ulrich, D., Riegger, S., & Østergaard, A. (2016, July). MetoP-SG SCA wind scatterometer design and performance. In 2016 
            IEEE International Geoscience and Remote Sensing Symposium (IGARSS) (pp. 7366-7369). IEEE."

    Output:
    -------

    """

    filename: Union[str, list] 
    f0: float = 5.4E9                                                   # Hoogeboom et al,. (2018)
    z0: float = 824E3                                                   # 823-848 km e.g. in Fois et al,. (2015)
    antenna_length: float = 3.2                                         # for mid beam, Fois et al,. (2015)
    antenna_height: float = 0.3                                         # for mid beam, Fois et al,. (2015)
    antenna_elements: int = 4                                           # for mid beam, Rostan et al,. (2016)
    antenna_weighting: float = 0.5                                      # ?
    beam_pattern: str = "sinc"                                          # ?, presumed tapered
    swath_start_incidence_angle_scat: float = 30                        # custom, valid range of incidence angles is 20-65 degrees, Hoogeboom et al,. (2018) (is this for fore/aft beam or mid?)
    boresight_elevation_angle_scat: float = 40                          # ?
    vx_sat: int = 6800                                                  # Hoogeboom et al,. (2018)
    PRF: int = 4                                                        # PRF per antenna, total PRF is 32 Hz for 6 antennas, Hoogeboom et al,. (2018)
    az_footprint_cutoff: int = 80_000                                   # custom
    grid_spacing: int = 75                                              # assuming 150 m ground range resolution, Hoogeboom et al,. (2018)
    resolution_product: int = 25_000                                    # Hoogeboom et al,. (2018)
    product_averaging_window: str = 'hann'                              # window available from scipy.signal.get_window
    era5_directory: str = "" 
    era5_file: Union[bool, str] = False 
    era5_undersample_factor: int = 10
    era5_smoothing_window: Union[types.NoneType, int] = None
    fill_nan_limit: Union[types.NoneType, int] = 1
    random_state: int = 42
    _pulsepair_noise: bool = True
    _speckle_noise: bool = True
    _interpolator: str = 'linear'


    def __post_init__(self):

        self.Lambda = c / self.f0 # m
        self.stride = self.vx_sat / self.PRF
        self.az_mask_pixels_cutoff = int(self.az_footprint_cutoff/2//self.grid_spacing) 
        self.grg_N = int(self.resolution_product // self.grid_spacing)           # number of fast-time samples to average to scene size
        self.slow_time_N = int(self.resolution_product // self.stride)           # number of slow-time samples to average to scene size
        if type(self.era5_smoothing_window) == types.NoneType:                   # set smoothing window of era5 based on grid size of loaded S1 data
            self.era5_smoothing_window = int((200 / self.grid_spacing) * 15 )
            
        # Store attributes in object
        attributes_to_store = copy.deepcopy(self.__dict__)

        # if input values are None or Booleans, convert them to string type
        attributes_to_store_updated = {key: value if value is not None and type(value) is not bool else str(value) for key, value in attributes_to_store.items()}
        self.attributes_to_store = attributes_to_store_updated

        # warn if aliasing might occur due to combination of spatial and sampling resolutions 
        if self.stride % self.grid_spacing != 0:
            warnings.warn("Combination of vx_sat, PRF and grid_spacing may lead to aliasing: (vx_sat / PRF) % grid_spacing != 0")

    @staticmethod
    def convert_to_0_360(longitude):
        return (longitude + 360) % 360

    @staticmethod
    def decorrelation(tau, T):
        return np.exp(-(tau/T)**2) # 

    @staticmethod
    def pulse_pair_sigma_v(T_pp, T_corr_surface, T_corr_Doppler, SNR, Lambda, N_L = 1):
        """
        Calculates the Pulse pair velocity standard deviation within a resolution cell due to coherence loss

        NOTE assumes broadside geometry (non-squinted)

        Parameters
        ----------
        T_pp : scaler
            Intra pulse pair time separation
        T_corr_surface : scaler
            Decorrelation time of ocean surface at scales of radio wavelength of interest
        T_corr_Doppler : scaler
            Decorrelation time of velocities within resolution cell as a result of satellite motion during pulse-pair transmit
        SNR : scaler
            Signal to Noise ratio (for pulse-pair system we assume signal to clutter ratio of 1 dominates)
        Lambda : scaler
            Wavelength of considered radiowave
        N_L : int
            Number of independent looks for given area

        Returns
        -------
        Scaler of estimates surface velocity standard deviation

        """
        wavenumber = 2 * np.pi / Lambda 

        gamma_velocity = S1DopplerLeakage.decorrelation(T_pp, T_corr_Doppler) # eq 6 & 7 Rodriguez (2018), NOTE not valid for squint NOTE assumes Gaussian beam pattern
        gamma_temporal = S1DopplerLeakage.decorrelation(T_pp, T_corr_surface)
        gamma_SNR = SNR / (1 + SNR)

        gamma = gamma_temporal * gamma_SNR * gamma_velocity
        
        variance = (1 / (2*wavenumber*T_pp))**2 / (2*N_L) * (1-gamma**2)/gamma**2 # eq 14 Rodriguez (2018)

        return np.sqrt(variance), gamma


    # TODO add chunking
    def open_data(self):
        """
        Open data from a file or a list of files. First check if file can be reloaded
        """

        attrs_to_str = ['start_date', 'stop_date', 'footprint', 'multidataset']
        vars_to_keep = ['ground_heading', 'time', 'incidence', 'latitude', 'longitude', 'sigma0']
        coords_to_drop = "spatial_ref"
        pol = "VV"

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
                unique_ids = '_'.join(unique_ids)
                
            id_end = ref_file.rfind('/') + 1
            storage_dir = ref_file[:id_end]
            return storage_dir + unique_ids + f'_res{grid_spacing}.nc' 


        def find_unique_id(filename, unique_id_length = 4):
            """
            Function to find the last four digits (unique ID) of Sentinel-1 naming convention
            """
            id_start = filename.rfind('_') + 1 
            return filename[id_start:id_start+unique_id_length]


        def open_new_file(filename, grid_spacing):
            """
            Loads Sentinel-1 .SAFE file(s) and, if multiple, concatenate along azimuth (sorted by time)
            """

            dim_concat = "line"
            var_sortby = "time"

            # Surpress some warnings
            output_buffer = io.StringIO()
            stdout_save = sys.stdout
            sys.stdout = output_buffer

            # Use the catch_warnings context manager to temporarily catch warnings
            with warnings.catch_warnings(record=True) as caught_warnings:
                self._successful_files = []
                if isinstance(filename, str):
                    S1_file = xsar.open_dataset(filename, resolution=grid_spacing)
                    self._successful_files.append(filename)
                elif isinstance(filename, list):
                    _file_contents = []
                    for i, file in enumerate(filename):
                        try:
                            content_partial = xsar.open_dataset(file, resolution=grid_spacing)
                            _file_contents.append(content_partial)
                            self._successful_files.append(file)
                        except Exception as e:
                            # temporarily stop surpressing warnings
                            sys.stdout = stdout_save
                            print(f'File {i} did not load properly. \nConsider manually adding file content to _file_contents. File in question: {file} \n Error: {e}')
                            sys.stdout = output_buffer

                    # concatenate data ensuring that it is sorted by time. 
                    data_concatened = xr.concat(_file_contents, dim_concat)
                    data_sorted = data_concatened.sortby(var_sortby)

                    # redifine coordinate labels of concatened dimension
                    line_step = data_sorted.line.diff(dim=dim_concat).max()
                    S1_file = data_sorted.assign_coords(line = line_step.data*np.arange(data_sorted[dim_concat].size))

            # Reset system output to saved version
            sys.stdout = stdout_save
            return S1_file


        storage_name = create_storage_name(self.filename, self.grid_spacing)

        # reload file if possible
        if os.path.isfile(storage_name):
            print(f"Associated file found and reloaded: {storage_name}")
            self.S1_file = xr.open_dataset(storage_name)

        # else open .SAFE files, process and save as .nc
        else:
            self.S1_file = open_new_file(self.filename, self.grid_spacing)
            storage_name = create_storage_name(self._successful_files, self.grid_spacing)

            # convert to attributes to ones storeable as .nc
            for attr in attrs_to_str:
                self.S1_file.attrs[attr] = str(self.S1_file.attrs[attr])

            # drop coordinate with a lot of attributes
            self.S1_file = self.S1_file.drop_vars(coords_to_drop)

            # keep only VV polarisation
            self.S1_file = self.S1_file.sel(pol = pol)

            # store as .nc file ...
            self.S1_file[vars_to_keep].to_netcdf(storage_name)

            # ... and reload
            self.S1_file = xr.open_dataset(storage_name)

            print(f"No pre-saved file found, instead saved loaded file as: {storage_name}")
        return


    def querry_era5(self):
        """
        Opens or downloads relevant ERA5 data to find the wind direction w.r.t. to the radar. 
        1. Will first attempt to load a specific file (if provided).
        2. Otherwise will check if previously downloaded file covers area of interest. 
        3. Lastely, will submit new download request.
        
        This method can be skipped by manually providing a "wdir_wrt_sensor" attribute to "self".
        Either an integer/float as an average over the area of interest, or an array, in which case the
        array must be the same shape as the backscatter array in the S1_file.
        """

        var_time = "time"
        var_lon = "longitude"
        var_lat = "latitude"
        var_azi = "line"
        var_grg = "sample"

        date = self.S1_file[var_time].min().values.astype('datetime64[m]').astype(object)
        date_rounded = round_to_hour(date)
        yy, mm, dd, hh = date_rounded.year, date_rounded.month, date_rounded.day, date_rounded.hour
        time = f"{hh:02}00"

        latitudes = self.S1_file[var_lat]
        longitudes = self.S1_file[var_lon]
        latmean, lonmean = latitudes.mean().values*1, longitudes.mean().values*1
        latmin, latmax = latitudes.min().values*1, latitudes.max().values*1
        lonmin, lonmax = longitudes.min().values*1, longitudes.max().values*1
        lonmin, lonmax = [self.convert_to_0_360(i) for i in [lonmin, lonmax]] # NOTE correction for fact that ERA5 goes between 0 - 360

        if type(self.era5_file) == str:
            era5 = xr.open_dataset(self.era5_file)
        else:
            sub_str = str(yy) + str(mm)
            try:
                # try to find if monthly data file exists which to load 
                era5_filename = [s for s in glob.glob(f"{self.era5_directory}*") if sub_str + '.nc' in s][0]
            except:
                #  if not, try to find single estimate hour ERA5 wind file
                era5_filename = f"era5_{yy}{mm:02d}{dd:02d}h{time}_lat{latmean:.1f}_lon{lonmean:.1f}.nc"
                era5_filename = era5_filename.replace('.', '_',  2)
                if not self.era5_directory is None:
                    era5_filename = os.path.join(self.era5_directory, era5_filename)

                # if neither monthly file nor hourly single estimate file exist, download new single estimate hour
                if not os.path.isfile(era5_filename):
                    era5_wind_area(year = yy,
                          month = mm,
                          day = dd,
                          time = time,
                          lonmin = lonmin,
                          lonmax = lonmax,
                          latmin = latmin,
                          latmax = latmax,
                          filename = era5_filename,
                          )
            
            print(f"Loading nearest ERA5 point w.r.t. observation from ERA5 file: {era5_filename}")
            era5 = xr.open_dataset(era5_filename)
            
        era5_subset_time = era5.sel(
            time = np.datetime64(date_rounded, 'ns'),
            method = 'nearest')

        era5_subset = era5_subset_time.sel(
            longitude = self.convert_to_0_360(self.S1_file[var_lon]),
            latitude = self.S1_file[var_lat],
            method = 'nearest')

        # ERA5 data is subsampled
        # this should not affect resolution much as, for example, the resolution of 1/4 deg ERA5 is approx 50km, 
        # The resampled grid size should still be ~50km (S1 resolution * era5_smoothing_window * era5_undersample_factor ~ 50km)
        resolution_condition = self.grid_spacing * self.era5_undersample_factor * (1 * self.era5_smoothing_window) # NOTE * 1 because only mean window operations considered

        # check whether resampling is unacceptable
        # resolution is ideally twice the grid spacing and 1 degree is approximately 100km
        resolution_era5_deg = 2 * min([abs(era5.latitude.diff(dim = 'latitude')).min(), abs(era5.longitude.diff(dim = 'longitude')).min()])
        resolution_era5_m = 100e3 * resolution_era5_deg 
        message = lambda x : f"Warning: interpolated ERA5 data may be {x[0]}. \nConsider plotting fields in self.era5 for inspection and/or {x[1]} the undersample_factor or the smoothing_window size."
        
        if resolution_condition >= 1.33 * resolution_era5_m: # arbitrary threshold at which ERA5 resolution may be degraded
            print(message(['over-smoothed', 'decreasing']))
        elif resolution_condition <= 0.50 * resolution_era5_m: # arbitrary threshold at which grainy ERA5 is expected (maybe cause artifacts)
            print(message(['under-smoothed', 'increasing']))

        # create a placeholder dataset to avoid having to subsample/oversample on datetime vectors
        azimuth_time_placeholder = np.arange(era5_subset.sizes[var_azi])
        ground_range_placeholder = np.arange(era5_subset.sizes[var_grg])
        era5_placeholder = era5_subset.assign_coords({var_azi:azimuth_time_placeholder, var_grg:ground_range_placeholder})

        # Subsample the dataset by interpolation
        new_azimuth_time = np.arange(era5_subset.sizes[var_azi]/self.era5_undersample_factor) * self.era5_undersample_factor
        new_ground_range = np.arange(era5_subset.sizes[var_grg]/self.era5_undersample_factor) * self.era5_undersample_factor
        era5_subsamp = era5_placeholder.interp({var_azi:new_azimuth_time, var_grg:new_ground_range}, method='linear')

        # first perform median filter (performs better if anomalies are present),then mean filter to smooth edges
        era5_smoothed = era5_subsamp.rolling({var_azi : self.era5_smoothing_window, var_grg : self.era5_smoothing_window}, center = True, min_periods=2).median()
        era5_smoothed = era5_smoothed.rolling({var_azi : self.era5_smoothing_window, var_grg : self.era5_smoothing_window}, center = True, min_periods=2).mean()

        # re-interpolate to the native resolution and add to object
        era5_interp = era5_smoothed.interp({var_azi : azimuth_time_placeholder, var_grg : ground_range_placeholder}, method='linear')
        self.era5 = era5_interp.assign_coords({var_azi : era5_subset[var_azi], var_grg : era5_subset[var_grg]})
        return 


    def wdir_from_era5(self):
        """
        Calculates the wind direction with respect to the sensor using the ground geometry and era5 wind vectors
        """

        var_lon = "longitude"
        var_lat = "latitude"
        wdir_era5 = np.rad2deg(np.arctan2(self.era5.u10, self.era5.v10))

        # compute ground footprint direction
        geodesic = pyproj.Geod(ellps='WGS84')
        corner_coords = [self.S1_file[var_lon][0, 0], self.S1_file[var_lat][0, 0], self.S1_file[var_lon][-1,0], self.S1_file[var_lat][-1,0]]
        ground_dir, _, _ = geodesic.inv(*corner_coords)

        # compute directional difference between satelite and era5 wind direction
        self.wdir_wrt_sensor = angular_difference(ground_dir, wdir_era5)
        return


    def create_dataset(self, var_nrcs: str = "sigma0", var_inc: str = "incidence"):
        """
        Creates a new xarray dataset with the coordinates and dimensions of interest using the Sentinel-1 data. 
        Computes the windfield given the Sentinel-1 and ERA5 data

        Parameters
        ----------
        var_nrcs: str,
            name for variable containing nrcs
        var_inc: str,
            name for variable containing incidence angle
        """  
        dim_az = "az"
        dim_grg = "grg"

        # calculate new ground range and azimuth range belonging to observation with scatterometer viewing geometry
        grg_offset = np.tan(np.deg2rad(self.swath_start_incidence_angle_scat)) * self.z0
        grg = np.arange(self.S1_file[var_nrcs].data.shape[1]) * self.grid_spacing + grg_offset
        az = (np.arange(self.S1_file[var_nrcs].data.shape[0]) - self.S1_file[var_nrcs].data.shape[0]//2) * self.grid_spacing
        x_sat = np.arange(az.min(), az.max(), self.stride)

        # create new dataset 
        data = xr.Dataset(
            data_vars=dict(
                nrcs = ([dim_az, dim_grg], self.S1_file[var_nrcs].data, {'units': 'm2/m2'}),
                inc = ([dim_az, dim_grg], self.S1_file[var_inc].data, {'units': 'Degrees'}),
            ),
            coords=dict(
                az = ([dim_az], az, {'units': 'm'}),
                grg = ([dim_grg], grg, {'units': 'm'}),
            ),
            attrs=dict(
                grid_spacing = self.grid_spacing),
        )

        # find points with nan's or poor backscatter estimates
        # condition_pre = data['nrcs'].isnull()
        condition_to_fix = ((data['nrcs'].isnull()) | (data['nrcs'] <= 0))
        data['nrcs'] = data['nrcs'].where(~condition_to_fix)

        # fill nans using limit, limit = 1 fills only single missing pixels,limit = None fill all, limit = 0 filters nothing, not consistent missing data
        interpolater = lambda x: x.interpolate_na(dim = dim_az, method= 'linear', limit= self.fill_nan_limit, fill_value= 'extrapolate')
        data['nrcs'] = interpolater(data['nrcs'])
        conditions_post = ((data['nrcs'].isnull()) |(data['nrcs'] <= 0))

        # add windfield
        if isinstance(self.wdir_wrt_sensor, xr.DataArray):
            data['wdir_wrt_sensor'] = ([dim_az, dim_grg], self.wdir_wrt_sensor.data)
        elif isinstance(self.wdir_wrt_sensor, (float, int)):
            data['wdir_wrt_sensor'] = ([dim_az, dim_grg], np.ones_like(data['nrcs']) * self.wdir_wrt_sensor)
        windfield = cmod5n_inverse(data["nrcs"].data, data['wdir_wrt_sensor'].data, data["inc"].data)

        data['windfield'] = ([dim_az, dim_grg], windfield)
        data["windfield"] = data["windfield"].assign_attrs(units= 'm/s', description = 'CMOD5n Windfield for Sentinel-1 backscatter')

        # Remove data that, even after filling, does not meet criteria
        data = data.where(~(conditions_post))

        # add another dimension for later use
        x_sat = da.arange(data[dim_az].min(), data[dim_az].max(), self.stride)
        slow_time = self.stride * da.arange(x_sat.shape[0])
        x_sat = xr.DataArray(x_sat, dims='slow_time', coords={'slow_time': slow_time})
        data = data.assign(x_sat=x_sat)

        # update with previously stored data
        data.attrs.update(self.attributes_to_store)

        self.data = data
        return
    

    def create_beam_mask(self): 
        """
        Function to remove data outside beam pattern footprint as determined by "az_mask_pixels_cutoff". 
        Creates a new stack of (potentially overlapping) observations centered on a new azimuthmul coordinate system
        """

        # find indexes along azimuth with beam beam center NOTE does not work for squinted geometries
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
        dim_new_res = self.data.attrs['grid_spacing']

        # array with azimuth indexes to select over slow time
        idx_az = np.array([masks[i] for i in idx_slow_time])
        self.idx_az = idx_az

        # prepare data by chunking and conversion to lower bit
        self.data = self.data.astype('float32').chunk('auto')
        for i, st in enumerate(idx_slow_time):

            a = self.data.isel(slow_time = st, az = idx_az[i])
            a = a.assign_coords({dim_new: (dim_filter, dim_new_res*np.arange(a.sizes[dim_filter]))})
            a = a.swap_dims({dim_filter:dim_new})
            a = a.reset_coords(names=dim_filter)

            _data.append(a)

        self.data = xr.concat(_data, dim = 'slow_time')
        return 
    

    def compute_scatt_eqv_backscatter(self):
        """
        From the submitted viewing geometry, calculates the nrcs as would be observed by the scatterometer
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
            Wrapper of CMOD5.n to enable dask's lazy computations
            
            input
            -----
            ds: xr.Dataset, dataset containing the fields 'windfield', 'inc_scatt_eqv' and the attribute 'wdir_wrt_sensor'
            dims: list, list of strings containing the dimensions for which the nrcs is calculated per the equivalent scatterometer incidence

            output
            ------
            ds: xr.Dataset, dataset containing a new variable 'nrcs_scat_eqv'
            """

            nrcs_scatterometer_equivalent = cmod5n_forward(ds['windfield'].data, ds['wdir_wrt_sensor'].data, ds['inc_scatt_eqv_cube'].data)
            ds['nrcs_scat_eqv'] = (dimensions, nrcs_scatterometer_equivalent, {'units': 'm/s'}) 
            return ds

        self.data = self.data.map_blocks(windfield_over_slow_time)
        self.data = self.data.astype('float32')
        return


    def compute_beam_pattern(self):
        """
        Computes a beam pattern to be applied along the entire dataset.
        NOTE assumes similar antenna parameters for both range and azimuth

        Input
        -----
        antenna_elements int; number of anetenna elements that are considered in beam tapering, only affects when beam pattern = phased_array
        antenna_weighting: float, int; weighting parameter as defined by the called beam pattern functions. only affects when beam pattern = phased_array
        """
        
        self.data['distance_slant_range'] = np.sqrt(self.data['distance_ground']**2 + self.z0**2)
        self.data['az_angle_wrt_boresight'] = np.arcsin((self.data['distance_az'])/self.data['distance_slant_range']) # NOTE arcsin instead of tan as distance_slant_range includes azimuthal angle
        self.data['grg_angle_wrt_boresight'] = np.deg2rad(self.data['inc_scatt_eqv'] - self.boresight_elevation_angle_scat)
        self.data = self.data.transpose('az_idx', 'grg', 'slow_time')

        # NOTE the following computations are directly computed, a delayed lazy computation may be better
        # NOTE Currently assumes same antenna elements and weighting in range as in azimuth 
        N = self.antenna_elements 
        w = self.antenna_weighting 
        
        # Assumes same patter on transmit and receive
        if self.beam_pattern == "sinc":
            beam_az_tx = sinc_bp(sin_angle=self.data.az_angle_wrt_boresight, L = self.antenna_length, f0 = self.f0)
            beam_az = beam_az_tx ** 2
        elif self.beam_pattern == "phased_array":
            # beam_az_tx = phased_array(sin_angle=self.data.az_angle_wrt_boresight, L = self.antenna_length, f0 = self.f0, N = N, w = w).squeeze()
            beam_az_rx = phased_array(sin_angle=self.data.az_angle_wrt_boresight, L = self.antenna_length, f0 = self.f0, N = N, w = w).squeeze()
            beam_az = beam_az_rx**2 # * beam_az_tx
        
        beam_grg_tx = sinc_bp(sin_angle=self.data.grg_angle_wrt_boresight, L = self.antenna_height, f0 = self.f0)
        beam_grg_rx = beam_grg_tx
        beam_grg = beam_grg_tx * beam_grg_rx
        beam = beam_az * beam_grg
        self.data['beam'] = ([*self.data.az_angle_wrt_boresight.sizes], beam)

        self.data = self.data.astype('float32')
        return

    def compute_leakage_velocity(self, add_pulse_pair_uncertainty = True):
        """
        Computes Line of Sight (LoS) leakage Doppler and velocity considering the beam pattern, nrcs weighting and geometric Doppler
        NOTE range-dependend gain compensation (weight_rg) returns overestimated signal at sidelobe nulls
        NOTE assumes no squint and a flat Earth
        NOTE computes LoS Dopplers
        NOTE backscatter weight calculated per range line

        Parameters
        ----------
        add_pulse_pair_uncertainty : Bool
            whether to add pulse-pair uncertainty following Hoogeboom et al., (2018)
        """

        # compute geometrical doppler, beam pattern and nrcs weigths
        self.data['nrcs_weight'] = (self.data['nrcs_scat_eqv'] / self.data['nrcs_scat_eqv'].mean(dim=['az_idx'])) 
        self.data['beam_weight'] = (self.data['beam'] / self.data['beam'].mean(dim=['az_idx'])) 
        self.data['elevation_angle'] = np.radians(self.data['inc_scatt_eqv']) # NOTE assumes flat Earth

        self.data['dop_geom'] = vel2dop(
            velocity=self.vx_sat,
            Lambda=self.Lambda,
            angle_incidence=self.data['elevation_angle'],
            angle_azimuth=self.data['az_angle_wrt_boresight'],
            degrees=False,
        ) 

        self.data['dop_beam_weighted'] = self.data['dop_geom'] * self.data['beam_weight'] * self.data['nrcs_weight']

        # beam and backscatter weighted geometric Doppler is interpreted as geophysical Doppler, i.e. Leakage
        self.data['V_leakage'] = dop2vel(
            Doppler=self.data['dop_beam_weighted'],
            Lambda=self.Lambda,
            angle_incidence=self.data['elevation_angle'],
            angle_azimuth= np.pi/2, # the geometric doppler is interpreted as LoS motion, so azimuth angle component must result in value of 1 (i.e. pi/2)
            degrees=False,
        )
        
        # calculate scatterometer nrcs at scatterometer resolution (integrate nrcs)
        self.data['nrcs_scat'] = mean_along_azimuth(self.data['nrcs_scat_eqv'] * self.data['beam_weight'])
        # sum over azimuth to receive range-slow_time results
        self.data[['doppler_pulse_rg', 'V_leakage_pulse_rg']] = mean_along_azimuth(self.data[['dop_beam_weighted', 'V_leakage']])
        
        # add attribute
        self.data['V_leakage_pulse_rg'] = self.data['V_leakage_pulse_rg'].assign_attrs(units= 'm/s', description = 'Line of Sight velocity ')
        self.data['doppler_pulse_rg'] = self.data['doppler_pulse_rg'].assign_attrs(units= 'Hz', description = 'Line of Sight Doppler frequency ')

        # add pulse pair velocity uncertainty
        if (self._pulsepair_noise) & (add_pulse_pair_uncertainty):
            wavenumber = 2 * np.pi / self.Lambda

            # -- calculates average azimuthal beam standard deviation within -3 dB 
            beam_db = 10 * np.log10(self.data.beam)
            beam_3dB = xr.where((beam_db- beam_db.max(dim = 'az_idx'))< -3, np.nan, 1)*self.data.az_angle_wrt_boresight
            sigma_az_angle = beam_3dB.std(dim = 'az_idx').mean().values*1

            T_corr_Doppler = 1 / (np.sqrt(2) * wavenumber * self.vx_sat * sigma_az_angle) # equation 7 from Rodriguez et al., (2018)
            T_pp = 1.15E-4 # intra pulse-pair pulse separation time, Hoogeboom et al., (2018)
            U = 6 # Average wind speed assumed of 6 m/s
            T_corr_surface = 3.29 * self.Lambda / U # Decorrelation time of surface at radio frequency of interest (below eq. 19 Theodosious et al., 2023)
            SNR = 1 # because SNR is dominated by signal to clutter, which for Pulse Pair is approx 1

            # NOTE assumes no squint
            self.velocity_error, self.gamma = self.pulse_pair_sigma_v(
                T_pp = T_pp, 
                T_corr_surface = T_corr_surface, 
                T_corr_Doppler = T_corr_Doppler, 
                SNR = SNR,  
                Lambda = self.Lambda)

        else:
            self.velocity_error = 0
            
        # compute approximate ground range spatial resolution of scatterometer (slant range resolution =/= ground range resolution)
        grg_for_safekeeping = self.data.grg

        # NOTE to do this nicely the azimuth angle should also be taken into account, as it slightly affects local incidence angle
        self.new_grg_pixel = slant2ground(
            spacing_slant_range=self.grid_spacing,
            height=self.z0,
            ground_range_max=self.data.grg.max().data*1,
            ground_range_min=self.data.grg.min().data*1,
            )

        # ------ interpolate data to new grg range pixels (effectively a variable low pass filter) -------
        self.data = self.data.interp(grg=self.new_grg_pixel, method=self._interpolator)

        # fix random state 
        np.random.seed(self.random_state)
        reference = 'V_leakage_pulse_rg'
        dim_interp = 'grg'
        shape_ref = self.data[reference].shape
        da_ones_independent = da_ones_independent_samples(self.data[reference], dim_to_resample= 0, samples_per_indepent_sample=2)
        V_pp = self.velocity_error * np.random.randn(*da_ones_independent.shape)
        da_V_pp = V_pp * da_ones_independent

        pad = compute_padding_1D(length_desired=self.data[reference].sizes[dim_interp], length_current=da_V_pp.sizes['dim_0'])
        
        # after padding in the Fourier domain the output is complex, complex part should be negligible
        V_pp_c = padding_fourier(da_V_pp, padding = (pad, pad), dimension= 'dim_0')

        # since iid noise, we can clip time domain to correct dimensions without affecting statistics 
        V_pp = V_pp_c[:shape_ref[0], :shape_ref[1]]
        self.V_pp_c = V_pp_c

        self.data['V_pp'] = xr.zeros_like(self.data['V_leakage_pulse_rg']) + V_pp.data
        self.data['V_sigma'] = self.data['V_leakage_pulse_rg'] + self.data['V_pp'].real # complex part should be negligible

        # ------- re-interpolate to higher sampling to maintain uniform ground samples -------
        self.data = self.data.interp(grg=grg_for_safekeeping, method=self._interpolator)
        self.data = self.data.astype('float32')

        # low-pass filter scatterometer data to subscene resolution
        data_4subscene= ['doppler_pulse_rg', 'V_leakage_pulse_rg', 'V_sigma', 'nrcs_scat']
        data_subscene = [name + '_subscene' for name in data_4subscene]

        fs_x, fs_y = 1/self.grid_spacing, 1/self.stride
        data_lp = low_pass_filter_2D_dataset(self.data[data_4subscene], 
                                             cutoff_frequency = 1 / (self.resolution_product), 
                                             fs_x=fs_x, 
                                             fs_y=fs_y,
                                             window=self.product_averaging_window,
                                             fill_nans=True) # FIXME why *2 ?
        self.data[data_subscene] = data_lp
        # self.data[data_subscene] = self.data[data_4subscene].rolling(grg=self.grg_N, slow_time=self.slow_time_N, center=True).mean()
        return


    def compute_leakage_velocity_estimate(self):
        """
        Method that estimates the leakage velocity using the scatterometer backscatter field. 
        Can be done more efficiently (multiple repeated calculations)

        Input
        -----
        speckle_noise: bool; whether to apply multiplicative speckle noise to scatterometer estimated nrcs
        """
        var_nrcs = "sigma0"
        var_inc = "incidence"
        var_azi = "line"
        var_grg = "sample"

        # find indexes of S1 scene that were cropped (outside full beam pattern)
        idx_start = self.idx_az[0][self.az_mask_pixels_cutoff]
        idx_end = self.idx_az[-1][self.az_mask_pixels_cutoff]

        # create placeholder S1 data (pre-cropped)
        new_nrcs = np.nan * np.ones_like(self.S1_file[var_nrcs])
        new_inc = np.nan * np.ones_like(self.S1_file[var_nrcs])

        # add speckle noise
        if self._speckle_noise:
            # noise_multiplier = self.speckle_noise(nrcs_scat_grg_interpolated.shape, random_state = self.random_state)

            reference = 'nrcs_scat'
            dim_interp = 'grg'
            nrcs_scat_grg = self.data[reference]
            nrcs_scat_sl = nrcs_scat_grg.interp(grg=self.new_grg_pixel, method=self._interpolator)

            shape_ref = nrcs_scat_sl.shape
            da_ones_independent = da_ones_independent_samples(nrcs_scat_sl, dim_to_resample= 0, samples_per_indepent_sample=2)
            # d = da_ones_independent_samples(test.data.nrcs_scat)
            speckle_c = complex_speckle_noise(da_ones_independent.shape, random_state=self.random_state)
            da_speckle_c = speckle_c * da_ones_independent

            # pad in fourier domain
            pad = compute_padding_1D(length_desired=nrcs_scat_sl.sizes[dim_interp], length_current=da_speckle_c.sizes['dim_0'])
            da_speckle_c_padded = padding_fourier(da_speckle_c, padding = (pad, pad), dimension= 'dim_0')

            da_speckle_c_padded_cut = da_speckle_c_padded[:shape_ref[0], :shape_ref[1]]
            self.da_speckle_c_padded_cut = da_speckle_c_padded_cut
            speckle = abs(da_speckle_c_padded_cut)**2 / 2

            nrcs_scat_sl_speckle = nrcs_scat_sl * speckle.data            
            # interpolate to grg
            nrcs_scat_speckle = nrcs_scat_sl_speckle.interp(grg=self.data.grg, method=self._interpolator)

        else:
            # already up and downscaled so not necessary here
            nrcs_scat_speckle = self.data.nrcs_scat
    
        # interpolate estimated scatterometer data back to S1 grid size
        slow_time_upsamp = np.linspace(self.data.slow_time[0], self.data.slow_time[-1], idx_end - idx_start) 
        nrcs_scat_upsamp = nrcs_scat_speckle.T.interp(slow_time = slow_time_upsamp)
        inc_scat_upsamp = self.data.inc_scatt_eqv_cube.mean(dim='az_idx').T.interp(slow_time = slow_time_upsamp)

        # apply cropping 
        new_nrcs[idx_start: idx_end, :] = nrcs_scat_upsamp
        new_inc[idx_start: idx_end, :] = inc_scat_upsamp

        # copy existing object to avoid overwritting
        self_copy = copy.deepcopy(self)

        # replace real S1 data with scatterometer data interpolated to S1
        self_copy.S1_file[var_nrcs] = ([var_azi, var_grg], new_nrcs)
        self_copy.S1_file[var_inc] = ([var_azi, var_grg], new_inc)

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
                                ['doppler_pulse_rg', 'V_leakage_pulse_rg', 'nrcs_scat', 'V_sigma', 'V_sigma_subscene',
                                 'doppler_pulse_rg_subscene', 'doppler_pulse_rg_subscene_inverted',
                                 'V_leakage_pulse_rg_subscene', 'V_leakage_pulse_rg_subscene_inverted',
                                 'nrcs_scat_subscene', 'doppler_pulse_rg_inverted',
                                 'V_leakage_pulse_rg_inverted', 'nrcs_scat_w_noise',
                                 'nrcs_scat_subscene_w_noise'],
                **kwargs):
        """
        Calls relevant methods in correct order and performs final computations after dask's lazy computations
        """

        self.open_data()
        self.querry_era5()
        self.wdir_from_era5()
        self.create_dataset()
        self.create_beam_mask()
        self.compute_scatt_eqv_backscatter()
        self.compute_beam_pattern(**kwargs)
        self.compute_leakage_velocity()
        self.compute_leakage_velocity_estimate()

        self.data[data_to_return] = self.data[data_to_return].chunk('auto').persist()
        return
    



def add_dca_to_leakage_class(cls: S1DopplerLeakage, files_dca) -> None:
    """function to add computed dca to S1DopplerLeakage class""" 

    obj_copy = copy.deepcopy(cls)

    dca_interp, wb_interp = DCA_helper(filenames=files_dca,
        latitudes=obj_copy.S1_file.latitude.values,
        longitudes=obj_copy.S1_file.longitude.values).add_dca()

    # add interpolated DCA to S1 file
    obj_copy.S1_file['dca'] = (['azimuth_time', 'ground_range'], dca_interp)
    obj_copy.S1_file['wb'] = (['azimuth_time', 'ground_range'], wb_interp)
    obj_copy.create_dataset()
    obj_copy.data['dca_s1'] = (['az', 'grg'], obj_copy.S1_file['dca'].data)
    obj_copy.data['wb_s1'] = (['az', 'grg'], obj_copy.S1_file['wb'].data)
    obj_copy.create_beam_mask()
    obj_copy.data = obj_copy.data.astype('float32')

    reprojection_factor = angular_projection_factor(
        inc_original = cls.data['inc'],
        inc_new = cls.data['inc_scatt_eqv']
    )

    obj_copy.data['dca_scatt'] = obj_copy.data['dca_s1'] * reprojection_factor
    obj_copy.data['wb_scatt'] = obj_copy.data['wb_s1'] * reprojection_factor
    
    
    cls.data[['dca', 'wb']] = obj_copy.data[['dca_scatt', 'wb_scatt']] * cls.data['beam_weight'] * cls.data['nrcs_weight']
    cls.data[['dca_pulse_rg', 'wb_pulse_rg']] = mean_along_azimuth(cls.data[['dca', 'wb']] )
    cls.data['doppler_w_dca'] =  (obj_copy.data['dca_scatt'] + cls.data['dop_geom']) * cls.data['beam_weight'] * cls.data['nrcs_weight']
    cls.data['doppler_w_dca_pulse_rg'] = mean_along_azimuth(cls.data['doppler_w_dca'])
    
    cls.data = cls.data.astype('float32')


    cls.data[['V_dca', 'V_wb']] = dop2vel(
            Doppler=cls.data[['dca', 'wb']],
            Lambda=cls.Lambda,
            angle_incidence=cls.data['inc_scatt_eqv'],
            angle_azimuth= 90, # the geometric doppler is interpreted as LoS motion, so azimuth angle component must result in value of 1 (i.e. pi/2 or 90 deg)
            degrees=True,
        )
    cls.data[['V_dca_pulse_rg', 'V_wb_pulse_rg']] = mean_along_azimuth(cls.data[['V_dca', 'V_wb']])

    cls.data['V_doppler_w_dca'] = dop2vel(
            Doppler=cls.data['doppler_w_dca'],
            Lambda=cls.Lambda,
            angle_incidence=cls.data['inc_scatt_eqv'],
            angle_azimuth= 90, # the geometric doppler is interpreted as LoS motion, so azimuth angle component must result in value of 1 (i.e. pi/2 or 90 deg)
            degrees=True,
        )

    cls.data['V_doppler_w_dca_pulse_rg'] = mean_along_azimuth(cls.data['V_doppler_w_dca'])

    # convert computed avriables from slant to ground range resolution prior to spatial averaging
    grg_for_safekeeping = cls.data.grg
    vars_slant2ground = ['wb', 'wb_pulse_rg', 'dca', 'dca_pulse_rg', 'doppler_w_dca', 'doppler_w_dca_pulse_rg', 'V_wb', 'V_dca', 'V_dca_pulse_rg', 'V_wb_pulse_rg', 'V_doppler_w_dca']    
    temp = cls.data[vars_slant2ground].interp(grg=cls.new_grg_pixel, method="linear")
    cls.data[vars_slant2ground] = temp.interp(grg=grg_for_safekeeping, method="linear")

    # perform averaging as prescribed in cls
    scenes_to_average = ['wb_pulse_rg', 'V_wb_pulse_rg', 'dca_pulse_rg', 'doppler_w_dca_pulse_rg', 'V_dca_pulse_rg', 'V_doppler_w_dca_pulse_rg']
    scenes_averaged = [i+'_subscene' for i in scenes_to_average]
    # cls.data[scenes_averaged] = cls.data[scenes_to_average].rolling(grg=cls.grg_N, slow_time=cls.slow_time_N, center=True).mean()

    fs_x, fs_y = 1/cls.grid_spacing, 1/cls.stride
    data_lp = low_pass_filter_2D_dataset(cls.data[scenes_to_average], 
                                         cutoff_frequency = 1/(cls.resolution_product), 
                                         fs_x=fs_x, 
                                         fs_y=fs_y,
                                         window=cls.product_averaging_window,
                                         fill_nans=True) # FIXME why *2 ?
    cls.data[scenes_averaged] = data_lp

    # a bit of reschuffling of coordinates
    cls.data = cls.data.transpose('az_idx', 'grg', 'slow_time').chunk('auto')
    cls.data[scenes_to_average + scenes_averaged] = cls.data[scenes_to_average + scenes_averaged].persist()
    return 