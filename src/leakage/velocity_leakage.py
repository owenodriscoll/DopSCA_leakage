import warnings
import xsar  # implicitely loaded somewhere
import dask  # implicitely loaded somewhere
import copy
import pyproj # implicitely loaded somewhere
import xrft # implicitely loaded somewhere
import bottleneck  # implicitely loaded somewhere
import numpy as np
import xarray as xr
import dask.array as da
from stereoid.oceans.GMF.cmod5n import cmod5n_inverse, cmod5n_forward
from drama.performance.sar.antenna_patterns import sinc_bp, phased_array
from dataclasses import dataclass

from .misc import calculate_distance
from .utils import mean_along_azimuth
from .conversions import dB, phase2vel, dop2vel, vel2dop, slant2ground
from .uncertainties import pulse_pair_coherence, generate_complex_speckle, speckle_intensity, phase_error_generator 
from .low_pass_filter import low_pass_filter_2D_dataset
from .frequency_domain_padding import padding_fourier, da_integer_oversample_like, compute_padding_1D
from .open_SAFE import open_S1
from .era5_download import wdir_from_era5, querry_era5
from . import constants

import types
from typing import Callable, Union, List, Dict, Any

# --------- TODO LIST ------------
# FIXME Find correct beam pattern (tapering/pointing?) for receive and transmit, as well as correct N sensor elements
# FIXME unfortunate combinates of vx_sat, PRF and grid_spacing can lead to artifacts, maybe interpolation?
# FIXME maybe not use linear for interpolation from slant range to ground range?


# TODO add slant to ground range beyond boresight (non squinted), now only exactly on boresight
# TODO add land mask filter
# TODO add dask chunking
# TODO add docstrings
    # TODO add kwargs to input for phased beam pattern in create_beampattern
# TODO currently inversion interpolates scatterometer grid size to S1 grid, change to first apply inversion and then interpolate 
# TODO replace current ERA5 interpolation with zero-padding in Fourier domain


# NOTE Calculations assume 
    # NOTE a rectilinear geometry
    # NOTE perfect right-looking radar with no pitch, yaw and roll, no squint assumed in 
        # NOTE beam mask construction
        # NOTE velocity variance calculation from coherence loss
        # NOTE slant to ground range
    # NOTE range cell migration already corrected for
    # NOTE neglecting any Earth-rotation effects. 
    # NOTE same beam pattern on transmit and receive
    # NOTE Assumes square Sentinel-1 pixels. The following processes rely on square pixels
        # NOTE ERA5 data is resampled to Sentinel-1 grid size and then smoothed
        # NOTE calculation of az_mask_pixels_cutoff
    # NOTE number of antenna elements in azimuth and range are considered equal
    # NOTE pulse pair noise assumes a pair of pulses, not any other combination


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
        length of antenna for computation of azimuthal component of beam pattern
    antenna_height: float
        height of antenna for computation of range component of beam pattern
    antenna_elements: int
        number of elements in azimuth and range (currently must be equal), only used for phased array tapering
    antenna_weighting: float
        element weighting in azimuth and range (currently must be equal), only used for phased array tapering
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
    SNR: float
        Signal to Noise Ratio for calculation of velocity variance. Because SNR is dominated by signal to clutter for pulse pair in DopSCA, SNR is approx 1 on average
    T_pp: float
        Separation time (in seconds) between pulses of pulse_pair
    az_footprint_cutoff: int
        along azimuth beam footprint to consider per pulse (outside is clipped off), in meters
    grid_spacing: int
        pixel spacing at which Sentinel-1 SAFE files are loaded
    resolution_product: int
        surface-projected spatial resolution to which subscene results are averaged
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

    """

    filename: Union[str, list]
    f0: float = 5.4e9  # Hoogeboom et al,. (2018)
    z0: float = 824e3  # 823-848 km e.g. in Fois et al,. (2015)
    antenna_length: float = 3.2  # for mid beam, Fois et al,. (2015)
    antenna_height: float = 0.3  # for mid beam, Fois et al,. (2015)
    antenna_elements: int = 4  # for mid beam, Rostan et al,. (2016)
    antenna_weighting: float = 0.75  # ?
    beam_pattern: str = "sinc"  # ?, presumed tapered
    swath_start_incidence_angle_scat: float = (
        30  # custom, valid range of incidence angles is 20-65 degrees, Hoogeboom et al,. (2018) (is this for fore/aft beam or mid?)
    )
    boresight_elevation_angle_scat: float = 40  # ?
    vx_sat: int = 6800  # Hoogeboom et al,. (2018)
    PRF: int = (
        4  # PRF per antenna, total PRF is 32 Hz for 6 antennas, Hoogeboom et al,. (2018)
    )
    SNR: float = 1.0  # Signal to noise ratio, for Pulse Pair is approx 1 on average
    T_pp: float = 1.15e-4  # pulse-pair separation time
    az_footprint_cutoff: int = 80_000  # custom
    grid_spacing: int = (
        75  # assuming 150 m ground range resolution, Hoogeboom et al,. (2018)
    )
    resolution_product: int = 25_000  # Hoogeboom et al,. (2018)
    product_averaging_window: str = "hann" 
    era5_directory: str = ""
    era5_file: Union[bool, str] = False
    era5_undersample_factor: int = 10
    era5_smoothing_window: Union[types.NoneType, int] = None
    fill_nan_limit: Union[types.NoneType, int] = 1
    random_state: int = 42
    _pulsepair_noise: bool = True
    _speckle_noise: bool = True
    _interpolator: str = "linear"
    _gamma_hardcode: Union[types.NoneType, float] = None

    def __post_init__(self):

        self.Lambda = constants.c / self.f0
        self.stride_theoretical = self.vx_sat / self.PRF
        self.az_mask_pixels_cutoff = int(
            self.az_footprint_cutoff / 2 // self.grid_spacing
        )

        # Store attributes in object
        attributes_to_store = copy.deepcopy(self.__dict__)

        # If input values are None or Booleans, convert them to string type
        attributes_to_store_updated = {
            key: value if value is not None and type(value) is not bool else str(value)
            for key, value in attributes_to_store.items()
        }
        self.attributes_to_store = attributes_to_store_updated

        # warn if aliasing might occur due to combination of spatial and sampling resolutions
        if self.stride_theoretical % self.grid_spacing != 0:
            warnings.warn(
                "Combination of vx_sat, PRF and grid_spacing may lead to aliasing: (vx_sat / PRF) % grid_spacing != 0"
            )

    def load_S1(self):
        """
        Open Sentinel-1 data from a file or a list of files. First check if file can be reloaded
        """

        self.S1_file = open_S1(self.filename, grid_spacing=self.grid_spacing)

        return

    def load_era5(self):
        """
        Opens or downloads relevant ERA5 data to find the wind direction w.r.t. to the radar.
        1. Will first attempt to load a specific file (if provided).
        2. Otherwise will check if previously downloaded file covers area of interest.
        3. Lastely, will submit new download request.

        This method can be skipped by manually providing a "wdir_wrt_sensor" attribute to "self".
        Either an integer/float as an average over the area of interest, or an array, in which case the
        array must be the same shape as the backscatter array in the S1_file.

        We do some ERA5 interpolating to the NRCS grid with some arbitrary choiced, these don't really affect results since ERA5 is coarse to begin with
        """

        var_time = "time"
        var_lon = "longitude"
        var_lat = "latitude"

        date = (
            self.S1_file[var_time].min().values.astype("datetime64[m]").astype(object)
        )

        latitudes = self.S1_file[var_lat]
        longitudes = self.S1_file[var_lon]

        era_data = querry_era5(
            date=date,
            latitudes=latitudes,
            longitudes=longitudes,
            grid_spacing=self.grid_spacing,
            era5_file=self.era5_file,
            directory=self.era5_directory,
            era5_smoothing_window=self.era5_smoothing_window,
            era5_undersample_factor=self.era5_undersample_factor
            )

        corner_coords = [
            longitudes[0, 0],
            latitudes[0, 0],
            longitudes[-1, 0],
            latitudes[-1, 0],
        ]

        self.wdir_wrt_sensor = wdir_from_era5(era_data, corner_coords)

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
        grg = (
            np.arange(self.S1_file[var_nrcs].data.shape[1]) * self.grid_spacing
            + grg_offset
        )
        az = (
            np.arange(self.S1_file[var_nrcs].data.shape[0])
            - self.S1_file[var_nrcs].data.shape[0] // 2
        ) * self.grid_spacing

        # create new dataset
        data = xr.Dataset(
            data_vars=dict(
                nrcs=(
                    [dim_az, dim_grg],
                    self.S1_file[var_nrcs].data,
                    {"units": "m2/m2"},
                ),
                inc=(
                    [dim_az, dim_grg],
                    self.S1_file[var_inc].data,
                    {"units": "Degrees"},
                ),
            ),
            coords=dict(
                az=([dim_az], az, {"units": "m"}),
                grg=([dim_grg], grg, {"units": "m"}),
            ),
            attrs=dict(grid_spacing=self.grid_spacing),
        )

        # find points with nan's or poor backscatter estimates
        condition_to_fix = (data["nrcs"].isnull()) | (data["nrcs"] <= 0)
        data["nrcs"] = data["nrcs"].where(~condition_to_fix)

        # fill nans using limit, limit = 1 fills only single missing pixels,limit = None fill all, limit = 0 filters nothing, not consistent missing data
        interpolater = lambda x: x.interpolate_na(
            dim=dim_az,
            method="linear",
            limit=self.fill_nan_limit,
            fill_value="extrapolate",
        )
        data["nrcs"] = interpolater(data["nrcs"])
        conditions_post = (data["nrcs"].isnull()) | (data["nrcs"] <= 0)

        # add wind direction to data
        if isinstance(self.wdir_wrt_sensor, xr.DataArray):
            data["wdir_wrt_sensor"] = ([dim_az, dim_grg], self.wdir_wrt_sensor.data)
        elif isinstance(self.wdir_wrt_sensor, (float, int)):
            data["wdir_wrt_sensor"] = (
                [dim_az, dim_grg],
                np.ones_like(data["nrcs"]) * self.wdir_wrt_sensor,
            )

        # compute wind field
        windfield = cmod5n_inverse(
            data["nrcs"].data, data["wdir_wrt_sensor"].data, data["inc"].data
        )
        data["windfield"] = ([dim_az, dim_grg], windfield)
        data["windfield"] = data["windfield"].assign_attrs(
            units="m/s", description="CMOD5n Windfield for Sentinel-1 backscatter"
        )

        # Remove data that, even after filling, does not meet criteria
        data = data.where(~(conditions_post))

        # update with previously stored data
        data.attrs.update(self.attributes_to_store)

        self.data = data
        return

    def create_beam_mask(self):
        """
        Function to remove data outside beam pattern footprint as determined by "az_mask_pixels_cutoff".
        Creates a new stack of (potentially overlapping) observations centered on a new azimuthmul coordinate system

        NOTE does not work for squinted geometries
        """

        dim_original = "grg"
        dim_filter = "az"
        dim_new = "slow_time"
        dim_window = "az_idx"

        self.window_size = np.round(self.az_footprint_cutoff / self.grid_spacing).astype("int")
        self.stride_elements = np.round(self.vx_sat / self.PRF / self.grid_spacing).astype("int")

        self.data = self.data.chunk({dim_filter: 1, dim_original:"auto"})
        self.data = self.data.rolling({dim_filter: self.window_size}, center=True).construct({dim_filter:dim_window}, stride=self.stride_elements)
        
        stride = (self.data[dim_filter][-1] - self.data[dim_filter][0]) / (self.data[dim_filter].sizes[dim_filter]-1)
        self.stride = float(stride.data)
        
        slow_time = da.arange(self.data[dim_filter].sizes[dim_filter]) * self.stride
        
        self.data = self.data.assign_coords(
                        {dim_new: (dim_filter, slow_time),
                        dim_window: (self.data[dim_window] * self.grid_spacing)}
                    )
        self.data = self.data.swap_dims({dim_filter: dim_new})
        self.data = self.data.reset_coords(names=dim_filter)

        delta = 1E-10
        self.to_clip = np.round(self.window_size/self.stride_elements + delta, 0).astype(int)
        self.start_idx = np.round(self.to_clip / 2).astype(int)
        self.end_idx = (self.to_clip - self.start_idx).astype(int)
        self.data = self.data.isel({dim_new : slice(self.start_idx, -self.end_idx)})
        self.data = self.data.chunk({dim_new: "auto", dim_original:"auto", dim_window:-1})
        return

    def compute_scatt_eqv_backscatter(self):
        """
        From the submitted viewing geometry, calculates the nrcs as would be observed by the scatterometer
        """

        dim_window = "az_idx"
        self.data["distance_az"] = (self.data[dim_window] - self.data[dim_window].mean())
        self.data["distance_ground"] = calculate_distance(
            x=self.data["distance_az"], y=self.data["grg"]
        )
        self.data["inc_scatt_eqv"] = np.rad2deg(
            np.arctan(self.data["distance_ground"] / self.z0)
        )
        slow_time_vector = self.data.slow_time
        self.data["inc_scatt_eqv_cube"] = self.data["inc_scatt_eqv"].expand_dims(
            dim={"slow_time": slow_time_vector}
        )

        self.data = self.data.transpose("az_idx", "grg", "slow_time")
        self.data = self.data.astype("float32").unify_chunks()

        def windfield_over_slow_time(ds, dimensions=["az_idx", "grg", "slow_time"]):
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

            nrcs_scatterometer_equivalent = cmod5n_forward(
                ds["windfield"].data,
                ds["wdir_wrt_sensor"].data,
                ds["inc_scatt_eqv_cube"].data,
            )
            ds["nrcs_scat_eqv"] = (
                dimensions,
                nrcs_scatterometer_equivalent,
                {"units": "m/s"},
            )
            return ds

        self.data = self.data.map_blocks(windfield_over_slow_time)
        self.data = self.data.astype("float32")
        return

    def compute_beam_pattern(self):
        """
        Computes a beam pattern to be applied along the entire dataset.
        # NOTE The following computations are directly computed, a delayed lazy computation may be better
        # NOTE Currently tapering only possible in azimuth
        # NOTE Same beam pattern assumed for transmit and receive
        # NOTE Beam patterns are already in intensity, square is only needed for two-way pattern

        Input
        -----
        antenna_elements int; number of anetenna elements that are considered in beam tapering, only affects when beam pattern = phased_array
        antenna_weighting: float, int; weighting parameter as defined by the called beam pattern functions. only affects when beam pattern = phased_array
        """

        self.data["distance_slant_range"] = calculate_distance(
            x=self.data["distance_ground"], y=self.z0
        )
        self.data["az_angle_wrt_boresight"] = np.arcsin(
            (self.data["distance_az"]) / self.data["distance_slant_range"]
        ) # NOTE arcsin instead of tan as distance_slant_range includes azimuthal angle
        self.data["grg_angle_wrt_boresight"] = np.deg2rad(
            self.data["inc_scatt_eqv"] - self.boresight_elevation_angle_scat
        )
        self.data = self.data.transpose("az_idx", "grg", "slow_time")

        N = self.antenna_elements
        w = self.antenna_weighting

        # Assumes same patter on transmit and receive
        if self.beam_pattern == "sinc":
            beam_az = sinc_bp(
                sin_angle=np.sin(self.data.az_angle_wrt_boresight),
                L=self.antenna_length,
                f0=self.f0,
            )
            beam_az_two_way = beam_az**2
        elif self.beam_pattern == "phased_array":
            beam_az = phased_array(
                sin_angle=np.sin(self.data.az_angle_wrt_boresight),
                L=self.antenna_length,
                f0=self.f0,
                N=N,
                w=w,
            ).squeeze()
            beam_az_two_way = beam_az**2

        beam_grg = sinc_bp(
            sin_angle=np.sin(self.data.grg_angle_wrt_boresight),
            L=self.antenna_height,
            f0=self.f0,
        )
        beam_grg_two_way = beam_grg**2

        beam = beam_az_two_way * beam_grg_two_way
        self.data["beam"] = ([*self.data.az_angle_wrt_boresight.sizes], beam)

        self.data = self.data.astype("float32")
        return


    def compute_leakage_velocity(self, add_pulse_pair_uncertainty=True):
        """
        Computes leakage Doppler and velocity considering the beam pattern, nrcs weighting and geometric Doppler
        NOTE assumes no squint and a flat Earth

        Parameters
        ----------
        add_pulse_pair_uncertainty : Bool
            whether to add pulse-pair uncertainty following Hoogeboom et al., (2018)
        """

        # compute geometrical doppler, beam pattern and nrcs weigths
        self.data["beam_weight"] = self.data["beam"] / mean_along_azimuth(
            self.data["beam"]
        ) 
        self.data["weight"] = (
            self.data["beam"] * self.data["nrcs_scat_eqv"]
        ) / mean_along_azimuth(self.data["beam"] * self.data["nrcs_scat_eqv"])
        self.data["elevation_angle"] = np.radians(
            self.data["inc_scatt_eqv"]
        )  # NOTE assumes flat Earth
        self.data["elevation_angle_scat"] = mean_along_azimuth(
            np.radians(self.data["inc_scatt_eqv_cube"])
        )

        self.data["dop_geom"] = vel2dop(
            velocity=self.vx_sat,
            Lambda=self.Lambda,
            angle_incidence=self.data["elevation_angle"],  # NOTE assumes flat Earth
            angle_azimuth=self.data["az_angle_wrt_boresight"],
            degrees=False,
        )

        self.data["dop_beam_weighted"] = (
            self.data["dop_geom"] * self.data["weight"]
        ) 

        # beam and backscatter weighted geometric Doppler is interpreted as geophysical Doppler, i.e. Leakage
        self.data["V_leakage"] = dop2vel(
            Doppler=self.data["dop_beam_weighted"],
            Lambda=self.Lambda,
            angle_incidence=self.data["elevation_angle"],
            angle_azimuth=np.pi
            / 2,  # the geometric doppler is interpreted as radial motion, so azimuth angle component must result in value of 1 (i.e. pi/2)
            degrees=False,
        )

        # calculate scatterometer nrcs at scatterometer resolution (integrate nrcs)
        self.data["nrcs_scat"] = mean_along_azimuth(
            self.data["nrcs_scat_eqv"] * self.data["beam_weight"]
        )

        # sum over azimuth to receive range-slow_time results
        self.data[["doppler_pulse_rg", "V_leakage_pulse_rg"]] = mean_along_azimuth(
            self.data[["dop_beam_weighted", "V_leakage"]]
        )

        # compute approximate ground range spatial resolution of scatterometer (slant range resolution =/= ground range resolution)
        grg_for_safekeeping = self.data.grg

        # NOTE to do this nicely the azimuth angle should also be taken into account, as it slightly affects local incidence angle
        self.new_grg_pixel = slant2ground(
            spacing_slant_range=self.grid_spacing,
            height=self.z0,
            ground_range_max=self.data.grg.max().data * 1,
            ground_range_min=self.data.grg.min().data * 1,
        )

        # ------ interpolate data to new grg range pixels (effectively a variable low pass filter) -------
        self.data = self.data.interp(grg=self.new_grg_pixel, method=self._interpolator)

        # fix random state
        np.random.seed(self.random_state)
        reference = "V_leakage_pulse_rg"
        dim_interp = "grg"
        shape_ref = self.data[reference].shape

        # assumes data along resample dim is oversampled by a factor 2, such that there are two samples per independent sample resolution
        da_ones_independent = da_integer_oversample_like(
            self.data[reference], dim_to_resample=0, new_samples_per_original_sample=2
        )

        # add pulse pair velocity uncertainty
        if (self._pulsepair_noise) & (add_pulse_pair_uncertainty):
            wavenumber = 2 * np.pi / self.Lambda

            # -- calculates average azimuthal beam standard deviation within -3 dB
            beam_db = dB(self.data.beam)
            beam_3dB = (
                xr.where((beam_db - beam_db.max(dim="az_idx")) < -3, np.nan, 1)
                * self.data.az_angle_wrt_boresight
            )
            sigma_az_angle = beam_3dB.std(dim="az_idx").mean().values * 1

            T_corr_Doppler = 1 / (np.sqrt(2) * wavenumber * self.vx_sat * sigma_az_angle)  
            U = 6  # Average wind speed assumed of 6 m/s, this matters very little for used parameters
            T_corr_surface = 3.29 * self.Lambda / U

            # NOTE assumes no squint
            self.gamma = pulse_pair_coherence(
                T_pp=self.T_pp,
                T_corr_surface=T_corr_surface,
                T_corr_Doppler=T_corr_Doppler,
                SNR=self.SNR
            )

            phase_uncertainty = phase_error_generator(
                gamma=self.gamma,
                n_samples=(da_ones_independent.shape),
                random_state=self.random_state,
            )
            
            self.velocity_error = phase2vel(
                phase=phase_uncertainty, 
                wavenumber=wavenumber,
                T=self.T_pp
            )

        else:
            self.velocity_error = 0

        da_V_pp = self.velocity_error * da_ones_independent

        pad = compute_padding_1D(
            length_desired=self.data[reference].sizes[dim_interp],
            length_current=da_V_pp.sizes["dim_0"],
        )

        # after padding in the Fourier domain the output is complex, complex part should be negligible
        self.da_V_pp = da_V_pp
        self.pad = pad
        self.V_pp_c = padding_fourier(da_V_pp, padding=(pad, pad), dimension="dim_0")

        # since iid noise, we can clip time domain to correct dimensions without affecting statistics
        V_pp = self.V_pp_c[: shape_ref[0], : shape_ref[1]]
        V_pp = V_pp.real # NOTE if complex part is not negligible this will slightly underestimate pulsepair noise

        self.data["V_pp"] = (
            xr.zeros_like(self.data["V_leakage_pulse_rg"]) + V_pp.data
        ) / np.sin(self.data["elevation_angle_scat"])
        self.data["V_sigma"] = (
            self.data["V_leakage_pulse_rg"] + self.data["V_pp"]
        )  

        # ------- re-interpolate to higher sampling to maintain uniform ground samples -------
        self.data = self.data.interp(grg=grg_for_safekeeping, method=self._interpolator)
        self.data = self.data.astype("float32")

        # low-pass filter scatterometer data to subscene resolution
        data_4subscene = [
            "doppler_pulse_rg",
            "V_leakage_pulse_rg",
            "V_sigma",
            "nrcs_scat",
        ]
        data_subscene = [name + "_subscene" for name in data_4subscene]

        fs_x, fs_y = 1 / self.grid_spacing, 1 / self.stride
        data_lowpass = low_pass_filter_2D_dataset(
            self.data[data_4subscene],
            cutoff_frequency=1 / (self.resolution_product),
            fs_x=fs_x,
            fs_y=fs_y,
            window=self.product_averaging_window,
            fill_nans=True,
        )
        self.data[data_subscene] = data_lowpass

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
        delta = 1E-10
        idx_start = int(np.round(self.window_size/2 + delta)) #self.idx_az[0][self.az_mask_pixels_cutoff]
        idx_end = int(self.S1_file.sizes['line'] - np.round(self.window_size/2 + delta)) #self.idx_az[-1][self.az_mask_pixels_cutoff]

        # create placeholder S1 data (pre-cropped)
        new_nrcs = np.nan * np.ones_like(self.S1_file[var_nrcs])
        new_inc = np.nan * np.ones_like(self.S1_file[var_nrcs])

        # add speckle noise
        if self._speckle_noise:

            reference = "nrcs_scat"
            dim_interp = "grg"
            nrcs_scat_grg = self.data[reference]
            nrcs_scat_slrg = nrcs_scat_grg.interp(
                grg=self.new_grg_pixel, method=self._interpolator
            )

            shape_ref = nrcs_scat_slrg.shape
            da_ones_independent = da_integer_oversample_like(
                nrcs_scat_slrg, dim_to_resample=0, new_samples_per_original_sample=2
            )
            speckle_c = generate_complex_speckle(
                da_ones_independent.shape, random_state=self.random_state
            )
            da_speckle_c = speckle_c * da_ones_independent

            # pad in fourier domain
            pad = compute_padding_1D(
                length_desired=nrcs_scat_slrg.sizes[dim_interp],
                length_current=da_speckle_c.sizes["dim_0"],
            )
            da_speckle_c_padded = padding_fourier(
                da_speckle_c, padding=(pad, pad), dimension="dim_0"
            )

            # we can again clip noise array since it is still independent of NRCS
            da_speckle_c_padded_cut = da_speckle_c_padded[
                : shape_ref[0], : shape_ref[1]
            ]
            self.da_speckle_c_padded_cut = da_speckle_c_padded_cut

            # compute real speckle and add to nrcs
            speckle = speckle_intensity(da_speckle_c_padded_cut)
            nrcs_scat_speckle_slrg = nrcs_scat_slrg * speckle.data

            # interpolate to grg
            nrcs_scat_speckle = nrcs_scat_speckle_slrg.interp(
                grg=self.data.grg, method=self._interpolator
            )

        else:
            # already up and downscaled so not necessary here
            nrcs_scat_speckle = self.data.nrcs_scat

        # interpolate estimated scatterometer data back to S1 grid size
        slow_time_upsamp = np.linspace(
            self.data.slow_time[0], self.data.slow_time[-1], idx_end - idx_start
        )
        nrcs_scat_upsamp = nrcs_scat_speckle.T.interp(slow_time=slow_time_upsamp)
        inc_scat_upsamp = np.degrees(self.data["elevation_angle_scat"]).T.interp(
            slow_time=slow_time_upsamp
        )

        # apply cropping
        new_nrcs[idx_start:idx_end, :] = nrcs_scat_upsamp
        new_inc[idx_start:idx_end, :] = inc_scat_upsamp

        # copy existing object to avoid overwritting
        self_copy = copy.deepcopy(self)

        # replace real S1 data with scatterometer data interpolated to S1
        self_copy.S1_file[var_nrcs] = ([var_azi, var_grg], new_nrcs)
        self_copy.S1_file[var_inc] = ([var_azi, var_grg], new_inc)

        # define names of variables to consider and return
        data_to_return = [
            "doppler_pulse_rg",
            "doppler_pulse_rg_subscene",
            "V_leakage_pulse_rg",
            "V_leakage_pulse_rg_subscene",
            "nrcs_scat",
            "nrcs_scat_subscene",
        ]
        data_to_return_new_names = [
            name + "_inverted" for name in data_to_return[:-2]
        ] + ["nrcs_scat_w_noise", "nrcs_scat_subscene_w_noise"]

        # repeat the  previous chain of computations NOTE this could be done more efficiently
        self_copy.create_dataset()
        self_copy.create_beam_mask()
        self_copy.compute_scatt_eqv_backscatter()
        self_copy.compute_beam_pattern()
        self_copy.compute_leakage_velocity()

        # add estimated leakage velocity back to original object
        self.data[data_to_return_new_names] = self_copy.data[data_to_return]

        return


    def apply(
        self,
        data_to_return: list[str] = [
            "az",
            "doppler_pulse_rg",
            "V_leakage_pulse_rg",
            "nrcs_scat",
            "V_sigma",
            "V_sigma_subscene",
            "doppler_pulse_rg_subscene",
            "doppler_pulse_rg_subscene_inverted",
            "V_leakage_pulse_rg_subscene",
            "V_leakage_pulse_rg_subscene_inverted",
            "nrcs_scat_subscene",
            "doppler_pulse_rg_inverted",
            "V_leakage_pulse_rg_inverted",
            "nrcs_scat_w_noise",
            "nrcs_scat_subscene_w_noise",
        ],
        **kwargs,
    ):
        """
        Calls relevant methods in correct order and performs final computations after dask's lazy computations
        """

        self.load_S1()
        self.load_era5()
        self.create_dataset()
        self.create_beam_mask()
        self.compute_scatt_eqv_backscatter()
        self.compute_beam_pattern(**kwargs)
        self.compute_leakage_velocity()
        self.compute_leakage_velocity_estimate()

        self.data[data_to_return] = self.data[data_to_return].persist()
        
        return