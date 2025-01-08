import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import LinearNDInterpolator
from dataclasses import dataclass
from typing import Callable, Union, Sequence

from .velocity_leakage import S1DopplerLeakage
from .conversions import angular_projection_factor, dop2vel
from .utils import mean_along_azimuth
from .low_pass_filter import low_pass_filter_2D_dataset
import copy


@dataclass
class DCA_helper:

    filenames: list[str]
    latitudes: Union[float, Sequence[float]]
    longitudes: Union[float, Sequence[float]]

    def load_dca(self):
        """
        Load individual DCA files and store the DCA and land_area_fraction data per lat/lon point
        """

        longitudes = []
        latitudes = []
        dcas = []
        wbs = []
        lafs = []

        # extract data in each file
        for file in self.filenames:
            data = xr.open_dataset(file)
            longitudes.append(data.lon.values.ravel())
            latitudes.append(data.lat.values.ravel())
            dcas.append(data.doppler_centroid_anomaly.values.ravel())
            wbs.append(data.wave_bias.values.ravel())
            lafs.append(data.land_area_fraction.values.ravel())

        # combine data from all files into single dataframe
        np.concatenate(dcas, axis=0)
        df = pd.DataFrame(
            data={
                "lon": np.concatenate(longitudes, axis=0),
                "lat": np.concatenate(latitudes, axis=0),
                "dca": np.concatenate(dcas, axis=0),
                "wb": np.concatenate(wbs, axis=0),
                "laf": np.concatenate(lafs, axis=0),
            }
        )

        # drop fill values
        df.where(abs(df) < 1e10, np.nan, inplace=True)
        df.dropna(inplace=True)
        df = df.astype(np.float32)
        self.df = df

    def interpolate_dca(self):
        """
        Interpolate extracted data to specified latitude and longitude points
        """

        # create interpolate objects
        interp_dca = LinearNDInterpolator(
            list(zip(self.df.lon.values, self.df.lat.values)), self.df.dca.values
        )
        interp_wb = LinearNDInterpolator(
            list(zip(self.df.lon.values, self.df.lat.values)), self.df.wb.values
        )
        interp_laf = LinearNDInterpolator(
            list(zip(self.df.lon.values, self.df.lat.values)), self.df.laf.values
        )

        # apply interpolation for specified points
        dca_interp = interp_dca(self.longitudes, self.latitudes)
        wb_interp = interp_wb(self.longitudes, self.latitudes)
        laf_interp = interp_laf(self.longitudes, self.latitudes)

        # filter pointswhere land area fraction suggests land contamination
        self.dca_interp = np.where(laf_interp > 0, np.nan, dca_interp)
        self.wb_interp = np.where(laf_interp > 0, np.nan, wb_interp)

    def add_dca(self):
        self.load_dca()
        self.interpolate_dca()
        return self.dca_interp, self.wb_interp


def add_dca_to_leakage_class(cls: S1DopplerLeakage, files_dca) -> None:
    """function to add computed dca to S1DopplerLeakage class"""

    obj_copy = copy.deepcopy(cls)

    dca_interp, wb_interp = DCA_helper(
        filenames=files_dca,
        latitudes=obj_copy.S1_file.latitude.values,
        longitudes=obj_copy.S1_file.longitude.values,
    ).add_dca()

    # add interpolated DCA to S1 file
    obj_copy.S1_file["dca"] = (["azimuth_time", "ground_range"], dca_interp)
    obj_copy.S1_file["wb"] = (["azimuth_time", "ground_range"], wb_interp)
    obj_copy.create_dataset()
    obj_copy.data["dca_s1"] = (["az", "grg"], obj_copy.S1_file["dca"].data)
    obj_copy.data["wb_s1"] = (["az", "grg"], obj_copy.S1_file["wb"].data)
    obj_copy.create_beam_mask()
    obj_copy.data = obj_copy.data.astype("float32")

    reprojection_factor = angular_projection_factor(
        inc_original=cls.data["inc"], inc_new=cls.data["inc_scatt_eqv"]
    )

    obj_copy.data["dca_scatt"] = obj_copy.data["dca_s1"] * reprojection_factor
    obj_copy.data["wb_scatt"] = obj_copy.data["wb_s1"] * reprojection_factor

    cls.data[["dca", "wb"]] = (
        obj_copy.data[["dca_scatt", "wb_scatt"]] * cls.data["weight"]
    )
    cls.data[["dca_pulse_rg", "wb_pulse_rg"]] = mean_along_azimuth(
        cls.data[["dca", "wb"]]
    )
    cls.data["doppler_w_dca"] = (
        obj_copy.data["dca_scatt"] + cls.data["dop_geom"]
    ) * cls.data["weight"]
    cls.data["doppler_w_dca_pulse_rg"] = mean_along_azimuth(cls.data["doppler_w_dca"])

    cls.data = cls.data.astype("float32")

    cls.data[["V_dca", "V_wb"]] = dop2vel(
        Doppler=cls.data[["dca", "wb"]],
        Lambda=cls.Lambda,
        angle_incidence=cls.data["inc_scatt_eqv"],
        angle_azimuth=90,  # the geometric doppler is interpreted as radial motion, so azimuth angle component must result in value of 1 (i.e. pi/2 or 90 deg)
        degrees=True,
    )
    cls.data[["V_dca_pulse_rg", "V_wb_pulse_rg"]] = mean_along_azimuth(
        cls.data[["V_dca", "V_wb"]]
    )

    cls.data["V_doppler_w_dca"] = dop2vel(
        Doppler=cls.data["doppler_w_dca"],
        Lambda=cls.Lambda,
        angle_incidence=cls.data["inc_scatt_eqv"],
        angle_azimuth=90,  # the geometric doppler is interpreted as radial motion, so azimuth angle component must result in value of 1 (i.e. pi/2 or 90 deg)
        degrees=True,
    )

    cls.data["V_doppler_w_dca_pulse_rg"] = mean_along_azimuth(
        cls.data["V_doppler_w_dca"]
    )

    # convert computed avriables from slant to ground range resolution prior to spatial averaging
    grg_for_safekeeping = cls.data.grg
    vars_slant2ground = [
        "wb",
        "wb_pulse_rg",
        "dca",
        "dca_pulse_rg",
        "doppler_w_dca",
        "doppler_w_dca_pulse_rg",
        "V_wb",
        "V_dca",
        "V_dca_pulse_rg",
        "V_wb_pulse_rg",
        "V_doppler_w_dca",
    ]
    temp = cls.data[vars_slant2ground].interp(grg=cls.new_grg_pixel, method="linear")
    cls.data[vars_slant2ground] = temp.interp(grg=grg_for_safekeeping, method="linear")

    # perform averaging as prescribed in cls
    scenes_to_average = [
        "wb_pulse_rg",
        "V_wb_pulse_rg",
        "dca_pulse_rg",
        "doppler_w_dca_pulse_rg",
        "V_dca_pulse_rg",
        "V_doppler_w_dca_pulse_rg",
    ]
    scenes_averaged = [i + "_subscene" for i in scenes_to_average]

    # NOTE here we confusingly switch the definitions of fs_x and fs_y because the coordinates of loaded Doppler data follows a different order
    fs_x, fs_y = 1 / cls.stride, 1 / cls.grid_spacing
    data_lowpass = low_pass_filter_2D_dataset(
        cls.data[scenes_to_average],
        cutoff_frequency=1 / (cls.resolution_product),
        fs_x=fs_x,
        fs_y=fs_y,
        window=cls.product_averaging_window,
        fill_nans=True,
    )
    cls.data[scenes_averaged] = data_lowpass

    # a bit of reschuffling of coordinates
    cls.data = cls.data.transpose("az_idx", "grg", "slow_time").chunk("auto")
    cls.data[scenes_to_average + scenes_averaged] = cls.data[
        scenes_to_average + scenes_averaged
    ].persist()
    return
