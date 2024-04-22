import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import LinearNDInterpolator
from dataclasses import dataclass
from typing import Callable, Union, Sequence
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
        lafs = []

        # extract data in each file
        for file in self.filenames:
            data = xr.open_dataset(file)
            longitudes.append(data.lon.values.ravel())
            latitudes.append(data.lat.values.ravel())
            dcas.append(data.doppler_centroid_anomaly.values.ravel())
            lafs.append(data.land_area_fraction.values.ravel())

        # combine data from all files into single dataframe
        np.concatenate(dcas, axis = 0)
        df = pd.DataFrame(
            data = {'lon': np.concatenate(longitudes, axis = 0),
                    'lat': np.concatenate(latitudes, axis = 0),
                    'dca': np.concatenate(dcas, axis = 0),
                    'laf': np.concatenate(lafs, axis = 0)})
        
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
        interp_dca = LinearNDInterpolator(list(zip(self.df.lon.values, self.df.lat.values)), self.df.dca.values)
        interp_laf = LinearNDInterpolator(list(zip(self.df.lon.values, self.df.lat.values)), self.df.laf.values)

        # apply interpolation for specified points
        dca_interp = interp_dca(self.longitudes, self.latitudes)
        laf_interp = interp_laf(self.longitudes, self.latitudes)

        # filter pointswhere land area fraction suggests land contamination
        self.dca_interp = np.where(laf_interp > 0, np.nan, dca_interp)

    def add_dca(self):
        self.load_dca()
        self.interpolate_dca()
        return self.dca_interp
    



