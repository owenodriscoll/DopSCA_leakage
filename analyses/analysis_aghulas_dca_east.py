import glob
import numpy as np
import xarray as xr
import argparse
from leakage.velocity_leakage import S1DopplerLeakage, add_dca_to_leakage_class

func_rmse = lambda x, rounding=3:  np.round(np.sqrt(np.mean(x**2)).values*1, rounding)
func_rmse_xr = lambda x, rounding=3:  np.round(np.sqrt(np.mean(x**2)), rounding)


# data_dir = "/Users/opodriscoll/Documents/Data/Sentinel1/IW/"
# data_dir_dca = "/Users/opodriscoll/Documents/Data/Sentinel1/DCA/"
# save_dir = '../../data/leakage/temp/aghulas_east_v3.nc'


if __name__ == "__main__":

    # Create an argument parser
    parser = argparse.ArgumentParser()

    # Define command-line arguments
    parser.add_argument("data_dir", type=str, help="directory containing Sentinel-1 GRD data folders")
    parser.add_argument("data_dir_dca", type=int, help= "directory containing corresponding DCA data folders")
    parser.add_argument("save_dir", type=int, help= "directory to save results")

    # Parse the arguments
    args = parser.parse_args()

    data_dir = args.data_dir
    data_dir_dca = args.data_dir_dca
    save_dir = args.save_dir

    scenarios = [
        'Aghulas_20200215',
        'Aghulas_20200227',
        'Aghulas_20200310',
        'Aghulas_20200322',
        'Aghulas_20200403',
        'Aghulas_20200415',
        'Aghulas_20200427',
    ]

    results = []
    for i, scenario in enumerate(scenarios):
        files = glob.glob(f"{data_dir+scenario}/*.SAFE")

        test = S1DopplerLeakage(
            filename=files,
            f0 = 5_400_000_000,
            z0 = 823_000,
            era5_directory='../../data/leakage/era5_winds/',
            resolution_product=50_000,
            az_footprint_cutoff=80_000,
            vx_sat=6_800,
            PRF=4,
            grid_spacing=75,
            antenna_length=2.87,
            antenna_height=0.32,
            beam_pattern= 'phased_array', #'phased_array', sinc
            antenna_elements=4,
            antenna_weighting=0.75,
            swath_start_incidence_angle_scat=35,
            boresight_elevation_angle_scat=40,
            random_state = 42 + i, # NOTE random state changes per scene
            fill_nan_limit = 1,
            product_averaging_window='hann',
            )
        test.apply()

        
        files_dca = glob.glob(f"{data_dir_dca+scenario}/*.nc")
        
        add_dca_to_leakage_class(test, files_dca=files_dca)
        results.append(test)



    samples = results

    residuals = [result.data.V_leakage_pulse_rg_subscene - result.data.V_leakage_pulse_rg_subscene_inverted for result in samples]
    backscatters = [result.data.nrcs_scat for result in samples]
    signals = [result.data.V_dca_pulse_rg_subscene for result in samples]
    currents =  [result.data.V_dca_pulse_rg_subscene - result.data.V_wb_pulse_rg_subscene for result in samples]
    noise = [result.data.V_sigma_subscene - result.data.V_leakage_pulse_rg_subscene_inverted for result in samples]

    ds_res = xr.Dataset()
    ds_res['residual'] = xr.concat(residuals, dim = 'time')
    ds_res['currents'] = xr.concat(currents, dim = 'time')
    ds_res['nrcs'] = xr.concat(backscatters, dim = 'time')
    ds_res['noise'] = xr.concat(noise, dim = 'time')

    ds_res.to_netcdf(save_dir)