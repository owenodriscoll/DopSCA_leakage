import glob
import numpy as np
import xarray as xr
import argparse
from leakage.velocity_leakage import S1DopplerLeakage

func_rmse = lambda x, rounding=3:  np.round(np.sqrt(np.mean(x**2)).values*1, rounding)
func_rmse_xr = lambda x, rounding=3:  np.round(np.sqrt(np.mean(x**2)), rounding)

scenarios = [
'Azores_20201127',
'Carrib_20231104',
'Hawaii_20201106',
'Iceland_20231107',
'Scotland_20231109',
]

version = 'v1'

antenna_length_multipliers = 10**(np.log10(2)*np.arange(0, 4))
N = len(antenna_length_multipliers)

if __name__ == "__main__":
        
    # Create an argument parser
    parser = argparse.ArgumentParser()

    # Define command-line arguments
    parser.add_argument("data_dir", type=str, help="directory containing Sentinel-1 GRD data folders")
    parser.add_argument("data_dir_era5", type=str, help= "directory where ERA5 data will be stored/loaded")
    parser.add_argument("save_dir", type=str, help= "directory to save results")

    # Parse the arguments
    args = parser.parse_args()

    data_dir = args.data_dir
    data_dir_era5 = args.data_dir_era5
    save_dir = args.save_dir

    for scenario in scenarios:
        print(scenario, flush=True)

        scenario_name = scenario[:scenario.find('_')].lower()
        save_dir_file = save_dir + f"{scenario_name}_design_param_{version}.nc"

        results_w_speck = []
        for i, antenna_length_multiplier in enumerate(antenna_length_multipliers):

            files = glob.glob(f"{data_dir+scenario}/*.SAFE")

            test = S1DopplerLeakage(
                filename=files,
                f0 = 5_400_000_000,
                z0 = 823_000,
                era5_directory=data_dir_era5,
                resolution_product=50_000,
                az_footprint_cutoff=80_000,
                vx_sat=6_800,
                PRF=4,
                grid_spacing=75,
                antenna_length=2.87 * antenna_length_multiplier,
                antenna_height=0.32,
                beam_pattern= 'phased_array',
                antenna_elements=4,
                antenna_weighting=0.75,
                swath_start_incidence_angle_scat=35,
                boresight_elevation_angle_scat=40,
                random_state = 42, # NOTE random state kept the same because we want to analyse exactly the same scene
                fill_nan_limit = 1,
                product_averaging_window='hann',
                _speckle_noise= True # NOTE first we keep speckle
                )
            test.apply()
            results_w_speck.append(test)


        results_w_o_speck = []
        for i, antenna_length_multiplier in enumerate(antenna_length_multipliers):

            files = glob.glob(f"{data_dir+scenario}/*.SAFE")

            test = S1DopplerLeakage(
                filename=files,
                f0 = 5_400_000_000,
                z0 = 823_000,
                era5_directory=data_dir_era5,
                resolution_product=50_000,
                az_footprint_cutoff=80_000,
                vx_sat=6_800,
                PRF=4,
                grid_spacing=75,
                antenna_length=2.87 * antenna_length_multiplier,
                antenna_height=0.32,
                beam_pattern= 'phased_array',
                antenna_elements=4,
                antenna_weighting=0.75,
                swath_start_incidence_angle_scat=35,
                boresight_elevation_angle_scat=40,
                random_state = 42, # NOTE random state kept the same because we want to analyse exactly the same scene
                fill_nan_limit = 1,
                product_averaging_window='hann',
                _speckle_noise= False # NOTE now we remove speckle
                )
            test.apply()
            results_w_o_speck.append(test)

        samples = results_w_speck + results_w_o_speck 

        residuals = [result.data.V_leakage_pulse_rg - result.data.V_leakage_pulse_rg_inverted for result in samples]
        backscatters = [result.data.nrcs_scat for result in samples]
        noise = [result.data.V_sigma - result.data.V_leakage_pulse_rg_inverted for result in samples]

        ds_temp = xr.Dataset()
        ds_temp['residual'] = xr.concat(residuals, dim = 'la')
        ds_temp['nrcs'] = xr.concat(backscatters, dim = 'la')
        ds_temp['noise'] = xr.concat(noise, dim = 'la')

        ds_res_speck = ds_temp.sel(la = range(0,N)).assign_coords(speckle=('speckle', [True]))
        ds_res_no_speck = ds_temp.sel(la = range(N,2*N)).assign_coords(speckle=('speckle', [False]))
        ds_res = xr.concat([ds_res_speck, ds_res_no_speck], dim = 'speckle')
        ds_res = ds_res.assign_coords(la = ('la', 2.87 * antenna_length_multipliers))

        ds_res.to_netcdf(save_dir_file)

        # clear memory 
        del results_w_speck
        del results_w_o_speck
        del samples
        del residuals
        del backscatters
        del noise
        del ds_temp
        del ds_res_speck
        del ds_res_no_speck
        del ds_res
