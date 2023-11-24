#%%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from stereoid.oceans.GMF.cmod5n import cmod5n_inverse
from stereoid.oceans.GMF.cdop import cdop

#%%
path = "/Volumes/Extreme SSD/copy_laptop_ifremer_25_04_2023/sentinel/wv_test_images_cells/"
scene = "SENTINEL1_DS__home_datawork-cersat-public_cache_project_mpc-sentinel1_data_esa_sentinel-1a_L1_WV_S1A_WV_SLC__1S_2019_246_S1A_WV_SLC__1SSV_20190903T221250_20190903T221830_028863_03455C_71F3_SAFE_WV_023.nc"
t = xr.open_dataset(path + scene)
# %%
phi = 90
pol = "VV"

U = cmod5n_inverse(sigma0_obs=t.sigma0.values, incidence=t.incidence.values, phi = phi)
dop = cdop(u10 = U,  phi=phi, inc=t.incidence, pol=pol)

t['windfield'] = (('atrack', 'xtrack'), U)
t['dop'] = (('atrack', 'xtrack'), dop)

# %%

t.dop.plot.hist()
