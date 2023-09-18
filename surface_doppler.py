import numpy as np
from matplotlib import pyplot as plt
from stereoid.oceans.forward_models.backscatter import backscatter_Kudry2023
from stereoid.oceans.forward_models.Doppler import DopRIM2023_DP
from stereoid.oceans.waves.wave_spectra import Kudry_spec

#%%
# wavelengths and wave numbers
g=9.81
n_k = 100  # number of frequencies single side (total 2*n_k - 1)
lambda_min = 0.01  # minimum wave length
lambda_max = 1000  # maximum wave length
k_min = 2 * np.pi / lambda_max  # minimum wave number
k_max = 2 * np.pi / lambda_min # should at least pass the Bragg wave number
k_x = np.reshape(10**np.linspace(np.log10(k_min),np.log10(k_max),n_k),(1,n_k))
k_x = np.append( np.append( -np.flip( k_x ), 0 ), k_x )  # double sided spectrum
dk=np.gradient(k_x,1)
k_x = np.dot( np.ones( (n_k * 2 + 1, 1) ), k_x.reshape( 1, n_k * 2 + 1 ) )  # two dimensional
k_y = np.transpose( k_x )
k = np.sqrt( k_x ** 2 + k_y ** 2 )
omega=np.where(k > 0, np.sqrt(g*k), 0)
phi = np.arctan2( k_y, k_x )  # 0 is cross-track direction waves, 90 along-track
dks = np.outer( dk, dk )  # patch size

u_10 = 7 # wind speed
fetch = 10e3 # fetch length for the non-equilibrium part of the spectrum
# phi_w = np.deg2rad(45) # wind direction
# dks = (np.pi*2/100)**2 # two-dimensional grid resolution
S=0 # Cartesian long-wave spectrum (non-equilibrium waves), optional: if set, set fetch=0

#%%

class temporary_class:
    def __init__(self, inc_m):
        self.inc_m = inc_m
        pass

def Doppler_inc(inc, phi_w):

    obs_geo = temporary_class(inc_m=inc)

    B,_,_,_=Kudry_spec(k_x, k_y, u_10, fetch, phi_w, dks)

    S=np.where(k>0,B*k**-4,0)

    c_sp_bar, c_wb_bar, c_br_bar, c_sp, c_wb, c_br_vv, c_br_hh=DopRIM2023_DP(S, k_x, k_y, dks, obs_geo.inc_m, u_10, phi_w, k_r=0)

    return c_sp_bar, c_wb_bar, c_br_bar, c_sp, c_wb, c_br_vv, c_br_hh