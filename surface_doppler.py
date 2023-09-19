import numpy as np
from stereoid.oceans.forward_models.Doppler import DopRIM2023_DP
from stereoid.oceans.forward_models.backscatter import backscatter_Kudry2023
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

# phi_w = np.deg2rad(45) # wind direction
# dks = (np.pi*2/100)**2 # two-dimensional grid resolution
S=0 # Cartesian long-wave spectrum (non-equilibrium waves), optional: if set, set fetch=0

#%%

class temporary_class:
    def __init__(self, inc_m):
        self.inc_m = inc_m
        pass

def Doppler_inc(inc, phi_w, k_r, u_10, fetch):
    """
    Input
    -----
    inc: incidence angle, rad
    phi_w: winddirection w.r.t. radar,rad
    k_r: radar wave number, rad /m
    u_10: wind speed at 10m eleveation, m/s 
    fetch: fetch length for the non-equilibrium part of the spectrum, m
    
    Returns
    -------

    """

    obs_geo = temporary_class(inc_m=inc)

    # calculate curvature spectrum
    B,_,_,_=Kudry_spec(k_x, k_y, u_10, fetch, phi_w, dks)

    # convert to elevation spectrum
    S=np.where(k>0,B*k**-4,0)

    # calculate backscatter TODO check in what system phi_w is expected
    sigma_los, dsigmadth, q = backscatter_Kudry2023(S, k_x, k_y, dks, phi_w=phi_w, theta=obs_geo.inc_m, u_10=u_10, k_r=k_r,
                                                    degrees=False)
    # sigma_los = np.array([sigma_sp, sigma_br_VV, sigma_br_HH, sigma_wb])
    # calculate backscatter ratios for VV polarization assuming specular is negligeable
    rat=[0, 
        sigma_los[-1]/np.sum(sigma_los[1] + sigma_los[-1]),
        sigma_los[1]/np.sum(sigma_los[1] + sigma_los[-1])]

    # calculate velocity of scattering facets
    c_sp_bar, c_wb_bar, c_br_bar, c_sp, c_wb, c_br_vv, c_br_hh=DopRIM2023_DP(S, k_x, k_y, dks, obs_geo.inc_m, u_10, phi_w, k_r=k_r)

    # calculate surface velocity doppler (from forward_models.Doppler.DopRIM)
    V = rat[0] * (c_sp_bar + c_sp) + \
        rat[1] * (c_br_bar + c_br_vv) + \
        rat[2] * (c_wb_bar + c_wb) 
    

    return V#, c_sp_bar, c_wb_bar, c_br_bar, c_sp, c_wb, c_br_vv, c_br_hh