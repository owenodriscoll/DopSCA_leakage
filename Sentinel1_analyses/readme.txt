Requires custom packages:
 - DRaMA 0.6: for radar processing functions (e.g. beam patterns) 
    https://gitlab.tudelft.nl/drama/drama
 - S1SEA 23.4.12: For Converting GRD's to NRCS fields
    https://gitlab.tudelft.nl/plopezdekker/s1sea
 - Xarray-Sentinel 0.9.5: called by S1SEA. In line 139 of xarray_sentinel/sentinel1.py the raise error is commented out
    https://github.com/bopen/xarray-sentinel 
    

Data downloaded from:
   https://dataspace.copernicus.eu/browser/