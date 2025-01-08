![Continuous Integration build in GitHub Actions](https://github.com/owenodriscoll/DopSCA/actions/workflows/main.yaml/badge.svg?branch=main)

## Data preparation
1. Create an ECMWF account to access the ERA5 CDS API (for setup go to [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu/how-to-api)). This is needed for wind-field information.

2. Sentinel-1 data can be downloaded at [browser.dataspace.copernicus.eu](https://browser.dataspace.copernicus.eu/). Recommended to download and unzip at least three sequential Sentinel-1 IW GRD's in VV+VH pol as the edges are lost during beam-pattern integration. 

## Environment preparation
Create a new environment and activate

```bash
conda create -n ENV_NAME python==3.12
conda activate ENV_NAME
```

Conda install GDAL (not enabled from pip)
```bash
conda install GDAL
```

Clone environment
```bash
git clone git@github.com:owenodriscoll/DopSCA.git
```

Navigate to installed directory and pip install other requirements
```bash
pip install -e .
```

For development and testing
```bash
pip install -e .[test]
```

## Minimal example
For a minimal example near the Azores, see [the example](https://github.com/owenodriscoll/DopSCA/blob/main/analysis/example.ipynb) in the analyses folder.
