Create a cdsapi account to access ERA5 wind-field information

Downloaded and unzip at least three sequential Sentinel-1 IW GRD's in VV/VH pol as the edges are lost during beam pattern integration. 

Create a new environment and activate

`conda create -n ENV_NAME python==3.12`

`conda activate ENV_NAME`

Conda install GDAL (not enabled from pip)

`conda install GDAL`

Clone environment

`git clone git@github.com:owenodriscoll/DopSCA.git`

Navigate to installed directory and pip install other requirements

`pip install -e .`

For development and testing
`pip install -e .[test]`