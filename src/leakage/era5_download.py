import sys
from .misc import era5_wind_monthly

"""
run file using command line as 

python era5_download.py {year} {month} {directory}

where "year" and "month" are obligatory and input types should preferably be strings

"""

if __name__ == '__main__':
    
    year = sys.argv[1]
    month =  sys.argv[2]

    try:
        directory = sys.argv[3]
    except:
        directory = ''

    era5_wind_monthly(
        year = year,
        month = month,
        directory = directory,
    )
