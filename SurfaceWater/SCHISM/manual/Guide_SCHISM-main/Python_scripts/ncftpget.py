# Python script to download Ifremer spectral data for the coupling SCHISM/WWM

import subprocess
year = 2018
lats = [54.0]
lons = [2.5]
model = 'GLOB-30M'
fils = f'WW3-{model}-'
filf = '_spec.nc'

k=0
for lo in lons:
    lon = lo*10
    lon = str(int(lon)).zfill(4)
    print(lon)
    for la in lats:
        ifail=0
        lat = la*10
        lat = str(int(lat)).zfill(3)
        print(lat)
        fil = f'{fils}E{lon}S{lat}_{year}{filf}'
        print(fil)
        #ftp://ftp.ifremer.fr/ifremer/ww3/HINDCAST/GLOBMULTI_ECMWF_02/GLOB-30M/
        #2018/SPEC_SE/WW3-GLOB-30M-E0000S680_2018_spec.nc
        cmd =
        f'ncftpget ftp://ftp.ifremer.fr/ifremer/ww3/HINDCAST/GLOBMULTI_ECMWF_02/{model}/{year}/SPEC_SE/{fil}'
        print(cmd)
        ifail = subprocess.call(cmd,shell=True)
        if ifail != 0:
            print(f'{fil} not found')
        #if k==0:exit()
