#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:42:35 2021

@author: christelle
"""
from netCDF4 import Dataset,num2date
from datetime import datetime
import numpy as np
import xarray as xr
#import pandas as pd

# get lon/lat of open boundaries nodes from hgrid.ll
f=open('hgrid_principal.ll','r')
data=f.readlines()

num_nodes_and_elements = data[1]
num_nodes = num_nodes_and_elements.split(" ")[1]
num_elements= num_nodes_and_elements.split(" ")[0]

nodes_start = 2
node_lines = data[nodes_start:nodes_start+int(num_nodes)]

elements_start = nodes_start + int(num_nodes) 
element_lines = data[elements_start:elements_start + int(num_elements)]

metadata1_start = elements_start + int(num_elements) 
metadata1_line = data[metadata1_start:metadata1_start + 3]
metadata_obnb=data[metadata1_start+1]
numnodes_obnb=metadata_obnb.split(" ")[0] #to have total number of nodes in open bnd
metadata_obnb1=data[metadata1_start+2]
numnodes_obnb1=metadata_obnb1.split(" ")[0] 

obnodes1_start =metadata1_start+ 3
obnnode1_lines=data[obnodes1_start:obnodes1_start+int(numnodes_obnb1)]

metadata2_start = obnodes1_start + int(numnodes_obnb1)
metadata_obnb2=data[metadata2_start]
numnodes_obnb2=metadata_obnb2.split(" ")[0]

#only one open bnd in this case
# obnodes2_start =metadata2_start+ 1
# obnnode2_lines=data[obnodes2_start:obnodes2_start+int(numnodes_obnb2)]


openbnd_lines=obnnode1_lines#+obnnode2_lines #all the lines for all nodes in open bnd

#get the coordinates corresponding to the open bnd nodes
bnd_coord=[]
nodeLon=[]
nodeLat=[]
nodeLont=[]
nodeLatt=[]
for i in range(len(openbnd_lines)):
        node_index = int(openbnd_lines[i])
        node= node_lines[node_index-1]
        #node_num=node[0:5]
        node_lont=node.split(" ")[5]
        node_latt=node.split(" ")[10]
        #append lon, lat
        nodeLont.append(node_lont)
        nodeLatt.append(node_latt)

bnd_coord= np.column_stack([nodeLont, nodeLatt])

nOpenBnbNodes=int(numnodes_obnb)

##create new netcdf elev2D.th.nc
filename = 'elev2D_RES.th.nc'
nc_elev = Dataset(filename, 'w', format='NETCDF4')
#nc_elev = Dataset(filename, 'w', format='NETCDF3_CLASSIC')
nc_elev.setncatts({"Conventions": "CF-1.0"})
nc_elev.createDimension('time', None)
nc_elev.createDimension('nComponents', 1) 
nc_elev.createDimension('nLevels', 1) 
nc_elev.createDimension('nOpenBndNodes', len(bnd_coord)) #nOpenBnbNodes
nc_elev.createDimension('one', 1) 

one=1

time_var = nc_elev.createVariable('time', np.double, ('time',))
time_var.units = 's'
time_var.long_name = 'simulation time in seconds'


timestep_var = nc_elev.createVariable('time_step', np.float32, 'one')
timestep_var.units = 's'
timestep_var.long_name = 'time step in seconds'
timestep_var[:]= 86400


elev_var = nc_elev.createVariable('time_series',np.float32,('time','nOpenBndNodes',
                              'nLevels', 'nComponents'),)
elev_var.long_name = 'elev_nontidal'
elev_var.units='m'



################Fill the new nc files with Mercator data ( ex 1month  01/07/2018)
d  = Dataset('merc-all-201012-1bis.nc') #modified from original version with 
#cdo to have time in seconds
tvar = 'time'
t = d[tvar][:]
t_unit = d.variables[tvar].units
#print(t_unit)
tvals = num2date(t,units = t_unit)
str_t = [i.strftime("%Y%m%d %H%M") for i in tvals] # to display dates as string
datetime_t = [datetime.strptime(i,"%Y%m%d %H%M") for i in str_t]
time_var[:]= t


######### need to find the values for lon/lat of openbnb =bnd_coord

nc=xr.open_dataset('merc-all-201012-1bis.nc') 
latitude=nc['latitude']
longitude=nc['longitude']
uo=nc['uo']
vo=nc['vo']
zos=nc.zos
t,nr,nco=zos.shape

#identify the positions for each node on the boundary
long_mesh = np.meshgrid(longitude, latitude, indexing='xy')[0]
lat_mesh = np.meshgrid(longitude, latitude, indexing='xy')[1]
positions = [near2d_unstruct(long_mesh.flatten(), lat_mesh.flatten(), 
                         lons,lats)for lons,lats in zip(bnd_coordF[:,0], bnd_coordF[:,1])]
             

i_columns = [i % nco for i in positions]
i_rows = [math.floor(i / nco) for i in positions]

elevation=zos.to_dataframe().fillna(method='ffill').to_xarray().zos.values

#remove the mean for Mercator files 
elevationM=np.nanmean(elevation, axis=0)
elevationRes=elevation-elevationM

elevation_bnd=elevationRes[:, i_rows, i_columns]

###### Fill netcdf

elev_var[:,:,:,:] = elevation_bnd


d.close()
nc_elev.close()



 
