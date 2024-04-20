# 前后处理Python脚本
## xwrf
![**xWRF**](https://xwrf.readthedocs.io/en/latest/tutorials/Overview.html) is a package designed to make the post-processing of WRF output data more pythonic. It’s aim is to smooth the rough edges around the unique, non CF-compliant WRF output data format and make the data accessible to utilities like dask and the wider Pangeo universe.
It is built as an Accessor on top of xarray, providing a very simple user interface.

## Pywinter
Python lib for Reading/Creating WRF-WPS intermediate file.

Pywinter is a Python3 library designed for handling files in WRF-WPS intermediate file format. Usually you don't need to deal with the intermediate files by your own because that is the function of ungrib.exe, but sometimes you don't have your meteorological data in GRIB format. Pywinter allows to read and create intermediate files by a simple way.

## wrf-python
A collection of diagnostic and interpolation routines for use with output from the Weather Research and Forecasting (WRF-ARW) Model.

This package provides over 30 diagnostic calculations, several interpolation routines, and utilities to help with plotting via cartopy, basemap, or PyNGL. The functionality is similar to what is provided by the NCL WRF package.

### wrf-python-notebooks
This repository contains Jupyter Notebooks that demonstrate basic uses of wrf-python to plot 2-D model fields, interpolate and plot model data on a specified vertical surface (here dubbed a 3-D plotting example), and interpolate and plot model data along a diagonal vertical cross-section.

These examples are adapted from the wrf-python Plotting Examples (https://wrf-python.readthedocs.io/en/latest/plot.html). The two map-based examples use cartopy for mapping. The vertical cross-section example has been substantially rewritten to use pressure rather than height as the vertical coordinate.

### wrf_python_tutorial
Student Workbook Repository for the wrf-python Tutorial

## pinterpy
Python function for vertical interpolation of WRF (Weather Research and Forecasting Model) output to pressure levels.
垂向插值到压强分层上

## namelist_plot
根据namelist.wps绘制图形

## WRF-ADIOS2-to-NetCDF4
This tool converts WRF outputted ADIOS2 files to NetCDF4 files for backwards compatibility.

# 自动设置和运行脚本
## PyWRF-Automation
Python automation script to download the Global Forecast System (GFS) data from NOMADS NOAA with spatial resolution 0.250 and execute Weather Research & Forecasting (WRF) model.

## pyWRFsetup
WRF SETUP & ANALYSIS  

## wrftools
A framework for running WRF simulations.

It is designed to be flexible and extendable. There are some tools out there which run WRF, but they are not easily modified. This is designed to provide  a framework which is easily customised and modified. 

## WRF-Run
This python script package automates the entire WRF process for use on cluster based computers. This is a fully self-contained script package that handles the tasks of obtaining the data, running the pre-processing executables, the WRF process, and forking the task to post-processing scripts for visualization.

## setup_wrf_epri
Python script to set up and run all the steps of WPS and WRF on NCAR's Cheyenne HPC, with an optional Python wrapper script to enable easier execution from crontab.