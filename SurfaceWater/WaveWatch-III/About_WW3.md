# About WW3     

https://github.com/NOAA-EMC/WW3/wiki/About-WW3

WAVEWATCH III is a community wave modeling framework that includes the latest scientific advancements in the field of wind-wave modeling and dynamics.

The core of the framework consists of the WAVEWATCH III? third-generation wave model (WAVE-height, WATer depth and Current Hindcasting), developed at NOAA/NCEP in the spirit of the WAM model (WAMDIG 1988, Komen et al, 1994), which evolved from WAVEWATCH (Delft: Tolman 1989, 1991a), and WAVEWATCH II (NASA Goddard: Tolman, 1992). WAVEWATCH III? differs from its predecessors in many important points such as governing equations, model structure, numerical methods and physical parameterizations.

WAVEWATCH III? solves the random phase spectral action density balance equation for wavenumber-direction spectra. The implicit assumption of this equation is that properties of medium (water depth and current) as well as the wave field itself vary on time and space scales that are much larger than the variation scales of a single wave. The model includes options for shallow-water (surf zone) applications, as well as wetting and drying of grid points. Propagation of a wave spectrum can be solved using regular (rectilinear or curvilinear) and unstructured (triangular) grids.

For more detailed information:

# Physics

# Numerics

# Output Options

# WW3 Manual References (As seen in the manual)

# MetEd WAVEWATCH III Course (requires login)


# The WAVEWATCH III Development Group (As seen in the manual)

# WAVEWATCH III Public releases

Here are the previous official model releases followed by their key added/modified features and capabilities and code structure changes. The present version (v7.xx) is actively under development and available to the public based on the previous official model release (version 6.07).

## v 2.22

the first official public release

single grid model,
underpinning of the numerical scheme set,
Modular Fortran 90 with MPI and OPENMP formulation,
Tolman Chalikov physics and WAM cycle 3 physics,
DIA and EXACT 每 NL,
Regular grids (lat每lon spherical or rectilinear),
Finite difference in spatial and spectral domains

# v 3.14

The second release in 2007

Model expanded to two每way nested mosaic system with multiple grids,
Linear growth term,
Depth limited wave breaking,
Numerical schemes for individual grids unchanged

## v 4.18

The third public release (March 19 2014)

Code development now by an international team of developers (svn),
Multiple grids formulation expanded to include curvilinear grids, **unstructured grids** and SMC grids,
Ardhuin et al physics package,
Babanin et al physics package,
Second-order spatial propagation scheme,
Iceberg blocking
Multiple wave 每 mud and wave 每 ice interaction packages,
Netcdf I/O added,
Triads interactions,
Expanded field of outputs (primarily for coupling),
SHOWEX bottom friction source term,
Grid splitting auxiliary code (for hyper scaling)

## v 5.16

The fourth public release (October 31, 2016)

Sea-ice scattering and creep dissipation added Optimization and updates to IC3 and IC2,
Capability to handle cpp macros,
Updates to SMC grid time, OpenMP and hybrid OpenMP/MPI,
Tripole grid functionality,
Updates/optimization to various source terms (IC2, IC3, ST4, ST6),
Coupler capabilities for NCEP coupler and OASIS coupler,
Namelist format option for multi grid input file (ww3_multi.nml),
Sea-state dependent stress calculations,
TSA nonlinear wave-wave interaction,
Calculation of space-time extremes

## v 6.07

The fifth public release (April 2019)

Enhanced Stokes drift computation options,
New module for ESMF interface,
Capability to update restart file＊s total energy based on independent significant wave height analysis,
Domain decomposition for unstructured implicit schemes using PDLIB,
Updates the namelist options for the following programs: ww3 ounf, ww3 ounp, ww3 trnc, ww3 bounc, and ww3 shel,
Adding IC5 as a sea ice source term option,
Other additions include updates on source term parameterizations such IC2, IS2, ST4, REF1,
Optional instrumentation to code for profiling of memory use

## WAVEWATCH III current development

7.01 Reading wind from restart: Adds option for reading/writing wind in restart file via WRST switch.
7.02 Interpolate internally winds to curvilinear subgrids: Adds capability to interpolate inputs to curvilinear subgrid in multi-grid.
7.03 Second restart file stream for checkpointing: Adds the second stream of restart file writes.
7.04 Change version reporting in codes to date string: Changes version reporting in codes to date strings.
7.05 : IS2 update default parameterization.
7.06 : ST4 inclusion of Romero Dissipation.
7.07 : Tide update and MPI implementation.
7.08: CFL on boundary condition for unstructured mesh: Adds optional opt-out for CFL on boundary condition on unstructured mesh.
7.09:Add options for wind-sea and swell component data assimilation to ww3_uprstr.
7.10: Extend parameter set for OASIS-NEMO coupled model systems.
7.11: Specify output boundary conditions to rotated grids: Adds method to output boundary conditions to a rotated lat/lon grid.
7.12: Improve NetCDF gridded output for CF compliance.
7.13: Major code clean up including removing support for NetCDFv3, making NC4 the default, making F90 the default (removing the DUM option), removal of compiler directive.


# Physics

The governing equations of WAVEWATCH III? include refraction and straining of the wave field due to temporal and spatial variations of the mean water depth and of the mean current (tides, surges etc.), when applicable.
Parameterizations of physical processes (source terms) include wave growth and decay due to the actions of wind, nonlinear resonant interactions, dissipation (`whitecapping'), bottom friction, surf-breaking (i.e., depth-induced breaking) and scattering due to wave-bottom interactions. The model is prepared for triad interactions and is prepared for other, as of yet undefined source terms, but the latter have not been implemented yet. WAM cycle 4, bottom scattering, surf zone physics and wetting/drying new in model version 3.14.
Wave propagation is considered to be linear. Relevant nonlinear effects such as resonant interactions are, therefore, included in the source terms (physics).
The model includes several alleviation methods for the Garden Sprinkler Effect (Booij and Holthuijsen, 1987, Tolman, 2002c).
The model includes sub-grid representation of unresolved islands (Tolman 2002e). A software package based on Matlab&reg has been developed to automate generation of grids including obstructions due to unresolved islands (Chawla and Tolman, 2007, 2008). Grid generation package first distributed with model version 3.14 through model distribution web page (see below).
The model includes options for choosing various term packages, some intended for operations, others for research. The source term packages are selected at the compile level.
The model includes dynamically updated ice coverage.
The model is prepared for data assimilation, but no data assimilation package is provided with the present distribution of the source code.
Spectral partitioning is now available for post-processing of point output, or for the entire wave model grid using the Vincent and Soille (1991) algorithm (Hanson and Jenssen, 2004; Hanson et al , 2006, 2009). New in model version 3.14
