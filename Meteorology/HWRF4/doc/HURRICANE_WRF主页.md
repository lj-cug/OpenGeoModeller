# HURRICANE WRF
https://dtcenter.org/community-code/hurricane-wrf-hwrf/download

Welcome to the Hurricane Weather Research and Forecasting (HWRF) users page. HWRF is designed to serve both operational forecasting and atmospheric research needs. It features the NMM dynamic core, multiple physical parameterizations, a variational data assimilation system, ability to couple with an ocean model, and a software architecture allowing for computational parallelism and system extensibility. From this website, users can obtain codes, datasets, and information for running HWRF.

The Developmental Testbed Center supports the use of all components of HWRF to the community, including the Weather Research and Forecasting (WRF) atmospheric model with its Preprocessing System (WPS), various vortex initialization procedures, the Princeton Ocean Model for Tropical Cyclones (MPIPOM-TC), the Gridpoint Statistical Interpolation (GSI) three-dimensional ensemble-variational data assimilation system, the NOAA National Centers for Environmental Prediction (NCEP) coupler, the NOAA Geophysical Fluid Dynamics Laboratory (GFDL) Vortex Tracker, and various post-processing and products utilities.

The effort to develop HWRF has been a collaborative partnership, principally between NOAA (NCEP, Atlantic Oceanographic and Meteorological Laboratory (AOML), and GFDL) and the University of Rhode Island.

## HWRF OVERVIEW
https://dtcenter.org/community-code/hurricane-wrf-hwrf/system-architecture

### DESCRIPTION
The Hurricane Weather Research and Forecast system (HWRF) has been operational at NCEP since 2007. This advanced hurricane prediction system was developed primarily at the NWS/NCEP's Environmental Modeling Center (EMC), in collaboration with NOAA GFDL, NOAA AOML, and the University of Rhode Island, taking advantage of the WRF model infrastructure developed at NCAR. HWRF is a state-of-the-art hurricane model that has the capability to address the intensity, structure, and rainfall forecast problems. It is currently used to provide real-time numerical guidance in all oceanic basins.

The HWRF model is a primitive equation, non-hydrostatic coupled atmosphere-ocean model. The model uses the Non-hydrostatic Mesoscale Model (NMM) dynamic core, including its rotated latitude-longitude projection with E-grid staggering. The model has an outer domain spanning about 77.2¡ã x 77.2¡ã, with two telescopic two-way interactive nest domains. The intermediate and inner nests, which move with the storm, cover areas of approximately 17.8¡ã x 17.8¡ã and 5.9¡ã x 5.9¡ã respectively. The stationary parent domain has a grid spacing of 0.099¡ã (about 13.5 km), the intermediate nest 0.033¡ã (about 4.5 km), and the inner nest domain 0.011¡ã (about 1.5 km).

The model physics includes a Scale Aware Simplified Arakawa-Schubert scheme for cumulus parameterization and the Ferrier-Aligo cloud microphysics package for explicit condensation. The Global Forecast System (GFS) Hybrid-Eddy Diffusivity Mass Flux planetary boundary layer parameterization is used. An upgraded GFDL model surface layer scheme, along with Noah land surface model, is used to compute surface fluxes. Radiation physics are evaluated by the RRTMG scheme.

The NCEP GFS analysis, the NCEP Global Data Assimilation System (GDAS) 6-h forecasts, and the storm message provided by the National Hurricane Center (NHC) are used to generate initial conditions for the hurricane model. The HWRF system does not use a bogus vortex. Instead, it contains a forecast/analysis cycle in which a 6-h HWRF forecast from the previous cycle is used in a vortex relocation procedure, after being adjusted for position, structure and intensity using the NHC storm message. After the vortex relocation, the initial conditions are further refined through the use of a 3D hybrid ensemble-variational data assimilation system. The GFS forecasted fields every 6 hours are used to provide lateral boundary conditions during each forecast.

HWRF runs coupled by default with the Message Passing Interface Princeton Ocean Model (MPIPOM-TC) for all oceanic basins in the northern hemisphere. The model can also be run in coupled mode in any ocean basin worldwide. The MPIPOM-TC is initialized by a diagnostic and prognostic spinup of the ocean circulations using available climatological ocean data in combination with real-time sea surface temperature and sea surface height data. In the Atlantic basin, realistic representations of the structure and positions of the Loop Current, Gulf Stream, and warm- and cold-core eddies are incorporated in the spin up phase. In the North Atlantic, Western North Pacific, North Indian Ocean basins the ocean is initialized with Generalized Digital Environmental Model (GDEM) climatology whereas in the Eastern Pacific and Central Pacific basins the initialization is done using the Real-Time Ocean Forecast System (RTOFS) data by default.

At this time, the Developmental Testbed Center (DTC) supports the following components of HWRF:
```
WRF v4.0a (contains the 2018 operational capability)
Global modeling preprocessing with prep_hybrid and the WRF Preprocessing System (WPS)
Vortex Initialization
Gridpoint Statistical Interpolation (GSI) data assimilation system
NCEP Coupler
MPIPOM-TC and its initialization
HWRF Post-processing
GFDL Vortex Tracker
Running scripts
Idealized capability
```
For more information, please refer to the operational HWRF page.

## HWRF IDEALIZED OVERVIEW
### DESCRIPTION OF HWRF IDEALIZED
The Hurricane Weather Research and Forecast system (HWRF) Idealized capability is distributed with version 4.0a. The HWRF model has two classes of initialization: a) ideal vortex and b) real data.

The HWRF Idealized capability includes a landfall option. Currently, only the GFDL Slab land surface physics option is supported for landfall capability. A namelist file for land surface configuration introduces a switch for landfalling capability, specifies the type of land surface, and initial land-surface temperature to be used over land. The default configuration introduces a homogeneous land surface but can be modified to account for heterogeneity through the source code. The direction of land motion can also be chosen. Two options are available ¨C West to East or East to West.

At the time of HWRF v4.0a release, the landfall capability is broken. If a patch becomes available, it will be released.

The idealized simulation requires 00-h and 120-h GFS GRIB2 forecasts, along with a sounding, specifications of the initial vortex, and a namelist. The idealized simulation is performed on the operational triple nested domain configuration, with grid spacings of 13.5-, 4.5- and 1.5-km. All the operational atmospheric physics, as well as the supported experimental physics options in HWRF, can be utilized in the idealized framework. The post processing can be done using the Unified Post-Processor.

Always look for Known Issues before setting up any experiment.

For the convenience of the user community, sample GFS GRIB2 files for running the idealized case are provided.

For more information, please refer to the HWRF User's Guide.
```
HWRF Documents - HWRF 4.0a Release (2018)
HWRF Users' Guide v4.0a (PDF)
HWRF Scientific Documentation - November 2018 (PDF)
WRF-NMM V4 User's Guide (PDF)
Tutorial Presentations
```

# Publications
```
Bao, S., L. Bernardet, G. Thompson, E. Kalina, K. Newman, and M. Biswas, 2020: Impact of the Hydrometeor Vertical Advection Method on HWRF¡¯s Simulated Hurricane Structure. Wea. Forecasting, 35, 723¨C737, https://doi.org/10.1175/WAF-D-19-0006.1

Biswas, M.K., J.A. Zhang, E. Grell, E. Kalina, K. Newman, L. Bernardet, L. Carson, J. Frimel, and G. Grell, 2020: Evaluation of the Grell-Freitas convective scheme in the Hurricane Weather Research and Forecasting (HWRF) model. Wea. Forecasting, 0, https://doi.org/10.1175/WAF-D-19-0124.1.

Biswas M. K., L. Bernardet, S. Abarca, I. Ginis, E. Grell, E. Kalina, Y. Kwon, B. Liu, Q. Liu, T. Marchok, A. Mehra, K. Newman, D. Sheinin, J. Sippel, S. Subramanian, V. Tallapragada, B. Thomas, M. Tong, S. Trahan, W. Wang, R. Yablonsky, X. Zhang, and Z. Zhang, 2017: Hurricane Weather Research and Forecasting (HWRF) Model: 2017 Scientific Documentation, NCAR Technical Note NCAR/TN-544+STR, doi: 10.5065/D6MK6BPR

Bernardet, L. Tallapragada, V., Bao, S., Trahan, S., Kwon, Y., Liu, Q., Tong, M., Biswas, M., Brown, T., Stark, D., Carson, L., Yablonsky, R., Uhlhorn, E., Gopalakrishnan, S., Zhang, X., Marchok, T., Kuo, B., and Gall, R.: 2014, Community Support and Transition of Research to Operations for the Hurricane Weather Research and Forecast (HWRF) Model, accepted for publication in Bull. Amer. Meteor. Soc doi: http://dx.doi.org/10.1175/BAMS-D-13-00093.1

Biswas, M. K., L. Bernardet, and J. Dudhia, 2014: Sensitivity of hurricane forecasts to cumulus parameterizations in the HWRF model. Geophysical Research Letters, 41, doi: 10.1002/2014GL062071.
```
# HELP WITH HWRF
## ACKNOWLEDGING DTC HWRF HELPDESK
If significant help was provided via the HWRF helpdesk for work resulting in a publication, please acknowledge the Developmental Testbed Center Hurricane Team.

HOW TO ASK QUESTION OR REPORT AN ISSUE WITH THE HWRF MODEL?
If you have a question regarding HWRF, please first check the documentation and the list of known issues.

All questions and issues should be directed to the DTC HWRF community forum.

## IMPORTANT UPDATE
Starting 1st October, 2021 the Developmental Testbed Center (DTC) will not be actively monitoring and replying to queries posted on the HWRF and the GFDL vortex tracker Users Forum. We encourage the user community to actively participate in the discussions and share their knowledge with fellow users. There are no plans to do a new public release of the HWRF modeling system. Please consult the HWRF v4.0a Users¡¯ Guide and Scientific Documentation when configuring new experiments.
 
# HWRF Developers Page
Welcome to the DTC HWRF developers page, the source for information concerning the developmental code for HWRF.

For those working on code development in collaboration with NOAA (with the intention of contributing code back to the HWRF repository), and for those that need to use the latest experimental HWRF code, access to the HWRF code repository is necessary. To determine if you are a candidate for accessing the HWRF code repository, please contact hwrf-access@ucar.edu with the subject line "HWRF Code Repository".

This website provides an overview of the HWRF Code Repository, how to request repository access, information about code management and how to contribute code back to HWRF, details on how to check out, build and update your code, and information on forecast skill. To start, navigate to the option on the right titled Getting Started. If you have already been granted repository access, skip to the subsection titled Repository Structure.