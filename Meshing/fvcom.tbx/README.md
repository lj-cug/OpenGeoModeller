# fvcom.tbx

fvcom.tbx is an R toolbox for the exploration of unstructured, prism-based hydrodynamic model outputs (i.e. from the Finite Coastal Ocean Volume Model, FVCOM) in R. The package has been designed specifically for the West Coast of Scotland Coastal Modelling System (WeStCOMS), which implements FVCOM. Package development has been motivated by the requirements of ecological research, and the need to link hydrodynamic model outputs with ecological analyses implemented in R. To this end, the package includes functions which facilitate the following operations:

Acquiring FVCOM outputs from the Scottish Association of Marine Sciences (SAMS) thredds server;
Processing FVCOM outputs for integration with R;
Computing new hydrodynamic/environmental fields;
Building unstructured mesh(es) as spatial objects and locating cells and coordinates;
Extracting and interpolating model predictions;
Exploring environmental conditions through space and time with statistical summaries and maps;
Validating FVCOM predictions with observations, including from diverse animal movement datasets;
This README file outlines the steps that are required to set up FVCOM outputs for integration with R via fvcom.tbx and some of the main functionality of the package. For further details, please consult the vignette and the reference manual.

https://hub.fgit.cf/edwardlavender/fvcom.tbx
