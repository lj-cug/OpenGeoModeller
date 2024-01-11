# GOTM-FABM-PDAF

[EAT-Wiki](https://github.com/BoldingBruggeman/eat/wiki/)

## Building and installing manually
```
git clone --recursive https://github.com/BoldingBruggeman/eat.git
cd eat
conda env create -f <ENVIRONMENT_YML>
conda activate eat
source ./install
```
### Update
```
git pull
git submodule update --init --recursive
conda env update -f <ENVIRONMENT_YML>
conda activate eat
source ./install
```
### Uninstall

conda env remove -n eat

## Code availability

All source code is publicly available, though for most applications it suffices to install a pre-compiled EAT
package from Anaconda (https://anaconda.org/bolding-bruggeman/eatpy).

The EAT source code is available from https://github.com/BoldingBruggeman/eat. It includes compatible versions of GOTM, FABM and PDAF as submodules. These individual components are also available standalone
from https://github.com/gotm-model/code (GOTM), https://github.com/fabm-model/fabm (FABM) and
https://github.com/pdaf/PDAF (PDAF).

The example applications use FABM-based biogeochemical models that are hosted externally:
```
PISCES: https://github.com/BoldingBruggeman/fabm-pisces
BFM: https://github.com/inogs/bfmforfabm
ERSEM: https://github.com/pmlmodelling/ersem
```

The exact version of the combined codes that were used is available at
https://doi.org/10.5281/zenodo.10306436.

EAT documentation is available at https://github.com/BoldingBruggeman/eat/wiki.

## Data availability

The three example applications are available from https://doi.org/10.5281/zenodo.10307316. 

This archive includes model configurations, observations, forcing data, run script and pre/post processing scripts.

## ╡н©╪ндов

Jorn Bruggeman, et al. EAT v0.9.6: a 1D testbed for physical-biogeochemical data assimilation in natural waters. https://doi.org/10.5194/gmd-2023-238
