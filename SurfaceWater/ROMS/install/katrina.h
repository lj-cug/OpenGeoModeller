#define REGCM_COUPLING
#define MODIFIED_CALDATE
#define PERFECT_RESTART
#define AVERAGES
#define AVERAGES_FLUX

/*
** Options associated with momentum equations:
*/

#define UV_ADV
#define UV_COR
#define UV_VIS2
#define UV_QDRAG
#define UV_U3HADVECTION
#define UV_C4VADVECTION

/*
** Options associated with tracers equations:
*/

#define TS_MPDATA
#define TS_DIF2
#define SALINITY
#define NONLIN_EOS
#define SOLAR_SOURCE

/*
** Options for pressure gradient algorithm:
*/

#define DJ_GRADPS
#define ATM_PRESS

/*
** Options for atmospheric boundary layer:
*/

!#define BULK_FLUXES
!#define EMINUSP
!#define EMINUSP_SSH
!#define LONGWAVE_OUT
#define SHORTWAVE
!#define SPECIFIC_HUMIDITY

/*
** Options for model configuration:
*/

#define SOLVE3D
#define MASKING
#define DIFF_GRID
#define VISC_GRID

/*
** Options for analytical fields configuration:
*/

#define ANA_BSFLUX
#define ANA_BTFLUX

/*
** Options for horizontal mixing of momentum:
*/

#define MIX_S_UV
#define MIX_GEO_TS 

/*
** Options for vertical mixing momentum and tracers:
*/

#define MY25_MIXING
#define KANTHA_CLAYSON
#define N2S2_HORAVG

/*
** Options for lateral boundary condition 
*/
#define RADIATION_2D
