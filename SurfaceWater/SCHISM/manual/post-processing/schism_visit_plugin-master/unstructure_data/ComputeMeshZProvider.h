#ifndef _COMPUTEMESHZPROVIDER_H_
#define _COMPUTEMESHZPROVIDER_H_
#include "SCHISMMeshProvider.h"

class ComputeMeshZProvider:public SCHISMMeshProvider
{

public:
	ComputeMeshZProvider(const std::string& a_selfeOutputFile);

	bool zcoords3D(float * a_zCachePtr,const int & a_timeStep) const;
	bool zcoords3D2(float * a_zCachePtr,const int & a_timeStep) const;

private:

	float  convertStoZ(const float    & a_sigma,
                       const float    & a_surface,
                       const float    & a_depth,
                       const float    & a_hs,
                       const float    & a_hc,
                       const float    & a_thetab,
                       const float    & a_thetaf) const;

};
#endif