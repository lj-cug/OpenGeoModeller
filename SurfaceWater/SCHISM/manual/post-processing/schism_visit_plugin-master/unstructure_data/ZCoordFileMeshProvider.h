#ifndef _ZCOORDFILEMESHPROVIDER_H_
#define _ZCOORDFILEMESHPROVIDER_H_
#include "SCHISMMeshProvider.h"

class ZCoordMeshProvider:public SCHISMMeshProvider
{

public:
	ZCoordMeshProvider(const std::string & a_fileHasMeshData);

	bool zcoords3D(float * a_zCachePtr,const int & a_timeStep) const;
	//return z coords cache but layer dim change first
    bool zcoords3D2(float * a_zCachePtr,const int & a_timeStep) const;

private:

};
#endif