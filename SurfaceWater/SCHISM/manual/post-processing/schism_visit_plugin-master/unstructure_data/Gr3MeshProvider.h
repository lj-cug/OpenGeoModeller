#ifndef  _GR3MESHPROVIDER_H_
#define  _GR3MESHPROVIDER_H_
#include "MeshProvider.h"

class Gr3MeshProvider:public MeshProvider
{
public :

   Gr3MeshProvider(const std::string & a_gr3File);
   
    ~Gr3MeshProvider();

   
   // provides point x,y,z coords at a time step
   bool fillPointCoord2D(float * a_pointCoord,const int & a_timeStep) const;

    // provides point x,y,z coords at a time step
   bool fillPointCoord3D(float * a_pointCoord,const int & a_timeStep) const;

   //provides mesh element nodes
   bool fillMeshElement(int * a_elementCache) const;
   
   // updates z coords at a timestep
   bool zcoords2D(float * a_zCachePtr,const int & a_timeStep) const;

   // updates z coords at a timestep
   bool zcoords3D(float * a_zCachePtr,const int & a_timeStep) const;

    // provides side centers x,y, z at a time step
   bool fillSideCenterCoord2D(float * a_pointCoord,const int & a_timeStep) const;

   bool fillEleCenterCoord2D(float * a_pointCoord,const int & a_timeStep) const;

   // updates z coords at a timestep
   bool zSideCenter2D(float * a_zCachePtr,const int & a_timeStep) const;

    // updates z coords at a timestep
   bool zEleCenter2D(float * a_zCachePtr,const int & a_timeStep) const;

   // return s layers
   bool slayers(float * a_cache) const;

   // return depth
   bool depth(float * a_cache) const;

   bool provide3DMesh() const;

protected:

	void loadMesh();

	
private:
	float * m_nodeXPtr      ;
    float * m_nodeYPtr      ;
    int *   m_faceNodesPtr  ;
    float * m_depthPtr      ; 
	int   * m_sideNodes     ;
};

#endif