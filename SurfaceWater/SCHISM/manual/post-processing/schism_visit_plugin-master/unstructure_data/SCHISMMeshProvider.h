#ifndef _SCHISMMESHPROVIDER_H_
#define _SCHISMMESHPROVIDER_H_    
#include "MeshProvider.h"
#include "SCHISMFile.h"

class SCHISMMeshProvider: public MeshProvider
{
public :

   SCHISMMeshProvider(const std::string & a_fileHasMeshData);
   
    ~SCHISMMeshProvider();

    
   // provides point x,y,z coords at a time step
   bool fillPointCoord2D(float * a_pointCoord,const int & a_timeStep) const;

    // provides point x,y,z coords at a time step
   bool fillPointCoord3D(float * a_pointCoord,const int & a_timeStep) const;

   //provides mesh element nodes
   bool fillMeshElement(int * a_elementCache) const;

    // provides side centers x,y, z at a time step
   bool fillSideCenterCoord2D(float * a_pointCoord,const int & a_timeStep) const;

   // provides side centers x,y, z at a time step
   bool fillSideCenterCoord3D(float * a_pointCoord,const int & a_timeStep) const;

    // provides sideface centers x,y, z at a time step
   bool fillSideFaceCenterCoord3D(float * a_pointCoord,const int & a_timeStep) const;

   bool fillEleCenterCoord2D(float * a_pointCoord,const int & a_timeStep) const;

   // provides side centers x,y, z at a time step
   bool fillEleCenterCoord3D(float * a_pointCoord,const int & a_timeStep) const;
   
   // updates z coords at a timestep
   bool zcoords2D(float * a_zCachePtr,const int & a_timeStep) const;

   // updates z coords at a timestep
   virtual bool zcoords3D(float * a_zCachePtr,const int & a_timeStep) const;

   //return z coords cache but layer dim change first
   virtual bool zcoords3D2(float * a_zCachePtr,const int & a_timeStep) const;

   // updates z coords at a timestep
   bool zSideCenter2D(float * a_zCachePtr,const int & a_timeStep) const;

   // updates z coords at a timestep
   bool zSideCenter3D(float * a_zCachePtr,const int & a_timeStep) const;

    // updates z coords at a timestep
   bool zEleCenter2D(float * a_zCachePtr,const int & a_timeStep) const;

   // updates z coords at a timestep
   bool zEleCenter3D(float * a_zCachePtr,const int & a_timeStep) const;

   // return s layers
   bool slayers(float * a_cache) const;

   // return depth
   bool depth(float * a_cache) const;

    //return kbp00
   bool fillKbp00(int * a_cache) const;

   bool fillKbs(int* a_cache) const;
  
   bool fillKbe(int* a_cache) const;

protected:

	void loadMeshDim();
	void genElementSide();
	
protected:

   SCHISMFile*   m_dataFilePtr;

   float        m_hs;
   float        m_hc;
   float        m_thetab;
   float        m_thetaf;
   float     *  m_layerSCoords;
   int       *  m_kbp00;
   int       *  m_kbs;
   int       *  m_kbe;
   int       *  m_sideNodes;

};

#endif