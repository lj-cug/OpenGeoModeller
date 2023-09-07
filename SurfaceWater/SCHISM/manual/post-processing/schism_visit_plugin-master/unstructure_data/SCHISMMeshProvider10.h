#ifndef _SCHISMMESHPROVIDER10_H_
#define _SCHISMMESHPROVIDER10_H_    
#include "MeshProvider10.h"
#include "SCHISMFile10.h"

class SCHISMMeshProvider10: public MeshProvider10
{
public :

   SCHISMMeshProvider10(const std::string & a_fileHasMeshData);
   
    virtual ~SCHISMMeshProvider10();

    
   // provides point x,y,z coords at a time step
   bool fillPointCoord2D(float * a_pointCoord,const int & a_timeStep) const;

    // provides point x,y,z coords at a time step
   bool fillPointCoord3D(float * a_pointCoord,const int & a_timeStep) const;

   //provides mesh element nodes
   bool fillMeshElement(long * a_elementCache) const;

    // provides side centers x,y, z at a time step
   bool fillSideCenterCoord2D(float * a_pointCoord,const int & a_timeStep) const;

   // provides side centers x,y, z at a time step
   bool fillSideCenterCoord3D(float * a_pointCoord,const int & a_timeStep) const;

    // provides sideface centers x,y, z at a time step
   bool fillSideFaceCenterCoord3D(float * a_pointCoord,const int & a_timeStep) const;

   bool fillEleCenterCoord2D(float * a_pointCoord,const int & a_timeStep) const;

   // provides side centers x,y, z at a time step
   bool fillEleCenterCoord3D(float * a_pointCoord,const int & a_timeStep) const;


      // provides point x,y,z coords at a time step
   bool fillPointCoord2D(double * a_pointCoord,const int & a_timeStep) const;

    // provides point x,y,z coords at a time step
   bool fillPointCoord3D(double * a_pointCoord,const int & a_timeStep) const;

    // provides side centers x,y, z at a time step
   bool fillSideCenterCoord2D(double * a_pointCoord,const int & a_timeStep) const;

   // provides side centers x,y, z at a time step
   bool fillSideCenterCoord3D(double * a_pointCoord,const int & a_timeStep) const;

    // provides sideface centers x,y, z at a time step
   bool fillSideFaceCenterCoord3D(double * a_pointCoord,const int & a_timeStep) const;

   bool fillEleCenterCoord2D(double * a_pointCoord,const int & a_timeStep) const;

   // provides side centers x,y, z at a time step
   bool fillEleCenterCoord3D(double * a_pointCoord,const int & a_timeStep) const;
   
   // updates z coords at a timestep
   bool zcoords2D(float * a_zCachePtr,const int & a_timeStep) const;

   // updates z coords with node dim change fast at a timestep
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
   bool fillKbp00(int * a_cache,const int & a_timeStep) const;

   bool fillKbs(int* a_cache,const int & a_timeStep) const;
  
   bool fillKbe(int* a_cache,const int & a_timeStep) const;

   void  fill_ele_dry_wet(int*  &a_ele_dry_wet, const int& a_step);
   void  fill_node_dry_wet(int* &a_node_dry_wet,int* a_ele_dry_wet);
   void  fill_side_dry_wet(int* &a_side_dry_wet,int* a_ele_dry_wet);

   bool  update_bottom_layer(const int & a_timeStep);

   virtual bool mesh3d_is_static() const;
   SCHISMFile10* get_mesh_data_ptr() const;
protected:

	bool loadMeshDim();
	bool loadSide();
	
protected:

   SCHISMFile10*   m_dataFilePtr;
   float        m_hs;
   float        m_hc;
   float        m_thetab;
   float        m_thetaf;
   float     *  m_layerSCoords;
   int       *  m_kbp00;
   int       *  m_kbs;
   int       *  m_kbe;
   long      *  m_sideNodes;
   long      *  m_node_neighbor_ele;
   long      *  m_side_neighbor_ele;
   bool         m_kbp_ele_filled;
   bool         m_kbp_side_filled;
   bool         m_kbp_node_filled;
   bool         m_node_neighbor_ele_filled;
   bool         m_side_neighbor_ele_filled;
   int          m_max_ele_at_node;
};

#endif