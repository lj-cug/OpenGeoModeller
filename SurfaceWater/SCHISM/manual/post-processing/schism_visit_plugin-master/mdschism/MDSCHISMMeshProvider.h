#ifndef _MDSCHISMMESHPROVIDER_H_
#define _MDSCHISMMESHPROVIDER_H_    
#include "MeshProvider10.h"
#include "SCHISMFile10.h"


class MDSCHISMMeshProvider: public MeshProvider10
{
public :

   MDSCHISMMeshProvider(const std::string & a_ncfile,SCHISMFile10 * a_nc_ptr,const std::string & a_local_gloabl_file);
   
    virtual ~MDSCHISMMeshProvider();

    
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
   bool zcoords2D(double * a_zCachePtr,const int & a_timeStep) const;

   // updates z coords with node dim change fast at a timestep
   bool zcoords3D(double * a_zCachePtr,const int & a_timeStep) const;

   //return z coords cache but layer dim change first
   bool zcoords3D2(double * a_zCachePtr,const int & a_timeStep) const;

   // updates z coords at a timestep
   bool zSideCenter2D(double * a_zCachePtr,const int & a_timeStep) const;

   // updates z coords at a timestep
   bool zSideCenter3D(double * a_zCachePtr,const int & a_timeStep) const;

    // updates z coords at a timestep
   bool zEleCenter2D(double * a_zCachePtr,const int & a_timeStep) const;

   // updates z coords at a timestep
   bool zEleCenter3D(double * a_zCachePtr,const int & a_timeStep) const;

      // updates z coords at a timestep
   bool zcoords2D(float * a_zCachePtr,const int & a_timeStep) const;

   // updates z coords at a timestep
   bool zcoords3D(float * a_zCachePtr,const int & a_timeStep) const;

     //return z coords cache but layer dim change first
   bool zcoords3D2(float * a_zCachePtr,const int & a_timeStep) const;

   // updates z coords at a timestep
   bool zSideCenter2D(float * a_zCachePtr,const int & a_timeStep) const;

   // updates z coords at a timestep
   bool zSideCenter3D(float * a_zCachePtr,const int & a_timeStep) const;

    // updates z coords at a timestep
   bool zEleCenter2D(float * a_zCachePtr,const int & a_timeStep) const;

   // updates z coords at a timestep
   bool zEleCenter3D(float * a_zCachePtr,const int & a_timeStep) const;

   // return s layers
   bool slayers(double * a_cache) const;

   // return depth
   bool depth(double * a_cache) const;

    //return kbp00
   bool fillKbp00(int * a_cache,const int & a_timeStep) const;

   bool fillKbs(int* a_cache,const int & a_timeStep) const;
  
   bool fillKbe(int* a_cache,const int & a_timeStep) const;


   void  fill_node_dry_wet(int* &a_node_dry_wet,int* a_ele_dry_wet);
   void  fill_side_dry_wet(int* &a_side_dry_wet,int* a_ele_dry_wet);
   void  fill_node_global_id(long * a_buff);
   void  fill_ele_global_id(long * a_buff);

   bool  update_bottom_layer(const int & a_timeStep);

   virtual bool mesh3d_is_static() const;
   bool mesh_loaded() const;
   bool set_data_file(SCHISMFile10* a_file);
public:
	long         m_number_side_no_ghost;
	long         m_number_node_no_ghost;
	long         m_number_element_no_ghost;
	long      *  m_local_node_id_to_global_id;
	long      *  m_local_ele_id_to_global_id;
	long      *  m_local_side_id_to_global_id;
	
private:

   double  convertStoZ(const double    & a_sigma,
                       const double    & a_surface,
                       const double    & a_depth,
                       const double    & a_hs,
                       const double    & a_hc,
                       const double    & a_thetab,
                       const double    & a_thetaf) const;

protected:

	bool loadMesh();
	//bool loadSide();
	
protected:

   SCHISMFile10*   m_dataFilePtr;
   std::string   m_local_global_file;
   double        m_hs;
   double        m_hc;
   double        m_thetab;
   double        m_thetaf;
   double     *  m_layerSCoords;
   int       *  m_kbp00;
   int       *  m_kbs;
   int       *  m_kbe;
   long      *  m_side_nodes;
   long      *  m_node_neighbor_ele;
   long      *  m_side_neighbor_ele;
   bool         m_kbp_ele_filled;
   bool         m_kbp_side_filled;
   bool         m_kbp_node_filled;
   bool         m_node_neighbor_ele_filled;
   bool         m_side_neighbor_ele_filled;
   int          m_max_ele_at_node;
   double    *  m_nodex;
   double    *  m_nodey;
   long      *  m_faceNodesPtr;
   
   double    *  m_dp;

   
};

#endif