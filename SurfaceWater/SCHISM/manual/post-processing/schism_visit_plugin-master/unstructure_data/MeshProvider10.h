#ifndef _MESHPROVIDER10_H_
#define _MESHPROVIDER10_H_

#include <iostream>
#include <fstream>    
#include <string>    
using std::ios;
using std::ifstream;
using namespace std;

/**
  Abstract class define interface providing points in 2d or 3d mesh.
*/

class MeshProvider10
{

public :

   MeshProvider10(const std::string & a_fileHasMeshData);
   
   virtual ~MeshProvider10();

   
   // provides point x,y,z coords at a time step
   virtual bool fillPointCoord2D(float * a_pointCoord,const int & a_timeStep) const;

    // provides point x,y,z coords at a time step
   virtual bool fillPointCoord3D(float * a_pointCoord,const int & a_timeStep) const;


   // provides side centers x,y, z at a time step
   virtual bool fillSideCenterCoord2D(float * a_pointCoord,const int & a_timeStep) const;

   // provides side centers x,y, z at a time step
   virtual bool fillSideCenterCoord3D(float * a_pointCoord,const int & a_timeStep) const;

    // provides side centers x,y, z at a time step
   virtual bool fillSideFaceCenterCoord3D(float * a_pointCoord,const int & a_timeStep) const;

   // provides side centers x,y, z at a time step
   virtual bool fillEleCenterCoord2D(float * a_pointCoord,const int & a_timeStep) const;

   // provides side centers x,y, z at a time step
   virtual bool fillEleCenterCoord3D(float * a_pointCoord,const int & a_timeStep) const;


      // provides point x,y,z coords at a time step
   virtual bool fillPointCoord2D(double * a_pointCoord,const int & a_timeStep) const;

    // provides point x,y,z coords at a time step
   virtual bool fillPointCoord3D(double * a_pointCoord,const int & a_timeStep) const;


   // provides side centers x,y, z at a time step
   virtual bool fillSideCenterCoord2D(double * a_pointCoord,const int & a_timeStep) const;

   // provides side centers x,y, z at a time step
   virtual bool fillSideCenterCoord3D(double * a_pointCoord,const int & a_timeStep) const;

    // provides side centers x,y, z at a time step
   virtual bool fillSideFaceCenterCoord3D(double * a_pointCoord,const int & a_timeStep) const;

   // provides side centers x,y, z at a time step
   virtual bool fillEleCenterCoord2D(double * a_pointCoord,const int & a_timeStep) const;

   // provides side centers x,y, z at a time step
   virtual bool fillEleCenterCoord3D(double * a_pointCoord,const int & a_timeStep) const;

   //provides mesh element nodes
   virtual bool fillMeshElement(long * a_elementCache) const;
   
   // updates z coords at a timestep
   virtual bool zcoords2D(float * a_zCachePtr,const int & a_timeStep) const;

   // updates z coords at a timestep
   virtual bool zcoords3D(float * a_zCachePtr,const int & a_timeStep) const;

   // updates z coords at a timestep
   virtual bool zSideCenter2D(float * a_zCachePtr,const int & a_timeStep) const;

   // updates z coords at a timestep
   virtual bool zSideCenter3D(float * a_zCachePtr,const int & a_timeStep) const;

    // updates z coords at a timestep
   virtual bool zEleCenter2D(float * a_zCachePtr,const int & a_timeStep) const;

   // updates z coords at a timestep
   virtual bool zEleCenter3D(float * a_zCachePtr,const int & a_timeStep) const;

   // return s layers
   virtual bool slayers(float * a_cache) const;

   // return depth
   virtual bool depth(float * a_cache) const;
   
   bool isValid() const;

   // number of 2D mesh element
   long numberOfElement() const;
   // number of 2d mesh node
   long numberOfNode() const;
   // number of side for 2d mesh
   long numberOfSide() const;
   
   // number of layers
   int numberOfLayer() const;

   // number of nodes per cell
   int numberOfNodesPerCell() const;

   //
   std::string file() const;

   // has 3D mesh or not
   virtual bool provide3DMesh() const;

 
    //return kbp00
   virtual bool fillKbp00(int * a_cache,const int & a_timeStep) const;

   //return kbs
   virtual bool fillKbs(int* a_cache,const int & a_timeStep) const;

   //return kbe
   virtual bool fillKbe(int* a_cache,const int & a_timeStep) const;

   //update 3d grid bottom
   virtual bool update_bottom_layer(const int& a_timeStep);
   virtual void  fill_ele_dry_wet(int*  &a_ele_dry_wet, const int& a_step);
   virtual void  fill_node_dry_wet(int* &a_node_dry_wet,int* a_ele_dry_wet);
   virtual void  fill_side_dry_wet(int* &a_side_dry_wet,int* a_ele_dry_wet);

   virtual bool mesh3d_is_static() const;
   
protected:
	std::string m_data_file;
	bool        m_valid_provider;
	bool        m_mesh_loaded;
	long         m_number_element;
	long         m_number_node;
	int         m_number_layer;
	int         m_number_nodes_per_cell;
	long         m_number_side;
  
  
};


#endif