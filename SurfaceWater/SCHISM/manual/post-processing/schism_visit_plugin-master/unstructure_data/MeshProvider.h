#ifndef _MESHPROVIDER_H_
#define _MESHPROVIDER_H_

#include <iostream>
#include <fstream>    
#include <string>    
using std::ios;
using std::ifstream;
using namespace std;

/**
  Abstract class define interface providing points in 2d or 3d mesh.
*/

class MeshProvider
{

public :

   MeshProvider(const std::string & a_fileHasMeshData);
   
   virtual ~MeshProvider();

   
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

   //provides mesh element nodes
   virtual bool fillMeshElement(int * a_elementCache) const;
   
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
   int numberOfElement() const;
   // number of 2d mesh node
   int numberOfNode() const;
   // number of side for 2d mesh
   int numberOfSide() const;
   
   // number of layers
   int numberOfLayer() const;

   // number of nodes per cell
   int numberOfNodesPerCell() const;

   //
   std::string file() const;

   // has 3D mesh or not
   virtual bool provide3DMesh() const;

 
    //return kbp00
   virtual bool fillKbp00(int * a_cache) const;

   //return kbs
   virtual bool fillKbs(int* a_cache) const;

   //return kbe
   virtual bool fillKbe(int* a_cache) const;

protected:
	std::string m_data_file;
	bool        m_valid_provider;
	bool        m_mesh_loaded;
	int         m_number_element;
	int         m_number_node;
	int         m_number_layer;
	int         m_number_nodes_per_cell;
	int         m_number_side;
  
  
};


#endif