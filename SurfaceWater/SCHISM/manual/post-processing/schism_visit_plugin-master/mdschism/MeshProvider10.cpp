#include "MeshProvider10.h"
#include "MeshConstants10.h"

MeshProvider10::MeshProvider10(const std::string & a_fileHasMeshData):m_data_file(a_fileHasMeshData),
                                                                  m_mesh_loaded(false),
																  m_number_element(-9999),
																  m_number_node(-9999),
																  m_number_layer(-9999),
																  m_number_side(-9999),
																  m_number_nodes_per_cell(MeshConstants10::MAX_NUM_NODE_PER_CELL)
{

}

MeshProvider10::~MeshProvider10()
{
}

bool MeshProvider10::provide3DMesh() const
{
	return true;
}
bool MeshProvider10::fillPointCoord2D(float * a_pointCoord,const int & a_timeStep) const
{
	return false;
}

bool MeshProvider10::fillPointCoord3D(float * a_pointCoord,const int & a_timeStep) const
{
	return false;
}

bool MeshProvider10::fillSideCenterCoord3D(float * a_pointCoord,const int & a_timeStep) const
{
	return false;
}

bool MeshProvider10::fillSideFaceCenterCoord3D(float * a_pointCoord,const int & a_timeStep) const
{
	return false;
}


bool MeshProvider10::fillSideCenterCoord2D(float * a_pointCoord,const int & a_timeStep) const
{
	return false;
}

bool MeshProvider10::fillEleCenterCoord3D(float * a_pointCoord,const int & a_timeStep) const
{
	return false;
}


bool MeshProvider10::fillEleCenterCoord2D(float * a_pointCoord,const int & a_timeStep) const
{
	return false;
}

bool MeshProvider10::fillPointCoord2D(double * a_pointCoord,const int & a_timeStep) const
{
	return false;
}

bool MeshProvider10::fillPointCoord3D(double * a_pointCoord,const int & a_timeStep) const
{
	return false;
}

bool MeshProvider10::fillSideCenterCoord3D(double * a_pointCoord,const int & a_timeStep) const
{
	return false;
}

bool MeshProvider10::fillSideFaceCenterCoord3D(double * a_pointCoord,const int & a_timeStep) const
{
	return false;
}


bool MeshProvider10::fillSideCenterCoord2D(double * a_pointCoord,const int & a_timeStep) const
{
	return false;
}

bool MeshProvider10::fillEleCenterCoord3D(double * a_pointCoord,const int & a_timeStep) const
{
	return false;
}


bool MeshProvider10::fillEleCenterCoord2D(double * a_pointCoord,const int & a_timeStep) const
{
	return false;
}

bool  MeshProvider10::fillMeshElement(long * a_elementCache) const
{
	return false;
}

bool MeshProvider10::zcoords2D(float * a_zCachePtr,const int & a_timeStep) const
{
	return false;
}

bool MeshProvider10::zcoords3D(float * a_zCachePtr,const int & a_timeStep) const
{
	return false;
}

 
bool MeshProvider10::zSideCenter2D(float * a_zCachePtr,const int & a_timeStep) const
{
	return false;
}

 
bool MeshProvider10::zSideCenter3D(float * a_zCachePtr,const int & a_timeStep) const
{
	return false;
}

bool MeshProvider10::zEleCenter2D(float * a_zCachePtr,const int & a_timeStep) const
{
	return false;
}

 
bool MeshProvider10::zEleCenter3D(float * a_zCachePtr,const int & a_timeStep) const
{
	return false;
}

bool MeshProvider10::slayers(float * a_cache) const
{
	return false;
}

bool MeshProvider10::depth(float * a_cache) const
{
	return false;
}

bool MeshProvider10::isValid() const
{
	return m_valid_provider;
}
  
long MeshProvider10::numberOfElement() const
{
	return m_number_element;
}

long MeshProvider10::numberOfSide() const
{
	return m_number_side;
}
   
long MeshProvider10::numberOfNode() const
{
	return m_number_node;
}
   
int MeshProvider10::numberOfLayer() const
{
	return m_number_layer;
}

int MeshProvider10::numberOfNodesPerCell() const
{
	return m_number_nodes_per_cell;
}


std::string MeshProvider10::file() const
{
	return m_data_file;
}

bool MeshProvider10::fillKbp00(int * a_cache,const int & a_timeStep) const
{
	for(long i=0;i<m_number_node;i++)
	{
		a_cache[i]=1; //bottom start from 1 (bottom layer) by default.
	}
	return true;
}

bool MeshProvider10::fillKbs(int * a_cache,const int & a_timeStep) const
{
	for(long i=0;i<m_number_side;i++)
	{
		a_cache[i]=1; //bottom start from 1 (bottom layer) by default.
	}
	return true;
}

bool MeshProvider10::fillKbe(int * a_cache,const int & a_timeStep) const
{
	for(long i=0;i<m_number_element;i++)
	{
		a_cache[i]=1; //bottom start from 1 (bottom layer) by default.
	}
	return true;
}

bool MeshProvider10::update_bottom_layer(const int & a_timeStep) 
{
	
	return false;
}

bool MeshProvider10::mesh3d_is_static() const
{
	return true;
}


 void   MeshProvider10::fill_node_dry_wet(int* &a_node_dry_wet,int* a_ele_dry_wet)
 {
	 return;
 }
 
 void   MeshProvider10::fill_side_dry_wet(int* &a_side_dry_wet,int* a_ele_dry_wet)
 {
	 return;
 }