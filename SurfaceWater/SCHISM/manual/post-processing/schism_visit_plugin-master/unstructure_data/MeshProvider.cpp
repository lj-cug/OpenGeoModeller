#include "MeshProvider.h"
#include "MeshConstants.h"

MeshProvider::MeshProvider(const std::string & a_fileHasMeshData):m_data_file(a_fileHasMeshData),
                                                                  m_mesh_loaded(false),
																  m_number_element(-9999),
																  m_number_node(-9999),
																  m_number_layer(-9999),
																  m_number_side(-9999),
																  m_number_nodes_per_cell(MeshConstants::MAX_NUM_NODE_PER_CELL)
{

}

MeshProvider::~MeshProvider()
{
}

bool MeshProvider::provide3DMesh() const
{
	return true;
}
bool MeshProvider::fillPointCoord2D(float * a_pointCoord,const int & a_timeStep) const
{
	return false;
}

bool MeshProvider::fillPointCoord3D(float * a_pointCoord,const int & a_timeStep) const
{
	return false;
}

bool MeshProvider::fillSideCenterCoord3D(float * a_pointCoord,const int & a_timeStep) const
{
	return false;
}

bool MeshProvider::fillSideFaceCenterCoord3D(float * a_pointCoord,const int & a_timeStep) const
{
	return false;
}


bool MeshProvider::fillSideCenterCoord2D(float * a_pointCoord,const int & a_timeStep) const
{
	return false;
}

bool MeshProvider::fillEleCenterCoord3D(float * a_pointCoord,const int & a_timeStep) const
{
	return false;
}


bool MeshProvider::fillEleCenterCoord2D(float * a_pointCoord,const int & a_timeStep) const
{
	return false;
}

bool  MeshProvider::fillMeshElement(int * a_elementCache) const
{
	return false;
}

bool MeshProvider::zcoords2D(float * a_zCachePtr,const int & a_timeStep) const
{
	return false;
}

bool MeshProvider::zcoords3D(float * a_zCachePtr,const int & a_timeStep) const
{
	return false;
}

 
bool MeshProvider::zSideCenter2D(float * a_zCachePtr,const int & a_timeStep) const
{
	return false;
}

 
bool MeshProvider::zSideCenter3D(float * a_zCachePtr,const int & a_timeStep) const
{
	return false;
}

bool MeshProvider::zEleCenter2D(float * a_zCachePtr,const int & a_timeStep) const
{
	return false;
}

 
bool MeshProvider::zEleCenter3D(float * a_zCachePtr,const int & a_timeStep) const
{
	return false;
}

bool MeshProvider::slayers(float * a_cache) const
{
	return false;
}

bool MeshProvider::depth(float * a_cache) const
{
	return false;
}

bool MeshProvider::isValid() const
{
	return m_valid_provider;
}
  
int MeshProvider::numberOfElement() const
{
	return m_number_element;
}

int MeshProvider::numberOfSide() const
{
	return m_number_side;
}
   
int MeshProvider::numberOfNode() const
{
	return m_number_node;
}
   
int MeshProvider::numberOfLayer() const
{
	return m_number_layer;
}

int MeshProvider::numberOfNodesPerCell() const
{
	return m_number_nodes_per_cell;
}


std::string MeshProvider::file() const
{
	return m_data_file;
}

bool MeshProvider::fillKbp00(int * a_cache) const
{
	for(int i=0;i<m_number_node;i++)
	{
		a_cache[i]=1; //bottom start from 1 (bottom layer) by default.
	}
	return true;
}

bool MeshProvider::fillKbs(int * a_cache) const
{
	for(int i=0;i<m_number_side;i++)
	{
		a_cache[i]=1; //bottom start from 1 (bottom layer) by default.
	}
	return true;
}

bool MeshProvider::fillKbe(int * a_cache) const
{
	for(int i=0;i<m_number_element;i++)
	{
		a_cache[i]=1; //bottom start from 1 (bottom layer) by default.
	}
	return true;
}