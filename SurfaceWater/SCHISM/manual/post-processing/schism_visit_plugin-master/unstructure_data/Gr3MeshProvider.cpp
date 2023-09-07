#include "Gr3MeshProvider.h"
#include "MeshConstants.h"
#include "SchismGeometry.h"

Gr3MeshProvider::Gr3MeshProvider(const std::string & a_gr3File):MeshProvider(a_gr3File),
																m_nodeXPtr(NULL),
																m_nodeYPtr(NULL),
																m_faceNodesPtr(NULL),
																m_depthPtr(NULL)

{
	loadMesh();
	if (m_mesh_loaded)
	{
       this->m_valid_provider = true;  
	}
	else
	{
	   this->m_valid_provider = false;
	}
	
}


Gr3MeshProvider::~Gr3MeshProvider()
{
	if(m_nodeXPtr)
	{
		delete m_nodeXPtr;
	}
	if(m_nodeYPtr)
	{
		delete m_nodeYPtr;
	}
	if(m_faceNodesPtr)
	{
		delete m_faceNodesPtr;
	}
	if(m_depthPtr)
	{
		delete m_depthPtr;
	}
}
bool Gr3MeshProvider::provide3DMesh() const
{
	return false;
}
void Gr3MeshProvider::loadMesh()
{

   ifstream    gr3FileStream(m_data_file.c_str()); 

   if (!gr3FileStream.good())
    {
       
       return;
    }
  
   int numMeshFaces, numMeshNodes;
   std::string variableName;  

   gr3FileStream>>variableName;
   gr3FileStream>>numMeshFaces>>numMeshNodes;
  
   m_number_element = numMeshFaces;
   m_number_node    = numMeshNodes;
   m_number_layer   = 1;


   m_nodeXPtr     = new float [numMeshNodes];
   m_nodeYPtr     = new float [numMeshNodes];
   m_faceNodesPtr = new int   [(MeshConstants::MAX_NUM_NODE_PER_CELL+1)*numMeshFaces];
   m_depthPtr     = new float [numMeshNodes];

   for(int iNode=0;iNode<numMeshNodes;iNode++)
     {
      int valtemp;
      gr3FileStream>>valtemp>>m_nodeXPtr[iNode]>>m_nodeYPtr[iNode]>>m_depthPtr[iNode];
    
     }
   
   int * faceNodePtr = m_faceNodesPtr;

   for(int iFace=0;iFace<numMeshFaces;iFace++)
     {
      int valtemp=MeshConstants::INVALID_NUM; 
      int numNode=MeshConstants::INVALID_NUM;
      gr3FileStream>>valtemp>>numNode;
	  (*faceNodePtr) = numNode;
	  faceNodePtr++;
	  for(int iNode=0;iNode<numNode;iNode++)
	  {
		  int nodeID=MeshConstants::INVALID_NUM;
		  gr3FileStream>>nodeID;
	      (*faceNodePtr) = nodeID;
		  faceNodePtr++;
	  }
	  for(int iNode=(numNode+1);iNode<(MeshConstants::MAX_NUM_NODE_PER_CELL+1);iNode++)
	  {
		  *faceNodePtr = MeshConstants::INVALID_NUM;
		  faceNodePtr++;
	  }
     }
   
   
   gr3FileStream.close();
  
   m_number_side = meshSideNum( m_faceNodesPtr,m_number_element,m_number_node);

   m_sideNodes = new int [m_number_side*2];
   meshSideNode(m_sideNodes,m_faceNodesPtr,m_number_side,m_number_element,m_number_node);

   m_mesh_loaded = true;
}


// z is filled with 0
bool Gr3MeshProvider::fillPointCoord2D(float * a_pointCoord,const int & a_timeStep) const
{
  if (!m_mesh_loaded)
  {
	  return false;
  }
  for(int iNode=0;iNode < m_number_node; iNode++)
    {
    float x            =  m_nodeXPtr[iNode];
    float y            =  m_nodeYPtr[iNode];
    *a_pointCoord++    = x;
    *a_pointCoord++    = y;
    // must put a dummy z value as visit manaul example does
    *a_pointCoord++    = MeshConstants::DUMMY_ELEVATION;
           
   }
	return true;
}

bool Gr3MeshProvider::fillPointCoord3D(float * a_pointCoord,const int & a_timeStep) const
{
	
    return false;
	
}

bool  Gr3MeshProvider::fillMeshElement(int * a_elementCache) const
{
  if (!m_mesh_loaded)
  {
	  return false;
  }
 int * faceNodesPtr = m_faceNodesPtr;
 for(int i=0;i<(MeshConstants::MAX_NUM_NODE_PER_CELL+1)*m_number_element;i++)
 {
  
    *a_elementCache++=*faceNodesPtr++;
 }
   return true;
}

bool Gr3MeshProvider::zcoords2D(float * a_zCachePtr,const int & a_timeStep) const
{
	 for(int iNode=0;iNode <m_number_node; iNode++)
        {
	
            *a_zCachePtr++         = MeshConstants::DUMMY_ELEVATION;
        }

	 return true;
}

bool Gr3MeshProvider::fillSideCenterCoord2D(float * a_pointCoord,const int & a_timeStep) const
{
	if( m_mesh_loaded==false)
	{
	   return false;
	}
	else
	{
		
      float half=0.5;
	  for(int iSide=0;iSide < m_number_side; iSide++)
          {
			int node1 = m_sideNodes[iSide*2];
		    int node2 = m_sideNodes[iSide*2+1];
            float x1            =  m_nodeXPtr[node1];
            float y1            =  m_nodeYPtr[node1];
			float x2            =  m_nodeXPtr[node2];
            float y2            =  m_nodeYPtr[node2];
            *a_pointCoord++    = half*(x1+x2);
            *a_pointCoord++    = half*(y1+y2);
            // must put a dummy z value as visit manaul example does
            *a_pointCoord++    = MeshConstants::DUMMY_ELEVATION;
           
	  }
	}
	return true;

}

bool Gr3MeshProvider::fillEleCenterCoord2D(float * a_pointCoord,const int & a_timeStep) const
{
	if( m_mesh_loaded==false)
	{
	   return false;
	}
	else
	{
	  int maxNodeInCell = MeshConstants::MAX_NUM_NODE_PER_CELL;

      for(int iEle=0;iEle <m_number_element; iEle++)
      {		
		int numNode = m_faceNodesPtr[iEle*(maxNodeInCell+1)];
		       
		float x_sum = 0.0;
		float y_sum = 0.0;
			
        for (int iNode=0;iNode<numNode;iNode++)
		{  
           int node=m_faceNodesPtr[iEle*(maxNodeInCell+1)+1+iNode]-1;
		   x_sum += m_nodeXPtr[iNode];
		   y_sum += m_nodeYPtr[iNode];
		}
      
		*a_pointCoord++     = x_sum/numNode;
        *a_pointCoord++     = y_sum/numNode;
        *a_pointCoord++     = MeshConstants::DUMMY_ELEVATION;
               
		}
     
	}
	return true;
}

 bool Gr3MeshProvider::zSideCenter2D(float * a_zCachePtr,const int & a_timeStep) const
 {
	 for(int iNode=0;iNode <m_number_side; iNode++)
        {
	
            *a_zCachePtr++         = MeshConstants::DUMMY_ELEVATION;
        }

	 return true;
 }

 bool Gr3MeshProvider::zEleCenter2D(float * a_zCachePtr,const int & a_timeStep) const
 {
	 for(int iNode=0;iNode <m_number_side; iNode++)
        {
	
            *a_zCachePtr++         = MeshConstants::DUMMY_ELEVATION;
        }

	 return true;
 }

bool Gr3MeshProvider::zcoords3D(float * a_zCachePtr,const int & a_timeStep) const
{
	return false;
}

bool Gr3MeshProvider::slayers(float * a_cache) const
{
	return false;
}

bool Gr3MeshProvider::depth(float * a_cache) const
{
  if (!m_mesh_loaded)
  {
	  return false;
  }
  for(int iNode=0;iNode < m_number_node; iNode++)
    {
    *a_cache++    = m_depthPtr[iNode];
           
   }
	return true;
}
