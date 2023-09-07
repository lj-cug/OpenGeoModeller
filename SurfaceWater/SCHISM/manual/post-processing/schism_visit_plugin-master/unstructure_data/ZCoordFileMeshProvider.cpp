#include "ZCoordFileMeshProvider.h"
#include "MeshConstants.h"
#include "SCHISMFile.h"
#include <algorithm> 

using namespace std;

ZCoordMeshProvider::ZCoordMeshProvider(const std::string& a_zcoordFile):SCHISMMeshProvider(a_zcoordFile)
{

}

bool ZCoordMeshProvider::zcoords3D(float * a_zCachePtr,const int & a_timeStep) const
{
	 int timeStart    = a_timeStep;

     SCHISMVar * zVarPtr = m_dataFilePtr->get_var(MeshConstants::ZCOORD);
  
     if (!(zVarPtr->is_valid()))
     {
      
       throw SCHISMFileException("invlaid var "+MeshConstants::ZCOORD+" for data file "+m_dataFilePtr->file());
     }
   
     zVarPtr->set_cur(timeStart);

	 int z_var_size =0;
	 int * node_start_index= new int [m_number_node];
	 for(int iNode=0;iNode<m_number_node;iNode++)
	 {
		 node_start_index[iNode]=z_var_size;
		 z_var_size+=m_number_layer-std::max(1,m_kbp00[iNode])+1;
	 }

     float*           zPtr = new float [z_var_size];
     if (!(zVarPtr->get(zPtr)))
     {
        throw SCHISMFileException("fail to retrieve var "+MeshConstants::ZCOORD+" from data file "+m_dataFilePtr->file());
     }

	 // zcore z stored as layer dimension change first,here convert to node dim change first
	
	  for (int iLayer= 0; iLayer<m_number_layer;iLayer++)
      { 
		 
          for(int iNode=0;iNode <m_number_node; iNode++)
            {
			  int start_index = node_start_index[iNode];
		      if (iLayer<(std::max(1,m_kbp00[iNode])-1))
			  {
				  //*a_zCachePtr++ = zPtr[start_index];
			  }
			  else
			  {
                  *a_zCachePtr =  zPtr[start_index+iLayer+1-std::max(1,m_kbp00[iNode])];
				  a_zCachePtr++;
			  }

            }
       }
	 delete zPtr;
	 return true;
}

bool ZCoordMeshProvider::zcoords3D2(float * a_zCachePtr,const int & a_timeStep) const
{
	 int timeStart    = a_timeStep;

     SCHISMVar * zVarPtr = m_dataFilePtr->get_var(MeshConstants::ZCOORD);
  
     if (!(zVarPtr->is_valid()))
     {
      
       throw SCHISMFileException("invlaid var "+MeshConstants::ZCOORD+" for data file "+m_dataFilePtr->file());
     }
   
     zVarPtr->set_cur(timeStart);

	 int z_var_size =0;
	
	 for(int iNode=0;iNode<m_number_node;iNode++)
	 {
	
		 z_var_size+=m_number_layer-std::max(1,m_kbp00[iNode])+1;
	 }

     float*           zPtr = new float [z_var_size];
     if (!(zVarPtr->get(zPtr)))
     {
        throw SCHISMFileException("fail to retrieve var "+MeshConstants::ZCOORD+" from data file "+m_dataFilePtr->file());
     }


	  for (int i= 0; i<z_var_size;i++)
      { 
         a_zCachePtr[i] =  zPtr[i];
       }
	 delete zPtr;
	 return true;
}