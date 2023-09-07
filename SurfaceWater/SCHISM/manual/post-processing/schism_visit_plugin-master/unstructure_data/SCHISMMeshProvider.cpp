#include <algorithm>
#include "SCHISMMeshProvider.h"
#include "MeshConstants.h"
#include "SCHISMFileUtil.h"
#include "NetcdfSCHISMOutput.h"
#include "NativeSCHISMOutput.h"
#include "SchismGeometry.h"

SCHISMMeshProvider::SCHISMMeshProvider(const std::string & a_fileHasMeshData):MeshProvider(a_fileHasMeshData),
	                                                                          m_kbp00(NULL),
																			  m_layerSCoords(NULL),
                                                                              m_kbs(NULL),
                                                                              m_kbe(NULL),
                                                                              m_sideNodes(NULL)
{
    
  size_t startPos       = a_fileHasMeshData.find_last_of(".");
  std::string suffix;

  if (!(startPos == std::string::npos))
  {
     suffix  = a_fileHasMeshData.substr(startPos+1,2);
  }

  if(suffix=="nc")
  {
	m_dataFilePtr=new NetcdfSchismOutput(a_fileHasMeshData);
  }
  else
  {
	m_dataFilePtr=new NativeSchismOutput(a_fileHasMeshData);
  }
  
  if (!(m_dataFilePtr->is_valid()))
  {
    m_valid_provider = false;
	return;
  }
 
  loadMeshDim();
  genElementSide();
  m_valid_provider = true;                                                           
}

SCHISMMeshProvider::~SCHISMMeshProvider()
{
   if (m_dataFilePtr)
   {
	   delete m_dataFilePtr;
   }

   if (m_kbp00)
   {
	   delete m_kbp00;
   }

   if (m_layerSCoords)
   {
	   delete m_layerSCoords;
   }

   if (m_kbs)
   {
	   delete m_kbs;
   }

   if (m_kbe)
   {
	   delete m_kbe;
   }

   if (m_sideNodes)
   {
	   delete m_sideNodes;
   }
}

bool SCHISMMeshProvider::fillKbp00(int * a_cache) const
 {

	 for (int i=0;i<m_number_node;i++)
	 {
		 a_cache[i]= m_kbp00[i];
	 }
	 return true;

 }

bool SCHISMMeshProvider::fillKbs(int * a_cache) const
 {

	 for (int i=0;i<m_number_side;i++)
	 {
		 a_cache[i]= m_kbs[i];
	 }
	 return true;

 }

bool SCHISMMeshProvider::fillKbe(int * a_cache) const
 {

	 for (int i=0;i<m_number_element;i++)
	 {
		 a_cache[i]= m_kbe[i];
	 }
	 return true;

 }

void SCHISMMeshProvider::genElementSide()
{
	int * meshElementNodes = new int [(MeshConstants::MAX_NUM_NODE_PER_CELL+1)*m_number_element];


	fillMeshElement(meshElementNodes);

	m_number_side = meshSideNum( meshElementNodes,m_number_element,m_number_node);

	m_sideNodes = new int [m_number_side*2];
	meshSideNode(m_sideNodes,meshElementNodes,m_number_side,m_number_element,m_number_node);

	//fill m_kbs
	m_kbs = new int [m_number_side];

	for(int iSide =0; iSide<m_number_side; iSide++)
	{

		int node1 = m_sideNodes[iSide*2];
		int node2 = m_sideNodes[iSide*2+1];
		m_kbs[iSide]= m_kbp00[node1];
		if (m_kbp00[node2]<m_kbs[iSide])
		{
           m_kbs[iSide]=m_kbp00[node2];
		}
	}

	//fill m_kbe
	m_kbe = new int [m_number_element];
    int maxNodeInCell = MeshConstants::MAX_NUM_NODE_PER_CELL;
	for(int iEle =0; iEle<m_number_element; iEle++)
	{
	
		int numNode = meshElementNodes[iEle*(maxNodeInCell+1)];
		
		int kbe = m_kbp00[meshElementNodes[iEle*(maxNodeInCell+1)+1]-1];

        for (int iNode=1;iNode<numNode;iNode++)
		{
          int node=meshElementNodes[iEle*(maxNodeInCell+1)+1+iNode]-1;
		  int kbet= m_kbp00[node];
		  if (kbet<kbe)
		  {
			  kbe=kbet;
		  }
		}
	    m_kbe[iEle]=kbe;
	}

	delete meshElementNodes;

}

void SCHISMMeshProvider::loadMeshDim()
{

	
  SCHISMDim * dimNodePtr      = m_dataFilePtr->get_dim(MeshConstants::DIM_MESH_NODES);
 
  if (dimNodePtr->is_valid())  
    {
      m_number_node = dimNodePtr->size();
    }
  else
    {
       m_number_node=MeshConstants::INVALID_NUM;
	   return;
    }

   m_kbp00      = new int [m_number_node];
    
   retrieve1DVar( m_kbp00,
	              m_dataFilePtr,
                  MeshConstants::NODE_BOTTOM,
                  m_number_node);

  // SELFEDim constructor and destructor are private, no need to del.
  SCHISMDim * dimFacePtr      = m_dataFilePtr->get_dim(MeshConstants::DIM_MESH_FACES);
  
  if (dimFacePtr->is_valid())
    {
      m_number_element = dimFacePtr->size(); 
    }
  else
    {
      m_number_element = MeshConstants::INVALID_NUM;
	  return;
    }
  

  SCHISMDim * dimLayerPtr      = m_dataFilePtr->get_dim(MeshConstants::DIM_LAYERS);
  
  if (dimLayerPtr->is_valid())
    {
      m_number_layer          = dimLayerPtr->size();
    }
  else
    {
       m_number_layer = MeshConstants::INVALID_NUM;
	   return;
    }

    SCHISMVar * sVarPtr = m_dataFilePtr->get_var(MeshConstants::LAYER_SCOORD);
    if (!(sVarPtr->is_valid()))
      {
        
         return;
      }
    else
      {
        SCHISMAtt * hsAttPtr = sVarPtr->get_att(MeshConstants::HS);
        if (!(hsAttPtr->is_valid()))
          {
            
            return;
          }
        m_hs              = hsAttPtr->float_value(0);

        SCHISMAtt * hcAttPtr = sVarPtr->get_att(MeshConstants::HC);
        if (!(hcAttPtr->is_valid()))
          {
            
            return;
          }
        m_hc              = hcAttPtr->float_value(0);
        
        
        SCHISMAtt * thetabAttPtr = sVarPtr->get_att(MeshConstants::THETAB);
        if (!(thetabAttPtr->is_valid()))
          {
            
            return;
          }
        m_thetab             = thetabAttPtr->float_value(0);
        
        SCHISMAtt * thetafAttPtr = sVarPtr->get_att(MeshConstants::THETAF);
        if (!(thetafAttPtr->is_valid()))
          {
            
            return;
          }
        m_thetaf             = thetafAttPtr->float_value(0);

		m_layerSCoords = new float [m_number_layer];

		if (!sVarPtr->get( m_layerSCoords))
        {
          throw SCHISMFileException("fail to retrieve var "+MeshConstants::LAYER_SCOORD+" from data file "+m_dataFilePtr->file());
        }

      }  

  m_mesh_loaded = true;
}


// z is filled with 0
bool SCHISMMeshProvider::fillPointCoord2D(float * a_pointCoord,const int & a_timeStep) const
{
	if( m_mesh_loaded==false)
	{
	   return false;
	}
	else
	{
		
      float * nodeXPtr      = new float [m_number_node];
    
      float * nodeYPtr      = new float [m_number_node];
    

      retrieve1DVar( nodeXPtr,
	                   m_dataFilePtr,
                       MeshConstants::NODE_X,
                       m_number_node);

      retrieve1DVar( nodeYPtr,
	                   m_dataFilePtr,
                       MeshConstants::NODE_Y,
                       m_number_node);


	  for(int iNode=0;iNode < m_number_node; iNode++)
          {
            float x            =  nodeXPtr[iNode];
            float y            =  nodeYPtr[iNode];
            *a_pointCoord++    = x;
            *a_pointCoord++    = y;
            // must put a dummy z value as visit manaul example does
            *a_pointCoord++    = MeshConstants::DUMMY_ELEVATION;
           
          }

	  delete nodeXPtr;
	  delete nodeYPtr;
      
	}
	return true;
}

bool SCHISMMeshProvider::fillSideCenterCoord3D(float * a_pointCoord,const int & a_timeStep) const
{
	if( m_mesh_loaded==false)
	{
	   return false;
	}
	else
	{
		
      float * nodeXPtr      = new float [m_number_node];
    
      float * nodeYPtr      = new float [m_number_node];
    
	  float * xPtr = nodeXPtr;
	  float*  yPtr = nodeYPtr;

      retrieve1DVar( nodeXPtr,
	                 m_dataFilePtr,
                     MeshConstants::NODE_X,
                     m_number_node);

      retrieve1DVar( nodeYPtr,
	                 m_dataFilePtr,
                     MeshConstants::NODE_Y,
                     m_number_node);


	  float*           zPtr = new float [m_number_layer*m_number_node];
	  zcoords3D2(zPtr,a_timeStep);
	 

	  int * node_z_start_index  = new int [m_number_node];
	  int valid_var_size = 0;

      for(int iNode=0;iNode<m_number_node;iNode++)
      {
	    node_z_start_index[iNode]=valid_var_size;
	    valid_var_size+=m_number_layer-std::max(1,m_kbp00[iNode])+1;
      }
	  float half=0.5;
	   for (int iLayer= 0; iLayer<m_number_layer;iLayer++)
        {
		 
          for(int iSide=0;iSide <m_number_side; iSide++)
            {
		 

			  if (iLayer>=(std::max(1,m_kbs[iSide])-1))
			  {
 
                 int node1 = m_sideNodes[iSide*2];
		         int node2 = m_sideNodes[iSide*2+1];
                 float x1            =  nodeXPtr[node1];
                 float y1            =  nodeYPtr[node1];
			     float x2            =  nodeXPtr[node2];
                 float y2            =  nodeYPtr[node2];
                 *a_pointCoord++     = half*(x1+x2);
                 *a_pointCoord++     = half*(y1+y2);

				 int node1_z_start = node_z_start_index[node1];
				 int node2_z_start = node_z_start_index[node2];
				 int z_displace_node1 = iLayer-std::max(1,m_kbp00[node1])+1;
				 int z_displace_node2 = iLayer-std::max(1,m_kbp00[node2])+1;

				 // degenerated side node at 1
				 if(z_displace_node1<0)
				 {
					 z_displace_node1=0;
				 }
				  // degenerated side node at 2
				  if(z_displace_node2<0)
				 {
					 z_displace_node2=0;
				 }
		
				 float z1 = zPtr[node1_z_start+z_displace_node1];
				 float z2 = zPtr[node2_z_start+z_displace_node2];

                 *a_pointCoord++     = half*(z1+z2);
			  }

            }

        }



	  delete nodeXPtr;
	  delete nodeYPtr;
	  delete zPtr;
	  delete node_z_start_index;
      
	}
	return true;
}

bool SCHISMMeshProvider::fillSideFaceCenterCoord3D(float * a_pointCoord,const int & a_timeStep) const
{
	if( m_mesh_loaded==false)
	{
	   return false;
	}
	else
	{
		
      float * nodeXPtr      = new float [m_number_node];
    
      float * nodeYPtr      = new float [m_number_node];
    
	  float * xPtr = nodeXPtr;
	  float*  yPtr = nodeYPtr;

      retrieve1DVar( nodeXPtr,
	                 m_dataFilePtr,
                     MeshConstants::NODE_X,
                     m_number_node);

      retrieve1DVar( nodeYPtr,
	                 m_dataFilePtr,
                     MeshConstants::NODE_Y,
                     m_number_node);


	  float*           zPtr = new float [m_number_layer*m_number_node];
	  zcoords3D2(zPtr,a_timeStep);
	 

	  int * node_z_start_index  = new int [m_number_node];
	  int valid_var_size = 0;

      for(int iNode=0;iNode<m_number_node;iNode++)
      {
	    node_z_start_index[iNode]=valid_var_size;
	    valid_var_size+=m_number_layer-std::max(1,m_kbp00[iNode])+1;
      }

	  float half =0.5;
	  float quater =0.25;
	   for (int iLayer= 1; iLayer<m_number_layer;iLayer++)
        {
		 
          for(int iSide=0;iSide <m_number_side; iSide++)
            {
		 

			  if ((iLayer-1)>=(std::max(1,m_kbs[iSide])-1))
			  {
 
                 int node1 = m_sideNodes[iSide*2];
		         int node2 = m_sideNodes[iSide*2+1];
                 float x1            =  nodeXPtr[node1];
                 float y1            =  nodeYPtr[node1];
			     float x2            =  nodeXPtr[node2];
                 float y2            =  nodeYPtr[node2];
                 *a_pointCoord++     = half*(x1+x2);
                 *a_pointCoord++     = half*(y1+y2);

				 int node1_z_start = node_z_start_index[node1];
				 int node2_z_start = node_z_start_index[node2];
				 int z_displace_node1 = iLayer-std::max(1,m_kbp00[node1])+1;
				 int z_displace_node2 = iLayer-std::max(1,m_kbp00[node2])+1;

				 // degenerated side node at 1
				 if(z_displace_node1<0)
				 {
					 z_displace_node1=0;
				 }
				  // degenerated side node at 2
				  if(z_displace_node2<0)
				 {
					 z_displace_node2=0;
				 }
		
				 float z1 = zPtr[node1_z_start+z_displace_node1];
				 float z2 = zPtr[node2_z_start+z_displace_node2];
				 float z_temp = z1+z2;
				 z_displace_node1 = iLayer-std::max(1,m_kbp00[node1]);
				 z_displace_node2 = iLayer-std::max(1,m_kbp00[node2]);

				 // degenerated side node at 1
				 if(z_displace_node1<0)
				 {
					 z_displace_node1=0;
				 }
				  // degenerated side node at 2
				  if(z_displace_node2<0)
				 {
					 z_displace_node2=0;
				 }
		
				 z1 = zPtr[node1_z_start+z_displace_node1];
				 z2 = zPtr[node2_z_start+z_displace_node2];


                 *a_pointCoord++     = quater*(z1+z2+z_temp);
			  }

            }

        }



	  delete nodeXPtr;
	  delete nodeYPtr;
	  delete zPtr;
	  delete node_z_start_index;
      
	}
	return true;
}




// z is filled with 0
bool SCHISMMeshProvider::fillSideCenterCoord2D(float * a_pointCoord,const int & a_timeStep) const
{
	if( m_mesh_loaded==false)
	{
	   return false;
	}
	else
	{
		
      float * nodeXPtr      = new float [m_number_node];
    
      float * nodeYPtr      = new float [m_number_node];

	  float half =0.5;
    

      retrieve1DVar( nodeXPtr,
	                   m_dataFilePtr,
                       MeshConstants::NODE_X,
                       m_number_node);

      retrieve1DVar( nodeYPtr,
	                   m_dataFilePtr,
                       MeshConstants::NODE_Y,
                       m_number_node);


	  for(int iSide=0;iSide < m_number_side; iSide++)
          {
			int node1 = m_sideNodes[iSide*2];
		    int node2 = m_sideNodes[iSide*2+1];
            float x1            =  nodeXPtr[node1];
            float y1            =  nodeYPtr[node1];
			float x2            =  nodeXPtr[node2];
            float y2            =  nodeYPtr[node2];
            *a_pointCoord++    = half*(x1+x2);
            *a_pointCoord++    = half*(y1+y2);
            // must put a dummy z value as visit manaul example does
            *a_pointCoord++    = MeshConstants::DUMMY_ELEVATION;
           
          }

	  delete nodeXPtr;
	  delete nodeYPtr;
      
	}
	return true;
}

bool SCHISMMeshProvider::fillEleCenterCoord3D(float * a_pointCoord,const int & a_timeStep) const
{
	if( m_mesh_loaded==false)
	{
	   return false;
	}
	else
	{

	  int * meshElementNodes = new int [(MeshConstants::MAX_NUM_NODE_PER_CELL+1)*m_number_element];
	  fillMeshElement(meshElementNodes);
		
      float * nodeXPtr      = new float [m_number_node];
    
      float * nodeYPtr      = new float [m_number_node];
    
	  float * xPtr = nodeXPtr;
	  float*  yPtr = nodeYPtr;

      retrieve1DVar( nodeXPtr,
	                 m_dataFilePtr,
                     MeshConstants::NODE_X,
                     m_number_node);

      retrieve1DVar( nodeYPtr,
	                 m_dataFilePtr,
                     MeshConstants::NODE_Y,
                     m_number_node);


	  float*           zPtr = new float [m_number_layer*m_number_node];
	  zcoords3D2(zPtr,a_timeStep);
	 

	  int * node_z_start_index  = new int [m_number_node];
	  int valid_var_size = 0;

      for(int iNode=0;iNode<m_number_node;iNode++)
      {
	    node_z_start_index[iNode]=valid_var_size;
	    valid_var_size+=m_number_layer-std::max(1,m_kbp00[iNode])+1;
      }

	  int maxNodeInCell = MeshConstants::MAX_NUM_NODE_PER_CELL;

	   for (int iLayer= 0; iLayer<m_number_layer;iLayer++)
        {
		 
          for(int iEle=0;iEle <m_number_element; iEle++)
            {
		 
			  if (iLayer>=(std::max(1,m_kbe[iEle])-1))
			  {

				
			     int numNode = meshElementNodes[iEle*(maxNodeInCell+1)];
		       
				 float x_sum = 0.0;
				 float y_sum = 0.0;
				 float z_sum = 0.0;

                 for (int iNode=0;iNode<numNode;iNode++)
		         {  
                   int node=meshElementNodes[iEle*(maxNodeInCell+1)+1+iNode]-1;
		           int kbp= m_kbp00[node];
				   x_sum += nodeXPtr[iNode];
				   y_sum += nodeYPtr[iNode];
				   int node_z_start = node_z_start_index[node];

				   int z_displace_node = iLayer-std::max(1,kbp)+1;

				    // degenerated ele node
				   if(z_displace_node<0)
				   {
					 z_displace_node=0;
				   }
				   float z = zPtr[node_z_start+z_displace_node];
				   z_sum+=z;
		         }
 
               
				 *a_pointCoord++     = x_sum/numNode;
                 *a_pointCoord++     = y_sum/numNode;
                 *a_pointCoord++     = z_sum/numNode;
			  }

            }

        }
	  delete nodeXPtr;
	  delete nodeYPtr;
	  delete zPtr;
	  delete node_z_start_index;
	  delete meshElementNodes;
	}
	return true;
}


// z is filled with 0
bool SCHISMMeshProvider::fillEleCenterCoord2D(float * a_pointCoord,const int & a_timeStep) const
{
	if( m_mesh_loaded==false)
	{
	   return false;
	}
	else
	{
	
	  int * meshElementNodes = new int [(MeshConstants::MAX_NUM_NODE_PER_CELL+1)*m_number_element];
	  fillMeshElement(meshElementNodes);
      float * nodeXPtr      = new float [m_number_node];
    
      float * nodeYPtr      = new float [m_number_node];
    

      retrieve1DVar( nodeXPtr,
	                   m_dataFilePtr,
                       MeshConstants::NODE_X,
                       m_number_node);

      retrieve1DVar( nodeYPtr,
	                   m_dataFilePtr,
                       MeshConstants::NODE_Y,
                       m_number_node);
	  int maxNodeInCell = MeshConstants::MAX_NUM_NODE_PER_CELL;

      for(int iEle=0;iEle <m_number_element; iEle++)
      {		
		int numNode = meshElementNodes[iEle*(maxNodeInCell+1)];
		       
		float x_sum = 0.0;
		float y_sum = 0.0;
			
        for (int iNode=0;iNode<numNode;iNode++)
		{  
           int node=meshElementNodes[iEle*(maxNodeInCell+1)+1+iNode]-1;
		   x_sum += nodeXPtr[iNode];
		   y_sum += nodeYPtr[iNode];
		}
      
		*a_pointCoord++     = x_sum/numNode;
        *a_pointCoord++     = y_sum/numNode;
        *a_pointCoord++     = MeshConstants::DUMMY_ELEVATION;
               
		}


	  delete nodeXPtr;
	  delete nodeYPtr;
      delete meshElementNodes;
	}
	return true;
}


bool SCHISMMeshProvider::fillPointCoord3D(float * a_pointCoord,const int & a_timeStep) const
{
	if( m_mesh_loaded==false)
	{
	   return false;
	}
	else
	{
		
      float * nodeXPtr      = new float [m_number_node];
      float * nodeYPtr      = new float [m_number_node];
    
	  float * xPtr = nodeXPtr;
	  float*  yPtr = nodeYPtr;

      retrieve1DVar( nodeXPtr,
	                 m_dataFilePtr,
                     MeshConstants::NODE_X,
                     m_number_node);

      retrieve1DVar( nodeYPtr,
	                 m_dataFilePtr,
                    MeshConstants::NODE_Y,
                     m_number_node);


	  float*           zPtr = new float [m_number_layer*m_number_node];
      float*           zPtrTemp = zPtr;
	  zcoords3D(zPtr,a_timeStep);

	   for (int iLayer= 0; iLayer<m_number_layer;iLayer++)
        {
		 
          for(int iNode=0;iNode <m_number_node; iNode++)
            {
		 
              float x            = xPtr[iNode];
              float y            = yPtr[iNode];
			  if (iLayer>=(std::max(1,m_kbp00[iNode])-1))
			  {
                  *a_pointCoord++         = x;
                  *a_pointCoord++         = y;
                  *a_pointCoord++         = *zPtrTemp++;
			  }

            }

        }



	  delete nodeXPtr;
	  delete nodeYPtr;
	  delete zPtr;
      
	}
	return true;
}

bool  SCHISMMeshProvider::fillMeshElement(int * a_elementCache) const
{
	if (!m_mesh_loaded)
	{
		return false;
	}
	else
	{
       SCHISMVar * ncFacePtr          = m_dataFilePtr->get_var(MeshConstants::MESH_FACE_NODES);
       if (!(ncFacePtr->is_valid()))
      {
      
         throw SCHISMFileException("no mesh element in "+m_dataFilePtr->file());
      }
 

     if (!ncFacePtr->get(a_elementCache))
     {
        throw SCHISMFileException("fail to read mesh element in "+m_dataFilePtr->file());
     }
   }
   return true;
}

bool SCHISMMeshProvider::zcoords2D(float * a_zCachePtr,const int & a_timeStep) const
{
	 for(int iNode=0;iNode <m_number_node; iNode++)
        {
	
            *a_zCachePtr++         = MeshConstants::DUMMY_ELEVATION ;
        }

	 return true;
}

bool SCHISMMeshProvider::zcoords3D(float * a_zCachePtr,const int & a_timeStep) const
{
	
	 return false;
}

bool SCHISMMeshProvider::zcoords3D2(float * a_zCachePtr,const int & a_timeStep) const
{
	
	 return false;
}

bool SCHISMMeshProvider::zSideCenter2D(float * a_zCachePtr,const int & a_timeStep) const
{
	for(int iNode=0;iNode <m_number_side; iNode++)
        {
	
            *a_zCachePtr++         = MeshConstants::DUMMY_ELEVATION;
        }

	 return true;
}

 
bool SCHISMMeshProvider::zSideCenter3D(float * a_zCachePtr,const int & a_timeStep) const
{
	float*           zPtr = new float [m_number_layer*m_number_node];
    float*           zPtrTemp = zPtr;
	bool temp=zcoords3D2(zPtr,a_timeStep);
	if (!temp) return temp;

	int * node_z_start_index  = new int [m_number_node];
	int valid_var_size = 0;
	float half =0.5;

    for(int iNode=0;iNode<m_number_node;iNode++)
    {
	node_z_start_index[iNode]=valid_var_size;
	valid_var_size+=m_number_layer-std::max(1,m_kbp00[iNode])+1;
    }

	for (int iLayer= 0; iLayer<m_number_layer;iLayer++)
    {
		 
        for(int iSide=0;iSide <m_number_side; iSide++)
        {
		 

			if (iLayer>=(std::max(1,m_kbs[iSide])-1))
			{
 
                int node1 = m_sideNodes[iSide*2];
		        int node2 = m_sideNodes[iSide*2+1];
               

				int node1_z_start = node_z_start_index[node1];
				int node2_z_start = node_z_start_index[node2];
				int z_displace_node1 = iLayer-m_kbp00[node1]+1;
				int z_displace_node2 = iLayer-m_kbp00[node2]+1;

				// degenerated side node at 1
				if(z_displace_node1<0)
				{
					z_displace_node1=0;
				}
				// degenerated side node at 2
				if(z_displace_node2<0)
				{
					z_displace_node2=0;
				}
		
				float z1 = zPtr[node1_z_start+z_displace_node1];
				float z2 = zPtr[node2_z_start+z_displace_node2];

                *a_zCachePtr++     = half*(z1+z2);
			}

        }

    }



	 
	delete zPtr;
	delete node_z_start_index;
    
	return true;
}

bool SCHISMMeshProvider::zEleCenter2D(float * a_zCachePtr,const int & a_timeStep) const
{
	for(int i=0;i <m_number_element; i++)
        {
	
            *a_zCachePtr++         = MeshConstants::DUMMY_ELEVATION;
        }

	 return true;
}

 
bool SCHISMMeshProvider::zEleCenter3D(float * a_zCachePtr,const int & a_timeStep) const
{
	float*           zPtr = new float [m_number_layer*m_number_node];
    float*           zPtrTemp = zPtr;
	bool temp=zcoords3D2(zPtr,a_timeStep);
	if (!temp) return temp;

	int * node_z_start_index  = new int [m_number_node];
	int valid_var_size = 0;

    for(int iNode=0;iNode<m_number_node;iNode++)
    {
	node_z_start_index[iNode]=valid_var_size;
	valid_var_size+=m_number_layer-std::max(1,m_kbp00[iNode])+1;
    }

	int maxNodeInCell = MeshConstants::MAX_NUM_NODE_PER_CELL;
	int * meshElementNodes = new int [(MeshConstants::MAX_NUM_NODE_PER_CELL+1)*m_number_element];
	fillMeshElement(meshElementNodes);
    
	
	for (int iLayer= 0; iLayer<m_number_layer;iLayer++)
    {
		
	  for(int iEle=0;iEle <m_number_element; iEle++)
       {
		    
			if (iLayer>=(std::max(1,m_kbe[iEle])-1))
			{
			    int numNode = meshElementNodes[iEle*(maxNodeInCell+1)];
				float z_sum = 0.0;
				
                for (int iNode=0;iNode<numNode;iNode++)
		        {  
					int node=meshElementNodes[iEle*(maxNodeInCell+1)+1+iNode]-1;
					int kbp= m_kbp00[node];
				
					int node_z_start = node_z_start_index[node];

					int z_displace_node = iLayer-std::max(1,kbp)+1;

					// degenerated ele node
					if(z_displace_node<0)
					{
						z_displace_node=0;
					}
					float z = zPtr[node_z_start+z_displace_node];
					z_sum+=z;
		        }
                *a_zCachePtr++     = z_sum/numNode;
				
			}
		 }
    }

	delete zPtr;
	delete node_z_start_index;
    delete meshElementNodes;
	return true;
}


bool SCHISMMeshProvider::slayers(float * a_cache) const
{
	if  (m_mesh_loaded)
	{
		for(int i=0;i<m_number_layer;i++)
		{
			a_cache[i]= m_layerSCoords[i];
		}
		return true;
	}
	return false;
}

bool SCHISMMeshProvider::depth(float * a_cache) const
{
	return false;
}
