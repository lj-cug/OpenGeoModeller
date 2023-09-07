#include "SCHISMMeshProvider10.h"
#include "MeshConstants10.h"
#include "SCHISMFileUtil10.h"
#include "NetcdfSCHISMOutput10.h"
#include "SchismGeometry10.h"
#include <algorithm>




SCHISMMeshProvider10::SCHISMMeshProvider10(const std::string & a_fileHasMeshData):MeshProvider10(a_fileHasMeshData),
	                                                                          m_kbp00(NULL),
																			  m_layerSCoords(NULL),
                                                                              m_kbs(NULL),
                                                                              m_kbe(NULL),
                                                                              m_sideNodes(NULL),
																			  m_kbp_ele_filled(false),
																			  m_kbp_side_filled(false),
																			  m_kbp_node_filled(false),
																			  m_node_neighbor_ele(NULL),
																			  m_side_neighbor_ele(NULL),
																			  m_node_neighbor_ele_filled(false),
																			  m_side_neighbor_ele_filled(false),
																			  m_max_ele_at_node(0)

{
    
  
  m_dataFilePtr=new NetcdfSchismOutput10(a_fileHasMeshData);

  if (!(m_dataFilePtr->is_valid()))
  {
    m_valid_provider = false;
	return;
  }
 
  bool temp=loadMeshDim();

  if (!temp)
  {
	  m_valid_provider=false;
	  return;
  }

  m_valid_provider = loadSide();                                                           
}

SCHISMMeshProvider10::~SCHISMMeshProvider10()
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

   if(m_node_neighbor_ele)
   {
	   delete m_node_neighbor_ele;
   }
   if(m_side_neighbor_ele)
   {
	   delete m_side_neighbor_ele;
   }


}

SCHISMFile10* SCHISMMeshProvider10::get_mesh_data_ptr() const
{
	return m_dataFilePtr;
}

bool SCHISMMeshProvider10::fillKbp00(int * a_cache,const int & a_timeStep) const
 {

	 m_dataFilePtr->get_node_bottom(a_cache,a_timeStep);
	 return true;

 }

bool SCHISMMeshProvider10::fillKbs(int * a_cache,const int & a_timeStep) const
 {

	 m_dataFilePtr->get_edge_bottom(a_cache,a_timeStep);
	 return true;

 }

bool SCHISMMeshProvider10::fillKbe(int * a_cache,const int & a_timeStep) const
 {

	m_dataFilePtr->get_face_bottom(a_cache,a_timeStep);
    return true;

 }

bool SCHISMMeshProvider10::update_bottom_layer(const int & a_timeStep) 
{
	
	return m_dataFilePtr->update_bottom_index(a_timeStep);
}

bool SCHISMMeshProvider10::loadSide()
{
	
	m_sideNodes = new long [m_number_side*2];

	if (m_dataFilePtr->inquire_var(MeshConstants10::EDGE_NODE))
	{
		SCHISMVar10 * side_node_ptr = m_dataFilePtr->get_var(MeshConstants10::EDGE_NODE);

		if (!(side_node_ptr->get(m_sideNodes)))
		{
			throw SCHISMFileException10("fail to retrieve dim " + MeshConstants10::EDGE_NODE + " from data file " + m_dataFilePtr->file());
			return false;
		}
		//decreae node id by one, for schism id is 1 based
		for (long i = 0; i < m_number_side * 2; i++)
		{
			m_sideNodes[i]--;
		}
		return true;
	}
	else //figure out from mesh 
	{
		long * mesh_nodes = new long[m_number_element*(MeshConstants10::MAX_NUM_NODE_PER_CELL + 1)];
		this->fillMeshElement(mesh_nodes);           
		meshSideNode(m_sideNodes,
			mesh_nodes,
			m_number_side,
			m_number_element,
			m_number_node);
		delete mesh_nodes;
		return true;
	}
 


}
void  SCHISMMeshProvider10::fill_ele_dry_wet(int*  &a_ele_dry_wet, const int& a_step)
{
	std::string SCHISMVarName = MeshConstants10::ELEM_DRYWET;
	if (!(m_dataFilePtr->inquire_var(SCHISMVarName)))
	{
		SCHISMVarName = MeshConstants10::ELEM_DRYWET2;
		if (!(m_dataFilePtr->inquire_var(SCHISMVarName)))
		{
			throw SCHISMFileException10("no ele dry wet flag in "+ m_dataFilePtr->file());
		}


	}
	SCHISMVar10* SCHISMVarPtr = NULL;
	try
	{
		SCHISMVarPtr = m_dataFilePtr->get_var(SCHISMVarName);
	}
	catch (...)
	{
		throw SCHISMFileException10("no " + SCHISMVarName + " in " + m_dataFilePtr->file());
	}

	SCHISMVarPtr->set_cur(a_step);
	if (!(SCHISMVarPtr->get(a_ele_dry_wet)))
	{
		stringstream msgStream(stringstream::out);
		msgStream << "Fail to retrieve " << SCHISMVarName << " at step " << a_step;
		throw  SCHISMFileException10(msgStream.str());
	}
}
 void  SCHISMMeshProvider10::fill_node_dry_wet(int* &a_node_dry_wet,int* a_ele_dry_wet)
 {

	 long * mesh_nodes =new long[m_number_element*(MeshConstants10::MAX_NUM_NODE_PER_CELL+1)];
	 this->fillMeshElement(mesh_nodes);
	 int max_node_in_cell=MeshConstants10::MAX_NUM_NODE_PER_CELL;
	 long invalid_id = MeshConstants10::INVALID_NUM;
	
	 
	 if (!m_node_neighbor_ele_filled)
	 {

		 int max_ele_at_node=0;
		 meshSideNodeNeiborEle(mesh_nodes,
                               m_node_neighbor_ele,
							   m_side_neighbor_ele,
							   max_ele_at_node,
							   m_number_element,
				               m_number_node,
							   m_number_side);

      m_node_neighbor_ele_filled=true;
	  m_side_neighbor_ele_filled=true;
	  m_max_ele_at_node=max_ele_at_node;
	 }

	 for(long inode=0;inode<m_number_node;inode++)
	 {
		 bool all_ele_dry=true;

		 for(int iele=0;iele<m_max_ele_at_node;iele++)
		 {
			 long ele_global_id=m_node_neighbor_ele[m_max_ele_at_node*inode+iele];
			 if (!(ele_global_id==invalid_id))
			 {
				 if(a_ele_dry_wet[ele_global_id]==MeshConstants10::WET_FLAG)
				 {
					 all_ele_dry=false;
					 break;
				 }
			 }

		 }

		 if (all_ele_dry)
		 {
			 a_node_dry_wet[inode]=MeshConstants10::DRY_FLAG;
			
		 }
		 else
		 {
			 a_node_dry_wet[inode]=MeshConstants10::WET_FLAG;
		 }
	 }
	
	 delete mesh_nodes;
	
 } 

 void  SCHISMMeshProvider10::fill_side_dry_wet(int* &a_side_dry_wet,int* a_ele_dry_wet)
{
	 long * mesh_nodes =new long[m_number_element*(MeshConstants10::MAX_NUM_NODE_PER_CELL+1)];
	 this->fillMeshElement(mesh_nodes);
	 long invalid_id = MeshConstants10::INVALID_NUM;
	
	 int num_side_neighbor_ele=2;

	 if (!m_side_neighbor_ele_filled)
	 {

		 int max_ele_at_node=0;
		 meshSideNodeNeiborEle(mesh_nodes,
                               m_node_neighbor_ele,
							   m_side_neighbor_ele,
							   max_ele_at_node,
							   m_number_element,
				               m_number_node,
							   m_number_side);

      m_node_neighbor_ele_filled=true;
	  m_side_neighbor_ele_filled=true;
	  m_max_ele_at_node=max_ele_at_node;
	 }
	 
	 for(long iside=0;iside<m_number_side;iside++)
	 {
		 bool all_ele_dry=true;

		 for(int iele=0;iele<num_side_neighbor_ele;iele++)
		 {
			 long ele_global_id=m_side_neighbor_ele[num_side_neighbor_ele*iside+iele];
			 if (!(ele_global_id==invalid_id))
			 {
				 if(a_ele_dry_wet[ele_global_id]==MeshConstants10::WET_FLAG)
				 {
					 all_ele_dry=false;
					 break;
				 }
			 }

		 }

		 if (all_ele_dry)
		 {
			 a_side_dry_wet[iside]=MeshConstants10::DRY_FLAG;
		 }
		 else
		 {
			 a_side_dry_wet[iside]=MeshConstants10::WET_FLAG;
		 }
	 }
	
	 delete mesh_nodes;
}

bool SCHISMMeshProvider10::loadMeshDim()
{

	
  SCHISMDim10 * dimNodePtr      = m_dataFilePtr->get_dim(MeshConstants10::DIM_MESH_NODES);
 
  if (dimNodePtr->is_valid())  
    {
      m_number_node = dimNodePtr->size();
    }
  else
    {
       m_number_node=MeshConstants10::INVALID_NUM;
	   throw SCHISMFileException10("fail to retrieve dim "+MeshConstants10::DIM_MESH_NODES+" from data file "+m_dataFilePtr->file());
	   return false;
    }
   

  // SELFEDim constructor and destructor are private, no need to del.
  SCHISMDim10 * dimFacePtr      = m_dataFilePtr->get_dim(MeshConstants10::DIM_MESH_FACES);
  
  if (dimFacePtr->is_valid())
    {
      m_number_element = dimFacePtr->size(); 
    }
  else
    {
      m_number_element = MeshConstants10::INVALID_NUM;
	  throw SCHISMFileException10("fail to retrieve dim "+MeshConstants10::DIM_MESH_FACES+" from data file "+m_dataFilePtr->file());
	  return false;
    }
  

  // SELFEDim constructor and destructor are private, no need to del.
  SCHISMDim10 * dimSidePtr      = m_dataFilePtr->get_dim(MeshConstants10::DIM_MESH_EDGES);
  
  if (dimSidePtr->is_valid())
    {
      m_number_side = dimSidePtr->size(); 
    }
  else
    {
      m_number_side = MeshConstants10::INVALID_NUM;
	  throw SCHISMFileException10("fail to retrieve dim "+MeshConstants10::DIM_MESH_EDGES+" from data file "+m_dataFilePtr->file());
	  return false;
    }

  SCHISMDim10 * dimLayerPtr      = m_dataFilePtr->get_dim(MeshConstants10::DIM_LAYERS);
  
  if (dimLayerPtr->is_valid())
    {
      m_number_layer          = dimLayerPtr->size();
    }
  else
    {
       m_number_layer = MeshConstants10::INVALID_NUM;
	   throw SCHISMFileException10("fail to retrieve dim "+MeshConstants10::DIM_LAYERS+" from data file "+m_dataFilePtr->file());
	   return false;
    }

  if (m_dataFilePtr->inquire_var(MeshConstants10::LAYER_SCOORD))
  {
	  SCHISMVar10 * sVarPtr = m_dataFilePtr->get_var(MeshConstants10::LAYER_SCOORD);

	  if (!(sVarPtr->is_valid()))
	  {
		  throw SCHISMFileException10("fail to retrieve dim " + MeshConstants10::LAYER_SCOORD + " from data file " + m_dataFilePtr->file());
		  return false;
	  }
	  else
	  {
		  SCHISMDim10 * dim_sigma_layers = sVarPtr->get_dim(0);
		  int num_sigma_layers = dim_sigma_layers->size();
		  SCHISMAtt10 * hsAttPtr = sVarPtr->get_att(MeshConstants10::HS);
		  if (!(hsAttPtr->is_valid()))
		  {
			  throw SCHISMFileException10("fail to retrieve dim " + MeshConstants10::HS + " from data file " + m_dataFilePtr->file());
			  return false;
		  }
		  m_hs = hsAttPtr->float_value(0);

		  SCHISMAtt10 * hcAttPtr = sVarPtr->get_att(MeshConstants10::HC);
		  if (!(hcAttPtr->is_valid()))
		  {
			  throw SCHISMFileException10("fail to retrieve dim " + MeshConstants10::HC + " from data file " + m_dataFilePtr->file());
			  return false;
		  }
		  m_hc = hcAttPtr->float_value(0);


		  SCHISMAtt10 * thetabAttPtr = sVarPtr->get_att(MeshConstants10::THETAB);
		  if (!(thetabAttPtr->is_valid()))
		  {
			  throw SCHISMFileException10("fail to retrieve dim " + MeshConstants10::THETAB + " from data file " + m_dataFilePtr->file());
			  return false;
		  }
		  m_thetab = thetabAttPtr->float_value(0);

		  SCHISMAtt10 * thetafAttPtr = sVarPtr->get_att(MeshConstants10::THETAF);
		  if (!(thetafAttPtr->is_valid()))
		  {
			  throw SCHISMFileException10("fail to retrieve dim " + MeshConstants10::THETAF + " from data file " + m_dataFilePtr->file());
			  return false;
		  }
		  m_thetaf = thetafAttPtr->float_value(0);

		  m_layerSCoords = new float[m_number_layer];

		  float * stemp = new float[num_sigma_layers];
		  if (!sVarPtr->get(stemp))
		  {
			  throw SCHISMFileException10("fail to retrieve var " + MeshConstants10::LAYER_SCOORD + " from data file " + m_dataFilePtr->file());
		  }

		  int sigma_layer_start_id = m_number_layer - num_sigma_layers;
		  int itemp = 0;
		  for (int iLayer = sigma_layer_start_id; iLayer < m_number_layer; iLayer++)
		  {
			  m_layerSCoords[iLayer] = stemp[itemp];
			  itemp++;
		  }
		  delete stemp;
	  }
  }
  m_mesh_loaded = true;
  return true;
}


// z is filled with 0
bool SCHISMMeshProvider10::fillPointCoord2D(float * a_pointCoord,const int & a_timeStep) const
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
                       MeshConstants10::NODE_X,
                       m_number_node);

      retrieve1DVar( nodeYPtr,
	                   m_dataFilePtr,
                       MeshConstants10::NODE_Y,
                       m_number_node);


	  for(long iNode=0;iNode < m_number_node; iNode++)
          {
            float x            =  nodeXPtr[iNode];
            float y            =  nodeYPtr[iNode];
            *a_pointCoord++    = x;
            *a_pointCoord++    = y;
            // must put a dummy z value as visit manaul example does
            *a_pointCoord++    = MeshConstants10::DUMMY_ELEVATION;
           
          }

	  delete nodeXPtr;
	  delete nodeYPtr;
      
	}
	return true;
}

bool SCHISMMeshProvider10::fillPointCoord2D(double * a_pointCoord,const int & a_timeStep) const
{
	if( m_mesh_loaded==false)
	{
	   return false;
	}
	else
	{
		
      double * nodeXPtr      = new double [m_number_node];
    
      double * nodeYPtr      = new double [m_number_node];
    

      retrieve1DVar( nodeXPtr,
	                   m_dataFilePtr,
                       MeshConstants10::NODE_X,
                       m_number_node);

      retrieve1DVar( nodeYPtr,
	                   m_dataFilePtr,
                       MeshConstants10::NODE_Y,
                       m_number_node);


	  for(long iNode=0;iNode < m_number_node; iNode++)
          {
            double x            =  nodeXPtr[iNode];
            double y            =  nodeYPtr[iNode];
            *a_pointCoord++    = x;
            *a_pointCoord++    = y;
            // must put a dummy z value as visit manaul example does
            *a_pointCoord++    = MeshConstants10::DUMMY_ELEVATION;
           
          }

	  delete nodeXPtr;
	  delete nodeYPtr;
      
	}
	return true;
}

bool SCHISMMeshProvider10::fillSideCenterCoord3D(float * a_pointCoord,const int & a_timeStep) const
{
	//if( m_mesh_loaded==false)
	//{
	//   return false;
	//}
	//else
	//{
		
      float * nodeXPtr      = new float [m_number_node];
    
      float * nodeYPtr      = new float [m_number_node];
    
	  float * xPtr = nodeXPtr;
	  float*  yPtr = nodeYPtr;

      retrieve1DVar( nodeXPtr,
	                 m_dataFilePtr,
                     MeshConstants10::NODE_X,
                     m_number_node);

      retrieve1DVar( nodeYPtr,
	                 m_dataFilePtr,
                     MeshConstants10::NODE_Y,
                     m_number_node);


	  float*           zPtr = new float [m_number_layer*m_number_node];
	  zcoords3D2(zPtr,a_timeStep);
	 

	  long * node_z_start_index  = new long [m_number_node];
	  long valid_var_size = 0;
	  int * kbp00 = new int [m_number_node];
	  fillKbp00(kbp00,a_timeStep);
	   int * kbs = new int [m_number_side];
	  fillKbs(kbs,a_timeStep);

      for(long iNode=0;iNode<m_number_node;iNode++)
      {
	    node_z_start_index[iNode]=valid_var_size;
	    valid_var_size+=(m_number_layer-std::max(1,kbp00[iNode])+1);
      }
	  float half=0.5;
	  
	   for (int iLayer= 0; iLayer<m_number_layer;iLayer++)
        {
		 
          for(long iSide=0;iSide <m_number_side; iSide++)
            {
		 

			  if (iLayer>=(std::max(1,kbs[iSide])-1))
			  {
 
                 long node1 = m_sideNodes[iSide*2];
		         long node2 = m_sideNodes[iSide*2+1];
                 float x1            =  nodeXPtr[node1];
                 float y1            =  nodeYPtr[node1];
			     float x2            =  nodeXPtr[node2];
                 float y2            =  nodeYPtr[node2];
                 *a_pointCoord++     = half*(x1+x2);
                 *a_pointCoord++     = half*(y1+y2);

				 long node1_z_start = node_z_start_index[node1];
				 long node2_z_start = node_z_start_index[node2];
				 int z_displace_node1 = iLayer-std::max(1,kbp00[node1])+1;
				 int z_displace_node2 = iLayer-std::max(1,kbp00[node2])+1;

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
	  delete kbp00;
	  delete kbs;
      
	//}
	return true;
}


bool SCHISMMeshProvider10::fillSideCenterCoord3D(double * a_pointCoord,const int & a_timeStep) const
{
	//if( m_mesh_loaded==false)
	//{
	//   return false;
	//}
	//else
	//{
		
      double * nodeXPtr      = new double [m_number_node];
    
      double * nodeYPtr      = new double [m_number_node];
    
	  double * xPtr = nodeXPtr;
	  double*  yPtr = nodeYPtr;

      retrieve1DVar( nodeXPtr,
	                 m_dataFilePtr,
                     MeshConstants10::NODE_X,
                     m_number_node);

      retrieve1DVar( nodeYPtr,
	                 m_dataFilePtr,
                     MeshConstants10::NODE_Y,
                     m_number_node);


	  float*           zPtr = new float [m_number_layer*m_number_node];
	  zcoords3D2(zPtr,a_timeStep);
	 

	  long * node_z_start_index  = new long [m_number_node];
	  long valid_var_size = 0;
	  int * kbp00 = new int [m_number_node];
	  fillKbp00(kbp00,a_timeStep);
	   int * kbs = new int [m_number_side];
	  fillKbs(kbs,a_timeStep);
	  
      for(long iNode=0;iNode<m_number_node;iNode++)
      {
	    node_z_start_index[iNode]=valid_var_size;
	    valid_var_size+=(m_number_layer-std::max(1,kbp00[iNode])+1);
      }
	  float half=0.5;
	  
	   for (int iLayer= 0; iLayer<m_number_layer;iLayer++)
        {
		 
          for(long iSide=0;iSide <m_number_side; iSide++)
            {
		 

			  if (iLayer>=(std::max(1,kbs[iSide])-1))
			  {
 
                 long node1 = m_sideNodes[iSide*2];
		         long node2 = m_sideNodes[iSide*2+1];
                 double x1            =  nodeXPtr[node1];
                 double y1            =  nodeYPtr[node1];
			     double x2            =  nodeXPtr[node2];
                 double y2            =  nodeYPtr[node2];
                 *a_pointCoord++     = half*(x1+x2);
                 *a_pointCoord++     = half*(y1+y2);

				 long node1_z_start = node_z_start_index[node1];
				 long node2_z_start = node_z_start_index[node2];
				 int z_displace_node1 = iLayer-std::max(1,kbp00[node1])+1;
				 int z_displace_node2 = iLayer-std::max(1,kbp00[node2])+1;

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
	  delete kbp00;
	  delete kbs;
      
	//}
	return true;
}

bool SCHISMMeshProvider10::fillSideFaceCenterCoord3D(float * a_pointCoord,const int & a_timeStep) const
{
	/*if( m_mesh_loaded==false)
	{
	   return false;
	}
	else
	{*/
		
      float * nodeXPtr      = new float [m_number_node];
    
      float * nodeYPtr      = new float [m_number_node];
    
	  float * xPtr = nodeXPtr;
	  float*  yPtr = nodeYPtr;

      retrieve1DVar( nodeXPtr,
	                 m_dataFilePtr,
                     MeshConstants10::NODE_X,
                     m_number_node);

      retrieve1DVar( nodeYPtr,
	                 m_dataFilePtr,
                     MeshConstants10::NODE_Y,
                     m_number_node);


	  float*           zPtr = new float [m_number_layer*m_number_node];
	  zcoords3D2(zPtr,a_timeStep);
	 
	  int * kbp00 = new int [m_number_node];
	  fillKbp00(kbp00,a_timeStep);
	  int * kbs= new int [m_number_side];
	  fillKbs(kbs,a_timeStep);

	  long * node_z_start_index  = new long [m_number_node];
	  long valid_var_size = 0;

      for(long iNode=0;iNode<m_number_node;iNode++)
      {
	    node_z_start_index[iNode]=valid_var_size;
	    valid_var_size+=m_number_layer-std::max(1,kbp00[iNode])+1;
      }

	  double half =0.5;
	  double quater =0.25;
	   for (int iLayer= 1; iLayer<m_number_layer;iLayer++)
        {
		 
          for(long iSide=0;iSide <m_number_side; iSide++)
            {
		 

			  if ((iLayer-1)>=(std::max(1,kbs[iSide])-1))
			  {
 
                 long node1 = m_sideNodes[iSide*2];
		         long node2 = m_sideNodes[iSide*2+1];
                 float x1            =  nodeXPtr[node1];
                 float y1            =  nodeYPtr[node1];
			     float x2            =  nodeXPtr[node2];
                 float y2            =  nodeYPtr[node2];
                 *a_pointCoord++     = half*(x1+x2);
                 *a_pointCoord++     = half*(y1+y2);

				 long node1_z_start = node_z_start_index[node1];
				 long node2_z_start = node_z_start_index[node2];
				 int z_displace_node1 = iLayer-std::max(1,kbp00[node1])+1;
				 int z_displace_node2 = iLayer-std::max(1,kbp00[node2])+1;

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
				 z_displace_node1 = iLayer-std::max(1,kbp00[node1]);
				 z_displace_node2 = iLayer-std::max(1,kbp00[node2]);

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
	  delete kbp00;
	  delete kbs;
      
	//}
	return true;
}

bool SCHISMMeshProvider10::fillSideFaceCenterCoord3D(double * a_pointCoord,const int & a_timeStep) const
{
	/*if( m_mesh_loaded==false)
	{
	   return false;
	}
	else
	{*/
		
      double * nodeXPtr      = new double [m_number_node];
    
      double * nodeYPtr      = new double [m_number_node];
    
	  double * xPtr = nodeXPtr;
	  double*  yPtr = nodeYPtr;

      retrieve1DVar( nodeXPtr,
	                 m_dataFilePtr,
                     MeshConstants10::NODE_X,
                     m_number_node);

      retrieve1DVar( nodeYPtr,
	                 m_dataFilePtr,
                     MeshConstants10::NODE_Y,
                     m_number_node);


	  float*           zPtr = new float [m_number_layer*m_number_node];
	  zcoords3D2(zPtr,a_timeStep);
	 
	  int * kbp00 = new int [m_number_node];
	  fillKbp00(kbp00,a_timeStep);
	  int * kbs= new int [m_number_side];
	  fillKbs(kbs,a_timeStep);

	  long * node_z_start_index  = new long [m_number_node];
	  long valid_var_size = 0;

      for(long iNode=0;iNode<m_number_node;iNode++)
      {
	    node_z_start_index[iNode]=valid_var_size;
	    valid_var_size+=m_number_layer-std::max(1,kbp00[iNode])+1;
      }

	  double half =0.5;
	  double quater =0.25;
	   for (int iLayer= 1; iLayer<m_number_layer;iLayer++)
        {
		 
          for(long iSide=0;iSide <m_number_side; iSide++)
            {
		 

			  if ((iLayer-1)>=(std::max(1,kbs[iSide])-1))
			  {
 
                 long node1 = m_sideNodes[iSide*2];
		         long node2 = m_sideNodes[iSide*2+1];
                 double x1            =  nodeXPtr[node1];
                 double y1            =  nodeYPtr[node1];
			     double x2            =  nodeXPtr[node2];
                 double y2            =  nodeYPtr[node2];
                 *a_pointCoord++     = half*(x1+x2);
                 *a_pointCoord++     = half*(y1+y2);

				 long node1_z_start = node_z_start_index[node1];
				 long node2_z_start = node_z_start_index[node2];
				 int z_displace_node1 = iLayer-std::max(1,kbp00[node1])+1;
				 int z_displace_node2 = iLayer-std::max(1,kbp00[node2])+1;

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
				 z_displace_node1 = iLayer-std::max(1,kbp00[node1]);
				 z_displace_node2 = iLayer-std::max(1,kbp00[node2]);

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
	  delete kbp00;
	  delete kbs;
      
	//}
	return true;
}


// z is filled with 0
bool SCHISMMeshProvider10::fillSideCenterCoord2D(float * a_pointCoord,const int & a_timeStep) const
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
                       MeshConstants10::NODE_X,
                       m_number_node);

      retrieve1DVar( nodeYPtr,
	                   m_dataFilePtr,
                       MeshConstants10::NODE_Y,
                       m_number_node);

	  
	  for(long iSide=0;iSide < m_number_side; iSide++)
          {
			long node1 = m_sideNodes[iSide*2];
		    long node2 = m_sideNodes[iSide*2+1];
            float x1            =  nodeXPtr[node1];
            float y1            =  nodeYPtr[node1];
			float x2            =  nodeXPtr[node2];
            float y2            =  nodeYPtr[node2];
            *a_pointCoord++    = half*(x1+x2);
            *a_pointCoord++    = half*(y1+y2);
            // must put a dummy z value as visit manaul example does
            *a_pointCoord++    = MeshConstants10::DUMMY_ELEVATION;
           
          }

	  delete nodeXPtr;
	  delete nodeYPtr;
      
	}
	return true;
}

// z is filled with 0
bool SCHISMMeshProvider10::fillSideCenterCoord2D(double * a_pointCoord,const int & a_timeStep) const
{
	if( m_mesh_loaded==false)
	{
	   return false;
	}
	else
	{
		
      double * nodeXPtr      = new double [m_number_node];
    
      double * nodeYPtr      = new double [m_number_node];

	  double half =0.5;
    

      retrieve1DVar( nodeXPtr,
	                   m_dataFilePtr,
                       MeshConstants10::NODE_X,
                       m_number_node);

      retrieve1DVar( nodeYPtr,
	                   m_dataFilePtr,
                       MeshConstants10::NODE_Y,
                       m_number_node);

	 
	  for(long iSide=0;iSide < m_number_side; iSide++)
          {
			long node1 = m_sideNodes[iSide*2];
		    long node2 = m_sideNodes[iSide*2+1];
            double x1            =  nodeXPtr[node1];
            double y1            =  nodeYPtr[node1];
			double x2            =  nodeXPtr[node2];
            double y2            =  nodeYPtr[node2];

            *a_pointCoord++    = half*(x1+x2);
            *a_pointCoord++    = half*(y1+y2);
            // must put a dummy z value as visit manaul example does
            *a_pointCoord++    = MeshConstants10::DUMMY_ELEVATION;
           
          }

	  delete nodeXPtr;
	  delete nodeYPtr;
	  
      
	}
	return true;
}

bool SCHISMMeshProvider10::fillEleCenterCoord3D(float * a_pointCoord,const int & a_timeStep) const
{
	/*if( m_mesh_loaded==false)
	{
	   return false;
	}
	else
	{*/

	  long * meshElementNodes = new long [(MeshConstants10::MAX_NUM_NODE_PER_CELL+1)*m_number_element];
	  fillMeshElement(meshElementNodes);
		
      float * nodeXPtr      = new float [m_number_node];
    
      float * nodeYPtr      = new float [m_number_node];
    
	  float * xPtr = nodeXPtr;
	  float*  yPtr = nodeYPtr;

      retrieve1DVar( nodeXPtr,
	                 m_dataFilePtr,
                     MeshConstants10::NODE_X,
                     m_number_node);

      retrieve1DVar( nodeYPtr,
	                 m_dataFilePtr,
                     MeshConstants10::NODE_Y,
                     m_number_node);


	  float*           zPtr = new float [m_number_layer*m_number_node];
	  zcoords3D2(zPtr,a_timeStep);
	  int * kbp00 = new int [m_number_node];
	  fillKbp00(kbp00,a_timeStep);
	  int * kbe = new int [m_number_element];
	  fillKbe(kbe,a_timeStep);
	 

	  long * node_z_start_index  = new long [m_number_node];
	  long valid_var_size = 0;

      for(long iNode=0;iNode<m_number_node;iNode++)
      {
	    node_z_start_index[iNode]=valid_var_size;
	    valid_var_size+=m_number_layer-std::max(1,kbp00[iNode])+1;
      }

	  int max_node_in_cell = MeshConstants10::MAX_NUM_NODE_PER_CELL;

	   for (int iLayer= 0; iLayer<m_number_layer;iLayer++)
        {
		 
          for(long iEle=0;iEle <m_number_element; iEle++)
            {
		 
			  if (iLayer>=(std::max(1,kbe[iEle])-1))
			  {

				
			     int numNode = meshElementNodes[iEle*(max_node_in_cell+1)];
		       
				 float x_sum = 0.0;
				 float y_sum = 0.0;
				 float z_sum = 0.0;

                 for (int iNode=0;iNode<numNode;iNode++)
		         {  
                   long node=meshElementNodes[iEle*(max_node_in_cell+1)+1+iNode]-1;
		           int kbp= m_kbp00[node];
				   x_sum += nodeXPtr[iNode];
				   y_sum += nodeYPtr[iNode];
				   long node_z_start = node_z_start_index[node];

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
	//}
	return true;
}

bool SCHISMMeshProvider10::fillEleCenterCoord3D(double * a_pointCoord,const int & a_timeStep) const
{
	/*if( m_mesh_loaded==false)
	{
	   return false;
	}
	else
	{*/

	  long * meshElementNodes = new long [(MeshConstants10::MAX_NUM_NODE_PER_CELL+1)*m_number_element];
	  fillMeshElement(meshElementNodes);
		
      double * nodeXPtr      = new double [m_number_node];
    
      double * nodeYPtr      = new double [m_number_node];
    
	  double * xPtr = nodeXPtr;
	  double*  yPtr = nodeYPtr;

      retrieve1DVar( nodeXPtr,
	                 m_dataFilePtr,
                     MeshConstants10::NODE_X,
                     m_number_node);

      retrieve1DVar( nodeYPtr,
	                 m_dataFilePtr,
                     MeshConstants10::NODE_Y,
                     m_number_node);


	  float*           zPtr = new float [m_number_layer*m_number_node];
	  zcoords3D2(zPtr,a_timeStep);
	  int * kbp00 = new int [m_number_node];
	  fillKbp00(kbp00,a_timeStep);
	  int * kbe = new int [m_number_element];
	  fillKbe(kbe,a_timeStep);
	 

	  long * node_z_start_index  = new long [m_number_node];
	  long valid_var_size = 0;

      for(long iNode=0;iNode<m_number_node;iNode++)
      {
	    node_z_start_index[iNode]=valid_var_size;
	    valid_var_size+=m_number_layer-std::max(1,kbp00[iNode])+1;
      }

	  int max_node_in_cell = MeshConstants10::MAX_NUM_NODE_PER_CELL;

	   for (int iLayer= 0; iLayer<m_number_layer;iLayer++)
        {
		 
          for(long iEle=0;iEle <m_number_element; iEle++)
            {
		 
			  if (iLayer>=(std::max(1,kbe[iEle])-1))
			  {

				
			     int numNode = meshElementNodes[iEle*(max_node_in_cell+1)];
		       
				 double x_sum = 0.0;
				 double y_sum = 0.0;
				 double z_sum = 0.0;

                 for (int iNode=0;iNode<numNode;iNode++)
		         {  
                   long node=meshElementNodes[iEle*(max_node_in_cell+1)+1+iNode]-1;
		           int kbp= m_kbp00[node];
				   x_sum += nodeXPtr[iNode];
				   y_sum += nodeYPtr[iNode];
				   long node_z_start = node_z_start_index[node];

				   int z_displace_node = iLayer-std::max(1,kbp)+1;

				    // degenerated ele node
				   if(z_displace_node<0)
				   {
					 z_displace_node=0;
				   }
				   double z = zPtr[node_z_start+z_displace_node];
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
	//}
	return true;
}
// z is filled with 0
bool SCHISMMeshProvider10::fillEleCenterCoord2D(float * a_pointCoord,const int & a_timeStep) const
{
	if( m_mesh_loaded==false)
	{
	   return false;
	}
	else
	{
	
	  long * meshElementNodes = new long [(MeshConstants10::MAX_NUM_NODE_PER_CELL+1)*m_number_element];
	  fillMeshElement(meshElementNodes);
      float * nodeXPtr      = new float [m_number_node];
    
      float * nodeYPtr      = new float [m_number_node];
    

      retrieve1DVar( nodeXPtr,
	                   m_dataFilePtr,
                       MeshConstants10::NODE_X,
                       m_number_node);

      retrieve1DVar( nodeYPtr,
	                   m_dataFilePtr,
                       MeshConstants10::NODE_Y,
                       m_number_node);
	  int max_node_in_cell = MeshConstants10::MAX_NUM_NODE_PER_CELL;

      for(long iEle=0;iEle <m_number_element; iEle++)
      {		
		int numNode = meshElementNodes[iEle*(max_node_in_cell+1)];
		       
		float x_sum = 0.0;
		float y_sum = 0.0;
			
        for (int iNode=0;iNode<numNode;iNode++)
		{  
           long node=meshElementNodes[iEle*(max_node_in_cell+1)+1+iNode]-1;
		   x_sum += nodeXPtr[iNode];
		   y_sum += nodeYPtr[iNode];
		}
      
		*a_pointCoord++     = x_sum/numNode;
        *a_pointCoord++     = y_sum/numNode;
        *a_pointCoord++     = MeshConstants10::DUMMY_ELEVATION;
               
		}


	  delete nodeXPtr;
	  delete nodeYPtr;
      delete meshElementNodes;
	}
	return true;
}


// z is filled with 0
bool SCHISMMeshProvider10::fillEleCenterCoord2D(double * a_pointCoord,const int & a_timeStep) const
{
	if( m_mesh_loaded==false)
	{
	   return false;
	}
	else
	{
	
	  long * meshElementNodes = new long [(MeshConstants10::MAX_NUM_NODE_PER_CELL+1)*m_number_element];
	  fillMeshElement(meshElementNodes);
      double * nodeXPtr      = new double [m_number_node];
    
      double * nodeYPtr      = new double [m_number_node];
    

      retrieve1DVar( nodeXPtr,
	                   m_dataFilePtr,
                       MeshConstants10::NODE_X,
                       m_number_node);

      retrieve1DVar( nodeYPtr,
	                   m_dataFilePtr,
                       MeshConstants10::NODE_Y,
                       m_number_node);
	  int max_node_in_cell = MeshConstants10::MAX_NUM_NODE_PER_CELL;

      for(long iEle=0;iEle <m_number_element; iEle++)
      {		
		int numNode = meshElementNodes[iEle*(max_node_in_cell+1)];
		       
		double x_sum = 0.0;
		double y_sum = 0.0;
			
        for (int iNode=0;iNode<numNode;iNode++)
		{  
           long node=meshElementNodes[iEle*(max_node_in_cell+1)+1+iNode]-1;
		   x_sum += nodeXPtr[iNode];
		   y_sum += nodeYPtr[iNode];
		}
      
		*a_pointCoord++     = x_sum/numNode;
        *a_pointCoord++     = y_sum/numNode;
        *a_pointCoord++     = MeshConstants10::DUMMY_ELEVATION;
               
		}


	  delete nodeXPtr;
	  delete nodeYPtr;
      delete meshElementNodes;
	}
	return true;
}


bool SCHISMMeshProvider10::fillPointCoord3D(float * a_pointCoord,const int & a_timeStep) const
{
	/*if( m_mesh_loaded==false)
	{
	   return false;
	}
	else
	{*/
		
      float * nodeXPtr      = new float [m_number_node];
      float * nodeYPtr      = new float [m_number_node];
    
	  float * xPtr = nodeXPtr;
	  float*  yPtr = nodeYPtr;

      retrieve1DVar( nodeXPtr,
	                 m_dataFilePtr,
                     MeshConstants10::NODE_X,
                     m_number_node);

      retrieve1DVar( nodeYPtr,
	                 m_dataFilePtr,
                    MeshConstants10::NODE_Y,
                     m_number_node);


	  float*           zPtr = new float [m_number_layer*m_number_node];
      float*           zPtrTemp = zPtr;
	  zcoords3D(zPtr,a_timeStep);

	 int * kbp00 = new int [m_number_node];
	 fillKbp00(kbp00,a_timeStep);

	   for (int iLayer= 0; iLayer<m_number_layer;iLayer++)
        {
		 
          for(long iNode=0;iNode <m_number_node; iNode++)
            {
		 
              float x            = xPtr[iNode];
              float y            = yPtr[iNode];
			  if (iLayer>=(std::max(1,kbp00[iNode])-1))
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
	  delete kbp00;
      
	//}
	return true;
}


bool SCHISMMeshProvider10::fillPointCoord3D(double * a_pointCoord,const int & a_timeStep) const
{
	/*if( m_mesh_loaded==false)
	{
	   return false;
	}
	else
	{*/
	
      double * nodeXPtr      = new double [m_number_node];
      double * nodeYPtr      = new double [m_number_node];
    
	  double * xPtr = nodeXPtr;
	  double*  yPtr = nodeYPtr;

      retrieve1DVar( nodeXPtr,
	                 m_dataFilePtr,
                     MeshConstants10::NODE_X,
                     m_number_node);

      retrieve1DVar( nodeYPtr,
	                 m_dataFilePtr,
                    MeshConstants10::NODE_Y,
                     m_number_node);


	  float*           zPtr = new float [m_number_layer*m_number_node];
      float*           zPtrTemp = zPtr;
	  
	  zcoords3D(zPtr,a_timeStep);
	 
	 int * kbp00 = new int [m_number_node];
	 fillKbp00(kbp00,a_timeStep);

	   for (int iLayer= 0; iLayer<m_number_layer;iLayer++)
        {
		 
          for(long iNode=0;iNode <m_number_node; iNode++)
            {
		 
              double x            = xPtr[iNode];
              double y            = yPtr[iNode];
			  if (iLayer>=(std::max(1,kbp00[iNode])-1))
			  {
                  *a_pointCoord++         = x;
                  *a_pointCoord++         = y;
				  double zval = *zPtrTemp++;
                  *a_pointCoord++         = zval;
			  }

            }

        }

	 
	  delete nodeXPtr;
	  delete nodeYPtr;
	  delete zPtr;
	  delete kbp00;
      
	//}
	return true;
}

bool  SCHISMMeshProvider10::fillMeshElement(long * a_elementCache) const
{
	if (!m_mesh_loaded)
	{
		return false;
	}
	else
	{
       SCHISMVar10 * ncFacePtr          = m_dataFilePtr->get_var(MeshConstants10::MESH_FACE_NODES);
       if (!(ncFacePtr->is_valid()))
      {
      
         throw SCHISMFileException10("no mesh element in "+m_dataFilePtr->file());
      }
 

     if (!ncFacePtr->get(a_elementCache))
     {
        throw SCHISMFileException10("fail to read mesh element in "+m_dataFilePtr->file());
     }
   }
   return true;
}

bool SCHISMMeshProvider10::zcoords2D(float * a_zCachePtr,const int & a_timeStep) const
{
	 for(long iNode=0;iNode <m_number_node; iNode++)
        {
	
            *a_zCachePtr++         = MeshConstants10::DUMMY_ELEVATION ;
        }

	 return true;
}

bool SCHISMMeshProvider10::zcoords3D(float * a_zCachePtr,const int & a_timeStep) const
{
	
	 return false;
}

bool SCHISMMeshProvider10::zcoords3D2(float * a_zCachePtr,const int & a_timeStep) const
{
	
	 return false;
}

bool SCHISMMeshProvider10::zSideCenter2D(float * a_zCachePtr,const int & a_timeStep) const
{
	for(long iNode=0;iNode <m_number_side; iNode++)
        {
	
            *a_zCachePtr++         = MeshConstants10::DUMMY_ELEVATION;
        }

	 return true;
}

 
bool SCHISMMeshProvider10::zSideCenter3D(float * a_zCachePtr,const int & a_timeStep) const
{
	float*           zPtr = new float [m_number_layer*m_number_node];
    float*           zPtrTemp = zPtr;
	zcoords3D2(zPtr,a_timeStep);
	int * kbp00 = new int [m_number_node];
	fillKbp00(kbp00,a_timeStep);
	int * kbs = new int [m_number_side];
	fillKbs(kbs,a_timeStep);

	long * node_z_start_index  = new long [m_number_node];
	long valid_var_size = 0;
	float half =0.5;

    for(long iNode=0;iNode<m_number_node;iNode++)
    {
	node_z_start_index[iNode]=valid_var_size;
	valid_var_size+=m_number_layer-std::max(1,kbp00[iNode])+1;
    }

	for (int iLayer= 0; iLayer<m_number_layer;iLayer++)
    {
		 
        for(long iSide=0;iSide <m_number_side; iSide++)
        {
		 

			if (iLayer>=(std::max(1,kbs[iSide])-1))
			{
 
                long node1 = m_sideNodes[iSide*2];
		        long node2 = m_sideNodes[iSide*2+1];
               

				long node1_z_start = node_z_start_index[node1];
				long node2_z_start = node_z_start_index[node2];
				int z_displace_node1 = iLayer-kbp00[node1]+1;
				int z_displace_node2 = iLayer-kbp00[node2]+1;

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
    delete kbp00;
	delete kbs;
	return true;
}

bool SCHISMMeshProvider10::zEleCenter2D(float * a_zCachePtr,const int & a_timeStep) const
{
	for(long i=0;i <m_number_element; i++)
        {
	
            *a_zCachePtr++         = MeshConstants10::DUMMY_ELEVATION;
        }

	 return true;
}

 
bool SCHISMMeshProvider10::zEleCenter3D(float * a_zCachePtr,const int & a_timeStep) const
{
	float*           zPtr = new float [m_number_layer*m_number_node];
    float*           zPtrTemp = zPtr;
	zcoords3D2(zPtr,a_timeStep);
	int * kbp00 = new int [m_number_node];
	fillKbp00(kbp00,a_timeStep);
	int * kbe = new int [m_number_element];
	fillKbe(kbe,a_timeStep);
	long * node_z_start_index  = new long [m_number_node];
	long valid_var_size = 0;

    for(long iNode=0;iNode<m_number_node;iNode++)
    {
	node_z_start_index[iNode]=valid_var_size;
	valid_var_size+=m_number_layer-std::max(1,kbp00[iNode])+1;
    }

	int max_node_in_cell = MeshConstants10::MAX_NUM_NODE_PER_CELL;
	long * meshElementNodes = new long [(MeshConstants10::MAX_NUM_NODE_PER_CELL+1)*m_number_element];
	fillMeshElement(meshElementNodes);
    
	
	for (int iLayer= 0; iLayer<m_number_layer;iLayer++)
    {
		
	  for(long iEle=0;iEle <m_number_element; iEle++)
       {
		    
			if (iLayer>=(std::max(1,kbe[iEle])-1))
			{
			    int numNode = meshElementNodes[iEle*(max_node_in_cell+1)];
				float z_sum = 0.0;
				
                for (int iNode=0;iNode<numNode;iNode++)
		        {  
					long node=meshElementNodes[iEle*(max_node_in_cell+1)+1+iNode]-1;
					int kbp= kbp00[node];
				
					long node_z_start = node_z_start_index[node];

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
	delete kbp00;
	delete kbe;
	return true;
}


bool SCHISMMeshProvider10::slayers(float * a_cache) const
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

bool SCHISMMeshProvider10::depth(float * a_cache) const
{
	return false;
}


bool SCHISMMeshProvider10::mesh3d_is_static() const
{
	int dynamic_flag = m_dataFilePtr->global_att_as_int(MeshConstants10::DYNAMIC_MESH);
	if(!dynamic_flag)
	{
	   return true;
	}
	else
	return false;
}