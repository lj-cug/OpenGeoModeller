#include "MDSCHISMMeshProvider.h"
#include "MeshConstants10.h"
#include "SCHISMFileUtil10.h"
#include "MDSCHISMOutput.h"
#include "SchismGeometry10.h"
#include "SCHISMFile10.h"
#include <iostream>
#include <fstream>  
#include <sstream> 
#include <algorithm>
using std::ios;
using std::ifstream;

MDSCHISMMeshProvider::MDSCHISMMeshProvider(const std::string & a_ncfile, SCHISMFile10 * a_nc_ptr,const std::string& a_local_file):MeshProvider10(a_ncfile),
	                                                                          m_kbp00(NULL),
																			  m_layerSCoords(NULL),
                                                                              m_kbs(NULL),
                                                                              m_kbe(NULL),
                                                                              m_side_nodes(NULL),
																			  m_kbp_ele_filled(false),
																			  m_kbp_side_filled(false),
																			  m_kbp_node_filled(false),
																			  m_node_neighbor_ele(NULL),
																			  m_side_neighbor_ele(NULL),
																			  m_node_neighbor_ele_filled(false),
																			  m_side_neighbor_ele_filled(false),
																			  m_max_ele_at_node(0)

{
    
  
  //m_dataFilePtr=new MDSchismOutput(a_ncfile,a_local_file);
	m_dataFilePtr = a_nc_ptr;
  m_local_global_file=a_local_file;

  if (!(m_dataFilePtr->is_valid()))
  {
    m_valid_provider = false;
	return;
  }
 
  bool temp=loadMesh();

  if (!temp)
  {
	  m_valid_provider=false;
	  return;
  }

  m_valid_provider = true;                                                           
}

MDSCHISMMeshProvider::~MDSCHISMMeshProvider()
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

   if (m_side_nodes)
   {
	   delete m_side_nodes;
   }

   if(m_node_neighbor_ele)
   {
	   delete m_node_neighbor_ele;
   }
   if(m_side_neighbor_ele)
   {
	   delete m_side_neighbor_ele;
   }

   if(m_local_node_id_to_global_id)
   {
	   delete m_local_node_id_to_global_id;
   }
   if(m_local_ele_id_to_global_id)
   {
	   delete m_local_ele_id_to_global_id;
   }

   if(m_local_side_id_to_global_id)
   {
	   delete m_local_side_id_to_global_id;
   }
   
   if(m_faceNodesPtr)
   {
	   delete m_faceNodesPtr;   
   }
   if(m_nodex)
   {
	   delete m_nodex;
   }
   if(m_nodey)
   {
	   delete m_nodey;
   }
   if(m_dp)
   {
	   delete m_dp;
   }
}

bool MDSCHISMMeshProvider::set_data_file(SCHISMFile10* a_file)
{
	m_dataFilePtr = a_file;
	return true;

}
bool MDSCHISMMeshProvider::fillKbp00(int * a_cache,const int & a_timeStep) const
 {

	 for(long i=0;i<m_number_node;i++)
	 {
	    a_cache[i]=m_kbp00[i];
	 }
	 return true;

 }

bool MDSCHISMMeshProvider::fillKbs(int * a_cache,const int & a_timeStep) const
 {

	for(long i=0;i<m_number_side;i++)
	 {
	    a_cache[i]=m_kbs[i];
	 }
	 return true;

 }

bool MDSCHISMMeshProvider::fillKbe(int * a_cache,const int & a_timeStep) const
 {

	for(long i=0;i<m_number_element;i++)
	 {
	    a_cache[i]=m_kbe[i];
	 }
	 return true;

 }

bool MDSCHISMMeshProvider::update_bottom_layer(const int & a_timeStep) 
{
	return true; //no changing bottom option now
	
}



 void  MDSCHISMMeshProvider::fill_node_dry_wet(int* &a_node_dry_wet,int* a_ele_dry_wet)
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
			 long ele_id=m_node_neighbor_ele[m_max_ele_at_node*inode+iele];
			 if (!(ele_id==invalid_id))
			 {
				 if(a_ele_dry_wet[ele_id]==MeshConstants10::WET_FLAG)
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

 void  MDSCHISMMeshProvider::fill_side_dry_wet(int* &a_side_dry_wet,int* a_ele_dry_wet)
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
			 long ele_id=m_side_neighbor_ele[num_side_neighbor_ele*iside+iele];
			 if (!(ele_id==invalid_id))
			 {
				 if(a_ele_dry_wet[ele_id]==MeshConstants10::WET_FLAG)
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

bool MDSCHISMMeshProvider::loadMesh()
{

   ifstream*    localFileStream = new ifstream(m_local_global_file.c_str()); 
   if (!localFileStream->good())
   {
       return false;
   }
    std::string  lineTemp;
  
    std::getline(*localFileStream,lineTemp);
	std::getline(*localFileStream,lineTemp);
	*(localFileStream)>> m_number_element;
	int stepsize =  MeshConstants10::MAX_NUM_NODE_PER_CELL +1;
	
	m_faceNodesPtr = new long   [stepsize*m_number_element];
	m_local_ele_id_to_global_id= new long [m_number_element];

	for(long iEle=0;iEle<m_number_element;iEle++)
		{
		long v1,v2;
		*(localFileStream)>>v1>>v2;
		m_local_ele_id_to_global_id[iEle]=v2; //here global id starts from 1
		}
	
	*(localFileStream)>> m_number_node;
    m_local_node_id_to_global_id= new long [m_number_node];

	for(long iNode=0;iNode<m_number_node;iNode++)
		{
		long v1,v2;
		*(localFileStream)>>v1>>v2;
		m_local_node_id_to_global_id[iNode]=v2; //here global id starts from 1
		}
		

	*(localFileStream)>> m_number_side;
    m_local_side_id_to_global_id= new long [m_number_side];

	for(long iSide=0;iSide<m_number_side;iSide++)
		{
		long v1,v2;
		*(localFileStream)>>v1>>v2;
		m_local_side_id_to_global_id[iSide]=v2; //here global id starts from 1
		}
   
     std::string temp;
	(*localFileStream)>>temp;
	int year,month,day;
    double time,tzone;
	(*localFileStream)>>year>>month>>day>>time>>tzone;
    
     
	int nrec, nspool,kz,ics;
	double v2,h0;
    (*localFileStream)>>nrec>>v2>>nspool>>m_number_layer>>kz>>h0>>m_hs>>m_hc>>m_thetab>>m_thetaf>>ics;

    std::getline(*localFileStream,lineTemp);
  
    m_layerSCoords = new double [m_number_layer];

	int num_sigma_layers=m_number_layer-kz+1;
	
	for(int iz=1;iz<kz;iz++)
	{
		double v; //z layer vals
		(*localFileStream)>>v;
	}
	
    int sigma_layer_start_id = m_number_layer-num_sigma_layers;
	int itemp=0;
	for(int iLayer=sigma_layer_start_id;iLayer<m_number_layer;iLayer++)
	{
		(*localFileStream)>>m_layerSCoords[iLayer];
		itemp++;	
    }  

	long t1,t2;
    (*localFileStream)>>t1>>t2;
	
	m_number_node_no_ghost=t1;
    m_number_element_no_ghost=t2;

	//read in x,y,dp, kpb00
	m_dp=new double [m_number_node];
	m_nodex     = new double [m_number_node];
	m_nodey     = new double [m_number_node];
	m_kbp00        = new int [m_number_node];
	m_kbe          = new int [m_number_element];
	m_kbs          = new int [m_number_side];
	for(long iNode=0;iNode<m_number_node;iNode++)
	{	
		(*localFileStream)>>m_nodex[iNode]>>m_nodey[iNode]>>m_dp[iNode]>>m_kbp00[iNode];
	}
	// read in face nodes
	for(long iEle=0;iEle<m_number_element;iEle++)
	{
		long numNode;
		(*localFileStream)>>numNode;
		m_faceNodesPtr[0+iEle*stepsize]=numNode;
		int min_bottom=1; //schism bottom level is 1
		for(long iNode=0;iNode<numNode;iNode++)
		{
			int node_id;
			(*localFileStream)>>node_id;
			m_faceNodesPtr[iNode+1+iEle*stepsize]=node_id;
			if(iNode==0)
			{
				min_bottom=m_kbp00[node_id];
			}
			else
			{
				if(min_bottom>m_kbp00[node_id])
				{
					min_bottom=m_kbp00[node_id];
				}
			}
		}
		m_kbe[iEle]=min_bottom;
		
	}
	m_side_nodes    = new long [m_number_side*2];
	//read in side nodes
	for(long iSide=0;iSide<m_number_side;iSide++)
	{
		long sideid,node1,node2;
		(*localFileStream)>>sideid>>node1>>node2;
		m_side_nodes[0+iSide*2]=node1-1;
		m_side_nodes[1+iSide*2]=node2-1;
		int min_bottom=m_kbp00[node1-1]; //schism bottom level is 1
		if (min_bottom>m_kbp00[node2-1])
		{
			min_bottom=m_kbp00[node2-1];
		}
		m_kbs[iSide]=min_bottom;
	}
	

	//(*localFileStream)>>m_number_node_no_ghost>>m_number_element_no_ghost>>m_number_side_no_ghost;
    m_mesh_loaded = true;
	localFileStream->close();
	delete localFileStream;
    return true;
}


// z is filled with 0
bool MDSCHISMMeshProvider::fillPointCoord2D(float * a_pointCoord,const int & a_timeStep) const
{
	if( m_mesh_loaded==false)
	{
	   return false;
	}
	else
	{
		
	  for(long iNode=0;iNode < m_number_node; iNode++)
          {
            float x            =  m_nodex[iNode];
            float y            =  m_nodey[iNode];
            *a_pointCoord++    = x;
            *a_pointCoord++    = y;
            // must put a dummy z value as visit manaul example does
            *a_pointCoord++    = MeshConstants10::DUMMY_ELEVATION;
           
          }

	}
	return true;
}

bool MDSCHISMMeshProvider::fillPointCoord2D(double * a_pointCoord,const int & a_timeStep) const
{
	if( m_mesh_loaded==false)
	{
	   return false;
	}
	else
	{
		
	  for(long iNode=0;iNode < m_number_node; iNode++)
          {
            double x            =  m_nodex[iNode];
            double y            =  m_nodey[iNode];
            *a_pointCoord++    = x;
            *a_pointCoord++    = y;
            // must put a dummy z value as visit manaul example does
            *a_pointCoord++    = MeshConstants10::DUMMY_ELEVATION;
          }
      
	}
	return true;
}

bool MDSCHISMMeshProvider::fillSideCenterCoord3D(float * a_pointCoord,const int & a_timeStep) const
{
	
	  double*           zPtr = new double [m_number_layer*m_number_node];
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
 
                 long node1 = m_side_nodes[iSide*2];
		         long node2 = m_side_nodes[iSide*2+1];
                 float x1            =  m_nodex[node1];
                 float y1            =  m_nodey[node1];
			     float x2            =  m_nodex[node2];
                 float y2            =  m_nodey[node2];
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

	  delete zPtr;
	  delete node_z_start_index;
	  delete kbp00;
	  delete kbs;
      
	//}
	return true;
}


bool MDSCHISMMeshProvider::fillSideCenterCoord3D(double * a_pointCoord,const int & a_timeStep) const
{
	
	  double*           zPtr = new double [m_number_layer*m_number_node];
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
 
                 long node1 = m_side_nodes[iSide*2];
		         long node2 = m_side_nodes[iSide*2+1];
                 double x1            =  m_nodex[node1];
                 double y1            =  m_nodey[node1];
			     double x2            =  m_nodex[node2];
                 double y2            =  m_nodey[node2];
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

	  delete zPtr;
	  delete node_z_start_index;
	  delete kbp00;
	  delete kbs;
      
	//}
	return true;
}

bool MDSCHISMMeshProvider::fillSideFaceCenterCoord3D(float * a_pointCoord,const int & a_timeStep) const
{
	


	  double*           zPtr = new double [m_number_layer*m_number_node];
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
 
                 long node1 = m_side_nodes[iSide*2];
		         long node2 = m_side_nodes[iSide*2+1];
                 float x1            =  m_nodex[node1];
                 float y1            =  m_nodey[node1];
			     float x2            =  m_nodex[node2];
                 float y2            =  m_nodey[node2];
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

	  delete zPtr;
	  delete node_z_start_index;
	  delete kbp00;
	  delete kbs;

	return true;
}

bool MDSCHISMMeshProvider::fillSideFaceCenterCoord3D(double * a_pointCoord,const int & a_timeStep) const
{
	
	  double*           zPtr = new double [m_number_layer*m_number_node];
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
 
                 long node1 = m_side_nodes[iSide*2];
		         long node2 = m_side_nodes[iSide*2+1];
                 double x1            =  m_nodex[node1];
                 double y1            =  m_nodey[node1];
			     double x2            =  m_nodex[node2];
                 double y2            =  m_nodey[node2];
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

	  delete zPtr;
	  delete node_z_start_index;
	  delete kbp00;
	  delete kbs;
      
	//}
	return true;
}


// z is filled with 0
bool MDSCHISMMeshProvider::fillSideCenterCoord2D(float * a_pointCoord,const int & a_timeStep) const
{
	if( m_mesh_loaded==false)
	{
	   return false;
	}
	else
	{
	  float half =0.5;
	  for(long iSide=0;iSide < m_number_side; iSide++)
          {
			long node1 = m_side_nodes[iSide*2];
		    long node2 = m_side_nodes[iSide*2+1];
            float x1            =  m_nodex[node1];
            float y1            =  m_nodey[node1];
			float x2            =  m_nodex[node2];
            float y2            =  m_nodey[node2];
            *a_pointCoord++    = half*(x1+x2);
            *a_pointCoord++    = half*(y1+y2);
            // must put a dummy z value as visit manaul example does
            *a_pointCoord++    = MeshConstants10::DUMMY_ELEVATION;
           
          }
      
	}
	return true;
}

// z is filled with 0
bool MDSCHISMMeshProvider::fillSideCenterCoord2D(double * a_pointCoord,const int & a_timeStep) const
{
	if( m_mesh_loaded==false)
	{
	   return false;
	}
	else
	{
	  double half =0.5;
    
	  for(long iSide=0;iSide < m_number_side; iSide++)
          {
			long node1 = m_side_nodes[iSide*2];
		    long node2 = m_side_nodes[iSide*2+1];
            double x1            =  m_nodex[node1];
            double y1            =  m_nodey[node1];
			double x2            =  m_nodex[node2];
            double y2            =  m_nodey[node2];
            *a_pointCoord++    = half*(x1+x2);
            *a_pointCoord++    = half*(y1+y2);
            // must put a dummy z value as visit manaul example does
            *a_pointCoord++    = MeshConstants10::DUMMY_ELEVATION;
           
          }

	}
	return true;
}

bool MDSCHISMMeshProvider::fillEleCenterCoord3D(float * a_pointCoord,const int & a_timeStep) const
{
	

	  long * meshElementNodes = new long [(MeshConstants10::MAX_NUM_NODE_PER_CELL+1)*m_number_element];
	  fillMeshElement(meshElementNodes);
		
	  double*           zPtr = new double [m_number_layer*m_number_node];
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
				   x_sum += m_nodex[iNode];
				   y_sum += m_nodey[iNode];
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
	 
	  delete zPtr;
	  delete node_z_start_index;
	  delete meshElementNodes;
	//}
	return true;
}

bool MDSCHISMMeshProvider::fillEleCenterCoord3D(double * a_pointCoord,const int & a_timeStep) const
{

	  long * meshElementNodes = new long [(MeshConstants10::MAX_NUM_NODE_PER_CELL+1)*m_number_element];
	  fillMeshElement(meshElementNodes);
		
      

	  double*           zPtr = new double [m_number_layer*m_number_node];
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
				   x_sum += m_nodex[iNode];
				   y_sum += m_nodey[iNode];
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
	
	  delete zPtr;
	  delete node_z_start_index;
	  delete meshElementNodes;
	//}
	return true;
}
// z is filled with 0
bool MDSCHISMMeshProvider::fillEleCenterCoord2D(float * a_pointCoord,const int & a_timeStep) const
{
	if( m_mesh_loaded==false)
	{
	   return false;
	}
	else
	{
	
	  long * meshElementNodes = new long [(MeshConstants10::MAX_NUM_NODE_PER_CELL+1)*m_number_element];
	  fillMeshElement(meshElementNodes);
     
	  int max_node_in_cell = MeshConstants10::MAX_NUM_NODE_PER_CELL;

      for(long iEle=0;iEle <m_number_element; iEle++)
      {		
		int numNode = meshElementNodes[iEle*(max_node_in_cell+1)];
		       
		float x_sum = 0.0;
		float y_sum = 0.0;
			
        for (int iNode=0;iNode<numNode;iNode++)
		{  
           long node=meshElementNodes[iEle*(max_node_in_cell+1)+1+iNode]-1;
		   x_sum += m_nodex[iNode];
		   y_sum += m_nodey[iNode];
		}
      
		*a_pointCoord++     = x_sum/numNode;
        *a_pointCoord++     = y_sum/numNode;
        *a_pointCoord++     = MeshConstants10::DUMMY_ELEVATION;
               
		}
      delete meshElementNodes;
	}
	return true;
}


// z is filled with 0
bool MDSCHISMMeshProvider::fillEleCenterCoord2D(double * a_pointCoord,const int & a_timeStep) const
{
	if( m_mesh_loaded==false)
	{
	   return false;
	}
	else
	{
	
	  long * meshElementNodes = new long [(MeshConstants10::MAX_NUM_NODE_PER_CELL+1)*m_number_element];
	  fillMeshElement(meshElementNodes);
     
	  int max_node_in_cell = MeshConstants10::MAX_NUM_NODE_PER_CELL;

      for(long iEle=0;iEle <m_number_element; iEle++)
      {		
		int numNode = meshElementNodes[iEle*(max_node_in_cell+1)];
		       
		double x_sum = 0.0;
		double y_sum = 0.0;
			
        for (int iNode=0;iNode<numNode;iNode++)
		{  
           long node=meshElementNodes[iEle*(max_node_in_cell+1)+1+iNode]-1;
		   x_sum += m_nodex[iNode];
		   y_sum += m_nodey[iNode];
		}
      
		*a_pointCoord++     = x_sum/numNode;
        *a_pointCoord++     = y_sum/numNode;
        *a_pointCoord++     = MeshConstants10::DUMMY_ELEVATION; 
		}
      delete meshElementNodes;
	}
	return true;
}


bool MDSCHISMMeshProvider::fillPointCoord3D(float * a_pointCoord,const int & a_timeStep) const
{
	
     
	  double*           zPtr = new double [m_number_layer*m_number_node];
      double*           zPtrTemp = zPtr;
	  zcoords3D(zPtr,a_timeStep);

	 int * kbp00 = new int [m_number_node];
	 fillKbp00(kbp00,a_timeStep);

	   for (int iLayer= 0; iLayer<m_number_layer;iLayer++)
        {
		 
          for(long iNode=0;iNode <m_number_node; iNode++)
            {
		 
              float x            = m_nodex[iNode];
              float y            = m_nodey[iNode];
			  if (iLayer>=(std::max(1,kbp00[iNode])-1))
			  {
                  *a_pointCoord++         = x;
                  *a_pointCoord++         = y;
                  *a_pointCoord++         = *zPtrTemp++;
			  }

            }

        }



	
	  delete zPtr;
	  delete kbp00;

	return true;
}


bool MDSCHISMMeshProvider::fillPointCoord3D(double * a_pointCoord,const int & a_timeStep) const
{
	
	  double*           zPtr = new double [m_number_layer*m_number_node];
      double*           zPtrTemp = zPtr;
	  zcoords3D(zPtr,a_timeStep);

	 int * kbp00 = new int [m_number_node];
	 fillKbp00(kbp00,a_timeStep);

	   for (int iLayer= 0; iLayer<m_number_layer;iLayer++)
        {
		 
          for(long iNode=0;iNode <m_number_node; iNode++)
            {
		 
              double x            = m_nodex[iNode];
              double y            = m_nodey[iNode];
			  if (iLayer>=(std::max(1,kbp00[iNode])-1))
			  {
                  *a_pointCoord++         = x;
                  *a_pointCoord++         = y;
                  *a_pointCoord++         = *zPtrTemp++;
			  }

            }

        }

	  delete zPtr;
	  delete kbp00;

	return true;
}

bool  MDSCHISMMeshProvider::fillMeshElement(long * a_elementCache) const
{
	//if (!m_mesh_loaded)
	//{
	//	loadMesh();
	//}
	//debug1 << "in fill mesh " << m_number_element << "\n";
       for(long i=0;i<(MeshConstants10::MAX_NUM_NODE_PER_CELL+1)*m_number_element;i++)
	   {
		   a_elementCache[i]=m_faceNodesPtr[i];
		  
	   }
     
   return true;
}

bool  MDSCHISMMeshProvider::mesh_loaded() const
{
	return m_mesh_loaded;
}

 // updates z coords at a timestep
bool MDSCHISMMeshProvider::zcoords2D(double * a_zCachePtr,const int & a_timeStep) const
{
	return true;
}

 // updates z coords at a timestep
bool MDSCHISMMeshProvider::zcoords2D(float * a_zCachePtr,const int & a_timeStep) const
{
	return true;
}

bool MDSCHISMMeshProvider::zcoords3D(double * a_zCachePtr,const int & a_timeStep) const
{
	
	 int timeStart    = a_timeStep;

     SCHISMVar10 * zVarPtr = m_dataFilePtr->get_var(MeshConstants10::ZCOORD);
  
     if (!(zVarPtr->is_valid()))
     {
      
       throw SCHISMFileException10("invlaid var "+MeshConstants10::ZCOORD+" for data file "+m_dataFilePtr->file());
     }
   
	
     float missing_val = MeshConstants10::MISSING_VALUE;
	 float dry_zcor = MeshConstants10::DRY_ZCOR;

     zVarPtr->set_cur(timeStart);

	 long z_var_size =0;
	 long * node_start_index= new long [m_number_node];
	 int * kbp00 = new int [m_number_node];
	 fillKbp00(kbp00,a_timeStep);
	 for(int iNode=0;iNode<m_number_node;iNode++)
	 {
		 node_start_index[iNode]=z_var_size;
		 z_var_size+=m_number_layer-std::max(1,kbp00[iNode])+1;
	 }

	

     float*           zPtr = new float [z_var_size];
     if (!(zVarPtr->get(zPtr)))
     {
        throw SCHISMFileException10("fail to retrieve var "+MeshConstants10::ZCOORD+" from data file "+m_dataFilePtr->file());
     }

	  for (int iLayer= 0; iLayer<m_number_layer;iLayer++)
      { 
		 
          for(long iNode=0;iNode <m_number_node; iNode++)
            {
			  long start_index = node_start_index[iNode];
		      if (iLayer<(std::max(1,kbp00[iNode])-1))
			  {
				 
			  }
			  else
			  {
				  float temp =   zPtr[start_index+iLayer+1-std::max(1,kbp00[iNode])];
				  if((temp!=dry_zcor)&&(temp!=missing_val))
				  {
                      *a_zCachePtr = temp;
				  }
				  else // if not a valid z use computed one to finish mesh 
				  {
					  float surface= MeshConstants10::DUMMY_ELEVATION;
				      float sigma        = m_layerSCoords[iLayer]; 
				      float depth        = m_dp[iNode];    
				      float z            = convertStoZ(sigma,
												       surface,
												       depth,
												       m_hs,
												       m_hc,
												       m_thetab,
												       m_thetaf);
				      *a_zCachePtr         = z;
				  }
				  a_zCachePtr++;
			  }

            }
       }
	 delete zPtr;
	 delete kbp00;
	 
	 return true;
}

void  MDSCHISMMeshProvider::fill_node_global_id(long * a_buff)
{
	for (long iNode = 0; iNode < m_number_node; iNode++)
	{
		a_buff[iNode] = m_local_node_id_to_global_id[iNode];
	}
}
void  MDSCHISMMeshProvider::fill_ele_global_id(long * a_buff)
{
	for (long iEle = 0; iEle < m_number_element; iEle++)
	{
		a_buff[iEle] = m_local_ele_id_to_global_id[iEle];
	}
}


bool MDSCHISMMeshProvider::zcoords3D(float * a_zCachePtr,const int & a_timeStep) const
{
	
	 int timeStart    = a_timeStep;

     SCHISMVar10 * zVarPtr = m_dataFilePtr->get_var(MeshConstants10::ZCOORD);
  
     if (!(zVarPtr->is_valid()))
     {
      
       throw SCHISMFileException10("invlaid var "+MeshConstants10::ZCOORD+" for data file "+m_dataFilePtr->file());
     }
   
	
     float missing_val = MeshConstants10::MISSING_VALUE;
	 float dry_zcor = MeshConstants10::DRY_ZCOR;

     zVarPtr->set_cur(timeStart);

	 long z_var_size =0;
	 long * node_start_index= new long [m_number_node];
	 int * kbp00 = new int [m_number_node];
	 fillKbp00(kbp00,a_timeStep);
	 for(int iNode=0;iNode<m_number_node;iNode++)
	 {
		 node_start_index[iNode]=z_var_size;
		 z_var_size+=m_number_layer-std::max(1,kbp00[iNode])+1;
	 }

	

     float*           zPtr = new float [z_var_size];
     if (!(zVarPtr->get(zPtr)))
     {
        throw SCHISMFileException10("fail to retrieve var "+MeshConstants10::ZCOORD+" from data file "+m_dataFilePtr->file());
     }

	  for (int iLayer= 0; iLayer<m_number_layer;iLayer++)
      { 
		 
          for(long iNode=0;iNode <m_number_node; iNode++)
            {
			  long start_index = node_start_index[iNode];
		      if (iLayer<(std::max(1,kbp00[iNode])-1))
			  {
				 
			  }
			  else
			  {
				  float temp =   zPtr[start_index+iLayer+1-std::max(1,kbp00[iNode])];
				  if((temp!=dry_zcor)&&(temp!=missing_val))
				  {
                      *a_zCachePtr = temp;
				  }
				  else // if not a valid z use computed one to finish mesh 
				  {
					  float surface= MeshConstants10::DUMMY_ELEVATION;
				      float sigma        = m_layerSCoords[iLayer]; 
				      float depth        = m_dp[iNode];    
				      float z            = convertStoZ(sigma,
												       surface,
												       depth,
												       m_hs,
												       m_hc,
												       m_thetab,
												       m_thetaf);
				      *a_zCachePtr         = z;
				  }
				  a_zCachePtr++;
			  }

            }
       }
	 delete zPtr;
	 delete kbp00;
	 
	 return true;
}

double  MDSCHISMMeshProvider::convertStoZ(const double    & a_sigma,
                                     const double    & a_surface,
                                     const double    & a_depth,
                                     const double    & a_hs,
                                     const double    & a_hc,
                                     const double    & a_thetab,
                                     const double    & a_thetaf) const
{

  double surface = a_surface;

  if (fabs(a_surface-MeshConstants10::DRY_SURFACE)<1.0e-6)
    {
      surface = 0.0;
    }

  double one =1.0;
  double half=0.5;
  double two =2.0;

  double csigma = (one-a_thetab)*sinh(a_thetaf*a_sigma)/sinh(a_thetaf)
  +a_thetab*(tanh(a_thetaf*(a_sigma+half))-tanh(a_thetaf/two))/(two*tanh(a_thetaf/two));
  

  double hat    = a_depth;
  if (a_hs < a_depth)
    {
      hat       = a_hs;
    }
   double z        = MeshConstants10::DRY_SURFACE;
   if (hat>a_hc)
    {
            z      = surface*(one+a_sigma)+a_hc*a_sigma+(hat-a_hc)*csigma;
    }
   else
    {
            z      = (hat+surface)*a_sigma+surface;   
    }
   
  return z;

}

bool MDSCHISMMeshProvider::zcoords3D2(double * a_zCachePtr,const int & a_timeStep) const
{
	
	int timeStart    = a_timeStep;

     SCHISMVar10 * zVarPtr = m_dataFilePtr->get_var(MeshConstants10::ZCOORD);
  
     if (!(zVarPtr->is_valid()))
     {
      
       throw SCHISMFileException10("invlaid var "+MeshConstants10::ZCOORD+" for data file "+m_dataFilePtr->file());
     }
   
     zVarPtr->set_cur(timeStart);
	 float missing_val = MeshConstants10::MISSING_VALUE;
	 float dry_zcor = MeshConstants10::DRY_ZCOR;
	

	 long z_var_size =0;
	 long * node_start_index= new long [m_number_node];
	 int * kbp00 = new int [m_number_node];
	 fillKbp00(kbp00,a_timeStep);
	
	 for(long iNode=0;iNode<m_number_node;iNode++)
	 {
	    node_start_index[iNode]=z_var_size;	     
		z_var_size+=m_number_layer-std::max(1,kbp00[iNode])+1;
	 }

     float*           zPtr = new float [z_var_size];
     if (!(zVarPtr->get(zPtr)))
     {
        throw SCHISMFileException10("fail to retrieve var "+MeshConstants10::ZCOORD+" from data file "+m_dataFilePtr->file());
     }

	  for (int iLayer= 0; iLayer<m_number_layer;iLayer++)
      { 
		 
          for(long iNode=0;iNode <m_number_node; iNode++)
            {
			  long start_index = node_start_index[iNode];
		      if (iLayer<(std::max(1,kbp00[iNode])-1))
			  {
				 
			  }
			  else
			  {
				  float temp =   zPtr[start_index+iLayer+1-std::max(1,kbp00[iNode])];
				  if((temp==dry_zcor)||(temp==missing_val))
				  {
					  float surface= MeshConstants10::DUMMY_ELEVATION;
				      float sigma        = m_layerSCoords[iLayer]; 
				      float depth        = m_dp[iNode];    
				      float z            = convertStoZ(sigma,
												       surface,
												       depth,
												       m_hs,
												       m_hc,
												       m_thetab,
												       m_thetaf);
				      zPtr[start_index+iLayer+1-std::max(1,kbp00[iNode])] = z;
				  }
			  }

            }
       }

	  for (int i= 0; i<z_var_size;i++)
      { 
         a_zCachePtr[i] =  zPtr[i];
       }
	 delete zPtr;
	 delete kbp00;
	 delete node_start_index;
	 return true;
}

bool MDSCHISMMeshProvider::zcoords3D2(float * a_zCachePtr,const int & a_timeStep) const
{
	
	int timeStart    = a_timeStep;

     SCHISMVar10 * zVarPtr = m_dataFilePtr->get_var(MeshConstants10::ZCOORD);
  
     if (!(zVarPtr->is_valid()))
     {
      
       throw SCHISMFileException10("invlaid var "+MeshConstants10::ZCOORD+" for data file "+m_dataFilePtr->file());
     }
   
     zVarPtr->set_cur(timeStart);
	 float missing_val = MeshConstants10::MISSING_VALUE;
	 float dry_zcor = MeshConstants10::DRY_ZCOR;
	

	 long z_var_size =0;
	 long * node_start_index= new long [m_number_node];
	 int * kbp00 = new int [m_number_node];
	 fillKbp00(kbp00,a_timeStep);
	
	 for(long iNode=0;iNode<m_number_node;iNode++)
	 {
	    node_start_index[iNode]=z_var_size;	     
		z_var_size+=m_number_layer-std::max(1,kbp00[iNode])+1;
	 }

     float*           zPtr = new float [z_var_size];
     if (!(zVarPtr->get(zPtr)))
     {
        throw SCHISMFileException10("fail to retrieve var "+MeshConstants10::ZCOORD+" from data file "+m_dataFilePtr->file());
     }

	  for (int iLayer= 0; iLayer<m_number_layer;iLayer++)
      { 
		 
          for(long iNode=0;iNode <m_number_node; iNode++)
            {
			  long start_index = node_start_index[iNode];
		      if (iLayer<(std::max(1,kbp00[iNode])-1))
			  {
				 
			  }
			  else
			  {
				  float temp =   zPtr[start_index+iLayer+1-std::max(1,kbp00[iNode])];
				  if((temp==dry_zcor)||(temp==missing_val))
				  {
					  float surface= MeshConstants10::DUMMY_ELEVATION;
				      float sigma        = m_layerSCoords[iLayer]; 
				      float depth        = m_dp[iNode];    
				      float z            = convertStoZ(sigma,
												       surface,
												       depth,
												       m_hs,
												       m_hc,
												       m_thetab,
												       m_thetaf);
				      zPtr[start_index+iLayer+1-std::max(1,kbp00[iNode])] = z;
				  }
			  }

            }
       }

	  for (int i= 0; i<z_var_size;i++)
      { 
         a_zCachePtr[i] =  zPtr[i];
       }
	 delete zPtr;
	 delete kbp00;
	 delete node_start_index;
	 return true;
}

bool MDSCHISMMeshProvider::zSideCenter2D(double * a_zCachePtr,const int & a_timeStep) const
{
	for(long iNode=0;iNode <m_number_side; iNode++)
        {
	
            *a_zCachePtr++         = MeshConstants10::DUMMY_ELEVATION;
        }

	 return true;
}
bool MDSCHISMMeshProvider::zSideCenter2D(float * a_zCachePtr,const int & a_timeStep) const
{
	for(long iNode=0;iNode <m_number_side; iNode++)
        {
	
            *a_zCachePtr++         = MeshConstants10::DUMMY_ELEVATION;
        }

	 return true;
}
 
bool MDSCHISMMeshProvider::zSideCenter3D(double * a_zCachePtr,const int & a_timeStep) const
{
	double*           zPtr = new double [m_number_layer*m_number_node];
    double*           zPtrTemp = zPtr;
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
 
                long node1 = m_side_nodes[iSide*2];
		        long node2 = m_side_nodes[iSide*2+1];
               

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

bool MDSCHISMMeshProvider::zSideCenter3D(float * a_zCachePtr,const int & a_timeStep) const
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
 
                long node1 = m_side_nodes[iSide*2];
		        long node2 = m_side_nodes[iSide*2+1];
               

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

bool MDSCHISMMeshProvider::zEleCenter2D(double * a_zCachePtr,const int & a_timeStep) const
{
	for(long i=0;i <m_number_element; i++)
        {
	
            *a_zCachePtr++         = MeshConstants10::DUMMY_ELEVATION;
        }

	 return true;
}

bool MDSCHISMMeshProvider::zEleCenter2D(float * a_zCachePtr,const int & a_timeStep) const
{
	for(long i=0;i <m_number_element; i++)
        {
	
            *a_zCachePtr++         = MeshConstants10::DUMMY_ELEVATION;
        }

	 return true;
}

 
bool MDSCHISMMeshProvider::zEleCenter3D(double * a_zCachePtr,const int & a_timeStep) const
{
	double*           zPtr = new double [m_number_layer*m_number_node];
    double*           zPtrTemp = zPtr;
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
z_displace_node = 0;
					}
					float z = zPtr[node_z_start + z_displace_node];
					z_sum += z;
				}
				*a_zCachePtr++ = z_sum / numNode;

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
bool MDSCHISMMeshProvider::zEleCenter3D(float * a_zCachePtr, const int & a_timeStep) const
{
	float*           zPtr = new float[m_number_layer*m_number_node];
	float*           zPtrTemp = zPtr;
	zcoords3D2(zPtr, a_timeStep);
	int * kbp00 = new int[m_number_node];
	fillKbp00(kbp00, a_timeStep);
	int * kbe = new int[m_number_element];
	fillKbe(kbe, a_timeStep);
	long * node_z_start_index = new long[m_number_node];
	long valid_var_size = 0;

	for (long iNode = 0; iNode < m_number_node; iNode++)
	{
		node_z_start_index[iNode] = valid_var_size;
		valid_var_size += m_number_layer - std::max(1, kbp00[iNode]) + 1;
	}

	int max_node_in_cell = MeshConstants10::MAX_NUM_NODE_PER_CELL;
	long * meshElementNodes = new long[(MeshConstants10::MAX_NUM_NODE_PER_CELL + 1)*m_number_element];
	fillMeshElement(meshElementNodes);


	for (int iLayer = 0; iLayer < m_number_layer; iLayer++)
	{

		for (long iEle = 0; iEle < m_number_element; iEle++)
		{

			if (iLayer >= (std::max(1, kbe[iEle]) - 1))
			{
				int numNode = meshElementNodes[iEle*(max_node_in_cell + 1)];
				float z_sum = 0.0;

				for (int iNode = 0; iNode < numNode; iNode++)
				{
					long node = meshElementNodes[iEle*(max_node_in_cell + 1) + 1 + iNode] - 1;
					int kbp = kbp00[node];

					long node_z_start = node_z_start_index[node];

					int z_displace_node = iLayer - std::max(1, kbp) + 1;

					// degenerated ele node
					if (z_displace_node < 0)
					{
						z_displace_node = 0;
					}
					float z = zPtr[node_z_start + z_displace_node];
					z_sum += z;
				}
				*a_zCachePtr++ = z_sum / numNode;

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


bool MDSCHISMMeshProvider::slayers(double * a_cache) const
{
	if (m_mesh_loaded)
	{
		for (int i = 0; i < m_number_layer; i++)
		{
			a_cache[i] = m_layerSCoords[i];
		}
		return true;
	}
	return false;
}

bool MDSCHISMMeshProvider::depth(double * a_cache) const
{
	for(long inode = 0; inode < m_number_node; inode++)
	{
		a_cache[inode] = m_dp[inode];
    }
	return true;
}


bool MDSCHISMMeshProvider::mesh3d_is_static() const
{
	int dynamic_flag = m_dataFilePtr->global_att_as_int(MeshConstants10::DYNAMIC_MESH);
	if(!dynamic_flag)
	{
	   return true;
	}
	else
	return false;
}