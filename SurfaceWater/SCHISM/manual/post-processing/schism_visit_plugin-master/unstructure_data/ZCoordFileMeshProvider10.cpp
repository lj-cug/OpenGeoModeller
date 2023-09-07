#include "ZCoordFileMeshProvider10.h"
#include "MeshConstants10.h"
#include "SCHISMFile10.h"
#include "SCHISMFileUtil10.h"
#include "NetcdfSCHISMOutput.h"
#include <algorithm>



using namespace std;

ZCoordMeshProvider10::ZCoordMeshProvider10(const std::string& a_meshFile):SCHISMMeshProvider10(a_meshFile),
                                                                          m_mesh_file(a_meshFile)
{
	
	size_t found = m_mesh_file.find_last_of("/\\");
    std:string data_file_path = m_mesh_file.substr(0,found);
	bool scribeIO = false;
	//find out format of output by reading global attribute source
    //if "V10" in the string 5.8 format, otherwise scribeIo format (no atribute or other values)
	try
	{
		std::string source = m_dataFilePtr->global_att_as_string(MeshConstants10::source);
		std::size_t found_v10 = source.find(MeshConstants10::SCHISM58_OUTPUT_FORMAT);
		if (found_v10 != std::string::npos)
		{
			scribeIO = false;
		}
		else
		{
			scribeIO = true;
		}
	}
	catch (...)
	{
		scribeIO = true;
	}
	//std::size_t found3 = a_meshFile.find("schout");
	//if (found3 == std::string::npos)
	if(scribeIO)
	{

		//see if there any separate zcor nc file available under current folder 
		size_t found2 = m_mesh_file.find_last_of("_");
		std::string suffix = m_mesh_file.substr(found2);
#ifdef _WIN32
		std::string zcor_file = data_file_path + "\\zCoordinates" + suffix;
#else
	    std::string zcor_file = data_file_path + "/zCoordinates" + suffix;
#endif
		try
		{
			m_zcor_file_ptr = new NetcdfSchismOutput10(zcor_file);
			m_zcor_file_ptr->set_mesh_data_ptr(this->get_mesh_data_ptr());
		}
		catch(...)
		{
			m_zcor_file_ptr = 0;
		}
	}

	
}
 ZCoordMeshProvider10::~ZCoordMeshProvider10()
 {
	 if(m_zcor_file_ptr)
	 {
		// delete m_zcor_file_ptr;
	 }
 }


float  ZCoordMeshProvider10::convertStoZ(const float    & a_sigma,
                                     const float    & a_surface,
                                     const float    & a_depth,
                                     const float    & a_hs,
                                     const float    & a_hc,
                                     const float    & a_thetab,
                                     const float    & a_thetaf) const
{

  float surface = a_surface;

  if (fabs(a_surface-MeshConstants10::DRY_SURFACE)<1.0e-6)
    {
      surface = 0.0;
    }

  float one =1.0;
  float half=0.5;
  float two =2.0;

  float csigma = (one-a_thetab)*sinh(a_thetaf*a_sigma)/sinh(a_thetaf)
  +a_thetab*(tanh(a_thetaf*(a_sigma+half))-tanh(a_thetaf/two))/(two*tanh(a_thetaf/two));
  

  float hat    = a_depth;
  if (a_hs < a_depth)
    {
      hat       = a_hs;
    }
   float z        = MeshConstants10::DRY_SURFACE;
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




//return  z core with node dim the first change then layer
bool ZCoordMeshProvider10::zcoords3D(float * a_zCachePtr,const int & a_timeStep) const
{
	 int timeStart    = a_timeStep;

     SCHISMVar10 * zVarPtr=0;
	 bool outZCoor = false;
	 if (m_dataFilePtr->inquire_var(MeshConstants10::ZCOORD))
	 {
		 zVarPtr = m_dataFilePtr->get_var(MeshConstants10::ZCOORD);
	 }
  
	 else 
     {
	   //first try to load from zcor_*.nc, for output might be in the new scriber format
	  //this zcor file must be at the same folder of current file
		 if (m_zcor_file_ptr)
		 {
			 if (m_zcor_file_ptr->is_valid())
			 {
				 zVarPtr = m_zcor_file_ptr->get_var(MeshConstants10::ZCOORD2);
			 }
			 else
			 {
				 return false;
				// throw SCHISMFileException10("No valid zcor data for file " + m_dataFilePtr->file());
			 }
		 }
		 else
		 {
			 return false;
		 }
	   outZCoor = true;
	  
     }
   
	 if (!(zVarPtr->is_valid()))
	 {
		 return false;
		 //throw SCHISMFileException10("invlaid var " + MeshConstants10::ZCOORD + " for data file " + m_dataFilePtr->file());
	 }
	 SCHISMAtt10* miss_val_ptr = 0;
	 miss_val_ptr = zVarPtr->get_att("missing_value");
	 float missing_val = MeshConstants10::DEGENERATED_Z;
	 if (miss_val_ptr)
	 {
		 missing_val = miss_val_ptr->float_value(0);
	 }
	 float dry_zcor = MeshConstants10::DRY_ZCOR;

     zVarPtr->set_cur(timeStart);

	 long z_var_size =0;
	 long * node_start_index= new long [m_number_node];
	 int * kbp00 = new int [m_number_node];
	 //debug1 << "getting node bottom\n";
	 fillKbp00(kbp00,a_timeStep);
	 //debug1 << "got node bottom\n";
	 for(int iNode=0;iNode<m_number_node;iNode++)
	 {
		 node_start_index[iNode]=z_var_size;
		 z_var_size+=m_number_layer-std::max(1,kbp00[iNode])+1;
	 }

	float * nodeDepthPtr  = new float [m_number_node];
    
 
    retrieve1DVar(nodeDepthPtr,
	              m_dataFilePtr,
                  MeshConstants10::NODE_DEPTH,
                  m_number_node);
	//debug1 << "got node depth\n";
     float*           zPtr = new float [z_var_size];
	 if (outZCoor)
	 {
		 if (!(zVarPtr->get(zPtr,kbp00)))
		 {
			 throw SCHISMFileException10("fail to retrieve var " + MeshConstants10::ZCOORD2 + " from data file " + m_zcor_file_ptr->file());
			 
		 }
	 }
	 else
	 {
		 if (!(zVarPtr->get(zPtr)))
		 {
			 throw SCHISMFileException10("fail to retrieve var " + MeshConstants10::ZCOORD + " from data file " + m_dataFilePtr->file());
			 
		 }
	 }
	 //debug1 << "got node zcor\n";
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
				  else
				  {
					  float surface= MeshConstants10::DUMMY_ELEVATION;
					  if (m_layerSCoords)
					  {
						  float sigma = m_layerSCoords[iLayer];
						  float depth = nodeDepthPtr[iNode];
						  float z = convertStoZ(sigma,
							  surface,
							  depth,
							  m_hs,
							  m_hc,
							  m_thetab,
							  m_thetaf);
						  *a_zCachePtr = z;
					  }
					  else
					  {
						  float depth = nodeDepthPtr[iNode];
						  float z = surface - depth * iLayer / m_number_layer;
						  *a_zCachePtr = z;
					  }
				  }
				  a_zCachePtr++;
			  }

            }
       }
	 delete zPtr;
	 delete kbp00;
	 delete nodeDepthPtr;
	 return true;
}

bool ZCoordMeshProvider10::zcoords3D2(float * a_zCachePtr,const int & a_timeStep) const
{
	 int timeStart    = a_timeStep;

     SCHISMVar10 * zVarPtr=0;
	 bool outZCoor = false;
	 if (m_dataFilePtr->inquire_var(MeshConstants10::ZCOORD))
	 {
		 zVarPtr = m_dataFilePtr->get_var(MeshConstants10::ZCOORD);
	 }

	 else
	 {
		 //first try to load from zcor_*.nc, for output might be in the new scriber format
		//this zcor file must be at the same folder of current file
		 if (m_zcor_file_ptr)
		 {
			 if (m_zcor_file_ptr->is_valid())
			 {
				 zVarPtr = m_zcor_file_ptr->get_var(MeshConstants10::ZCOORD2);
			 }
			 else
			 {
				 //throw SCHISMFileException10("No valid zcor data for file " + m_dataFilePtr->file());
				 return false;
			 }
			 outZCoor = true;
		 }
		 else
		 {
			 return false;
		 }
	 }

	 if (!(zVarPtr->is_valid()))
	 {
		 //throw SCHISMFileException10("invlaid var " + MeshConstants10::ZCOORD + " for data file " + m_dataFilePtr->file());
		 return false;
	 }
   
   
     zVarPtr->set_cur(timeStart);
	 SCHISMAtt10* miss_val_ptr = 0;
	 
	 miss_val_ptr = zVarPtr->get_att("missing_value");
	 float missing_val = MeshConstants10::DEGENERATED_Z;
	 if (miss_val_ptr)
	 {
		 missing_val = miss_val_ptr->float_value(0);
	 }
	 
	 float dry_zcor = MeshConstants10::DRY_ZCOR;
	 float * node_depth_ptr  = new float [m_number_node];
    
 
    retrieve1DVar(node_depth_ptr,
	              m_dataFilePtr,
                  MeshConstants10::NODE_DEPTH,
                  m_number_node);
	//debug1 << "got node depth in zcor3d2\n";
	 long z_var_size =0;
	 long * node_start_index= new long [m_number_node];
	 int * kbp00 = new int [m_number_node];
	 fillKbp00(kbp00,a_timeStep);
	 //debug1 << "got node bottom in zcor3d2\n";
	 for(long iNode=0;iNode<m_number_node;iNode++)
	 {
	    node_start_index[iNode]=z_var_size;	     
		z_var_size+=m_number_layer-std::max(1,kbp00[iNode])+1;
	 }

     float*           zPtr = new float [z_var_size];


	 if (outZCoor)
	 {
		 if (!(zVarPtr->get(zPtr, kbp00)))
		 {
			 throw SCHISMFileException10("fail to retrieve var " + MeshConstants10::ZCOORD2 + " from data file " + m_zcor_file_ptr->file());
		 }
	 }
	 else
	 {
		 if (!(zVarPtr->get(zPtr)))
		 {
			 throw SCHISMFileException10("fail to retrieve var " + MeshConstants10::ZCOORD + " from data file " + m_dataFilePtr->file());
		 }
	 }
	 //debug1 << "got node zcor in zcor3d2\n";
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
					  if (m_layerSCoords)
					  {
						  float sigma = m_layerSCoords[iLayer];
						  float depth = node_depth_ptr[iNode];
						  float z = convertStoZ(sigma,
							  surface,
							  depth,
							  m_hs,
							  m_hc,
							  m_thetab,
							  m_thetaf);
						  zPtr[start_index + iLayer + 1 - std::max(1, kbp00[iNode])] = z;
					  }
					  else
					  {
						  float depth = node_depth_ptr[iNode];
						  float z = surface - depth * iLayer / m_number_layer;
						  zPtr[start_index + iLayer + 1 - std::max(1, kbp00[iNode])] = z;
					  }
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
	 delete node_depth_ptr;
	 return true;
}