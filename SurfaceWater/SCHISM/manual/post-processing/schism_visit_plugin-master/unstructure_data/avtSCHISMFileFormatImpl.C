
// ************************************************************************* //
//                            avtSCHISMFileFormatImpl.C                           //
// ************************************************************************* //

#include <avtSCHISMFileFormatImpl.h>
#include <avtMTSDFileFormat.h>
#include <string>
#include <iostream>
#include <sstream>
#include <time.h>
#include <math.h>

#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkRectilinearGrid.h>
#include <vtkRectilinearGrid.h>
#include <vtkUnstructuredGrid.h>
#include <vtkSmartPointer.h>
#include <vtkCellArray.h>
#include <vtkPolyhedron.h>
#include <vtkPlaneSource.h>
#include <vtkMath.h>
#include <vtkPoints.h>
#include <vtkCellType.h> 

#include <avtDatabaseMetaData.h>
#include <avtVariableCache.h>

#include <DBOptionsAttributes.h>
#include <Expression.h>

#include <InvalidVariableException.h>
#include <InvalidDBTypeException.h>
#include <InvalidTimeStepException.h>
#include <InvalidFilesException.h>
#include <DBYieldedNoDataException.h>
#include <DebugStream.h>
//L3 #include <malloc.h>
#if defined(__MACH__)
#include <stdlib.h>
#else
#include <malloc.h>
#endif

#include "ComputeMeshZProvider.h"
#include "Gr3MeshProvider.h"
#include "ZCoordFileMeshProvider.h"
#include "SCHISMFileUtil.h"
#include "Average.h"
#include "MeshConstants.h"
#include "NetcdfSCHISMOutput.h"
#include "NativeSCHISMOutput.h"
#include "Registar.h"
#include "avtSCHISMFileFormat.h"

using     std::string;
using     std::stringstream;

const std::string NODE      = MeshConstants::NODE;
const std::string FACE      = MeshConstants::ELEM;
const std::string SIDE      = MeshConstants::EDGE;
const std::string UNKOWN    ="unkown";
const int NODESPERELE       = MeshConstants::MAX_NUM_NODE_PER_CELL;
const int NODESPERWEDGE     = NODESPERELE*2;



avtSCHISMFileFormatImpl::avtSCHISMFileFormatImpl():
      m_initialized(false),
      m_plugin_name("SCHISM_output_plugin"),
      m_data_description("data_description"),
      m_mesh_var("Mesh2"),
      m_var_label_att("long_name"),
      m_var_location_att("location"),
      m_mesh_3d("3D_Mesh"),
	  m_layer_mesh("Layer_Mesh"),
      m_mesh_2d("2D_Mesh"),
	  m_side_center_point_3d_mesh("side_center_3D"),
	  m_side_center_point_2d_mesh("side_center_2D"),
	  m_face_center_point_3d_mesh("face_center_3D"),
      m_dim_time(MeshConstants::DIM_TIME),
      m_time(MeshConstants::TIME),
      m_node_depth(MeshConstants::NODE_DEPTH),
      m_node_depth_label(MeshConstants::NODE_DEPTH),
      m_node_surface(MeshConstants::NODE_SURFACE),
      m_node_surface_label("surface_elev"),
      m_dim_layers(MeshConstants::DIM_LAYERS),
	  m_dim_var_component(MeshConstants::DIM_VAR_COMPONENT),
	  m_time_ptr(NULL),
      m_node_x_ptr(NULL),
      m_node_y_ptr(NULL),
	  m_node_z_ptr(NULL),
	  m_kbp00(NULL),
	  m_kbp_data(NULL),
	  m_data_file_ptr(NULL),
      m_surface_state_suffix("_surface"),
      m_bottom_state_suffix("_near_bottom"),
      m_depth_average_suffix("_depth_average"),
      m_dry_surface(-9999.0),
	  m_total_valid_3D_point(0),
	  m_total_valid_3D_side(0),
	  m_total_valid_3D_ele(0)
{
  // AVT_NODECENT, AVT_ZONECENT, AVT_UNKNOWN_CENT
  m_center_map[NODE]  = AVT_NODECENT;
  m_center_map[FACE]  = AVT_ZONECENT;
  m_center_map[UNKOWN]= AVT_UNKNOWN_CENT;

  m_var_name_label_map[m_node_surface_label]  = m_node_surface;
  m_var_name_label_map[m_node_depth_label]    = m_node_depth;

}

FileFormatFavorInterface * avtSCHISMFileFormatImpl::create()
{
	return new avtSCHISMFileFormatImpl();
}


// ****************************************************************************
//  Method: avtEMSTDFileFormat::GetNTimesteps
//
//  Purpose:
//      Tells the rest of the code how many timesteps there are in this file.
//
//  
//   
//
// ****************************************************************************

int
avtSCHISMFileFormatImpl::GetNTimesteps(const std::string& a_filename)
{
  Initialize(a_filename);
  return m_num_time_step;
}


// ****************************************************************************
//  Method: avtSCHISMFileFormatImpl::FreeUpResources
//
//  Purpose:
//      When VisIt is done focusing on a particular timestep, it asks that
//      timestep to free up any resources (memory, file descriptors) that
//      it has associated with it.  This method is the mechanism for doing
//      that.
//
//   
//  
//
// ****************************************************************************

void
avtSCHISMFileFormatImpl::FreeUpResources(void)
{
   if (m_time_ptr)
    {
      delete m_time_ptr;
    }
  
  if (m_node_x_ptr)
    {
      delete m_node_x_ptr;
    }
  if (m_node_y_ptr)
    {
      delete m_node_y_ptr;
    }
   
   if (m_data_file_ptr)
   {
	   delete m_data_file_ptr;
   }
   if(m_kbp00)
   {
	   delete m_kbp00;
   }
   if (m_kbp_data)
   {
	   delete m_kbp_data;
   }
   if (m_external_mesh_provider)
   {
	   delete  m_external_mesh_provider;
   }
  debug1<<"finish free res \n";
}


// ****************************************************************************
//  Method: avtSCHISMFileFormatImpl::PopulateDatabaseMetaData
//
//  Purpose:
//      This database meta-data object is like a table of contents for the
//      file.  By populating it, you are telling the rest of VisIt what
//      information it can request from you.
//
//  
//  
//
// ****************************************************************************

void
avtSCHISMFileFormatImpl::PopulateDatabaseMetaData(avtDatabaseMetaData *a_metaData, avtSCHISMFileFormat * a_avtFile,int a_timeState)
{
   
    //
	avtCentering  nodeCent = AVT_NODECENT;
	avtCentering  zoneCent = AVT_ZONECENT;
    string mesh_name          =  m_mesh_3d;
	
    // AVT_RECTILINEAR_MESH, AVT_CURVILINEAR_MESH, AVT_UNSTRUCTURED_MESH,
    // AVT_POINT_MESH, AVT_SURFACE_MESH, AVT_UNKNOWN_MESH
    avtMeshType mt            = AVT_UNSTRUCTURED_MESH;
    //
    int nblocks               = 1; 
    int block_origin          = 0;
    int spatial_dimension     = 3;
    int topological_dimension = 3;
    double *extents           = NULL;
   
	if (m_external_mesh_provider->provide3DMesh())
	{
      a_avtFile->addMeshToMetaData(a_metaData, 
                        mesh_name, 
                        mt, 
                        extents, 
                        nblocks, 
                        block_origin,
                        spatial_dimension, 
                        topological_dimension);
	}

	 //
    // add layered 2d mesh
	mesh_name              =  m_layer_mesh;
    topological_dimension = 2;
   
	if (m_external_mesh_provider->provide3DMesh())
	{
      a_avtFile->addMeshToMetaData(a_metaData, 
                        mesh_name, 
                        mt, 
                        extents, 
                        nblocks, 
                        block_origin,
                        spatial_dimension, 
                        topological_dimension);
	}

    //
    // add surface 2d mesh
    //
    mesh_name                =  m_mesh_2d;
	
    spatial_dimension       =  2;
    topological_dimension   =  2;

    a_avtFile->addMeshToMetaData(a_metaData, 
                      mesh_name, 
                      mt, 
                      extents, 
                      nblocks, 
                      block_origin,
                      spatial_dimension, 
                      topological_dimension);

	// add 3d side center point mesh
	mesh_name                =   m_side_center_point_3d_mesh ;
	
    spatial_dimension       =  3;
    topological_dimension   =  0;

	if (m_external_mesh_provider->provide3DMesh())
	{
     a_avtFile->addMeshToMetaData(a_metaData, 
                      mesh_name, 
                      mt, 
                      extents, 
                      nblocks, 
                      block_origin,
                      spatial_dimension, 
                      topological_dimension);
	}

	// add 3d side face center point mesh
	mesh_name                =   m_face_center_point_3d_mesh ;
	
    spatial_dimension       =  3;
    topological_dimension   =  0;

	if (m_external_mesh_provider->provide3DMesh())
	{
     a_avtFile->addMeshToMetaData(a_metaData, 
                      mesh_name, 
                      mt, 
                      extents, 
                      nblocks, 
                      block_origin,
                      spatial_dimension, 
                      topological_dimension);
	}

	// add 2d side center point mesh
	mesh_name                =   m_side_center_point_2d_mesh ;
	
    spatial_dimension       =  2;
    topological_dimension   =  0;

    a_avtFile->addMeshToMetaData(a_metaData, 
                      mesh_name, 
                      mt, 
                      extents, 
                      nblocks, 
                      block_origin,
                      spatial_dimension, 
                      topological_dimension);

    //
    // add water surface and depth scalar
    string mesh            = m_mesh_2d;
   
	if (m_data_center!= MeshConstants::ELEM )
	{
	   if(m_data_center == MeshConstants::EDGE)
	   {
		   mesh = m_side_center_point_2d_mesh;
	   }
       a_avtFile->addScalarVarToMetaData(a_metaData, m_node_surface_label, mesh, nodeCent);
       a_avtFile->addScalarVarToMetaData(a_metaData, m_node_depth_label,   mesh, nodeCent);
	}
	else
	{
	   a_avtFile->addScalarVarToMetaData(a_metaData, m_node_surface_label, mesh, zoneCent);
       a_avtFile->addScalarVarToMetaData(a_metaData, m_node_depth_label,   mesh, zoneCent);
	}


    m_var_mesh_map[m_node_surface_label] = mesh;
    m_var_mesh_map[m_node_depth_label]   = mesh;

	//add 3D node layer label

    
    PopulateStateMetaData(a_metaData,a_avtFile,a_timeState);
    debug1<<"finish populate metadata \n";
}

void    avtSCHISMFileFormatImpl::addFaceCenterData(avtDatabaseMetaData * a_metaData,
	                                                SCHISMVar            * a_varPtr,
													avtSCHISMFileFormat * a_avtFile,
								                    const std::string   & a_varName,
								                    const std::string   & a_varLabel,
								                    const avtCentering  & a_center)

{
	  // only add face centered  var now 
       string mesh2d     = m_mesh_2d;
	   string mesh3d     = m_mesh_3d;
	   if (m_level_center == MeshConstants::FULL_LAYER)
	   {
		   mesh3d = m_layer_mesh;
	   }
      
       avtCentering  faceCent(AVT_ZONECENT);
       // scalar data 2d mesh
       if (a_varPtr->num_dims()==2)
        {     
          a_avtFile->addScalarVarToMetaData(a_metaData, a_varLabel, mesh2d, faceCent);
          m_var_mesh_map[a_varLabel] = mesh2d;
		  m_var_dim[a_varName]=2;
        }
       // vector data
       else if (SCHISMVarIs3D(a_varPtr))
        {   
			if (!(m_external_mesh_provider->provide3DMesh()))
	        {
				stringstream msgStream(stringstream::out);
                msgStream <<"3D variable "<<a_varLabel<<" is not supported by a 2D meshprovider\n";
			    EXCEPTION1(InvalidVariableException,msgStream.str());
			}
		   if (SCHISMVarIsVector(a_varPtr)) // 3d vector
		   {
               // last dim is vector component         
              SCHISMDim* comDim = a_varPtr->get_dim(3);
              int ncomps       = comDim->size();
              int ucomps       = (ncomps == 2 ? 3 : ncomps);

              a_avtFile->addVectorVarToMetaData(a_metaData,a_varLabel, mesh3d, faceCent,ucomps);  
              m_var_mesh_map[a_varLabel] = mesh3d; 
			  m_var_dim[a_varName]=3;

			   // also add bottom, surface and depth average state option
              a_avtFile->addVectorVarToMetaData(a_metaData,
                                 a_varLabel+m_surface_state_suffix, 
                                 m_mesh_2d, 
                                 faceCent,
                                 ucomps); 
              //debug1<<"add  "<<label+m_surface_state_suffix<<" ";
              a_avtFile->addVectorVarToMetaData(a_metaData,
                                 a_varLabel+m_bottom_state_suffix, 
                                 m_mesh_2d, 
                                 faceCent,
                                 ucomps); 
              //debug1<<"add  "<<label+m_bottom_state_suffix<<" ";
              a_avtFile->addVectorVarToMetaData(a_metaData,
                                 a_varLabel+m_depth_average_suffix, 
                                 m_mesh_2d, 
                                 faceCent,
                                 ucomps); 
            // debug1<<"add  "<< label+m_depth_average_suffix<<" ";
             m_var_name_label_map[ a_varLabel+m_surface_state_suffix] = a_varName;
             m_var_name_label_map[ a_varLabel+m_bottom_state_suffix]  = a_varName;
             m_var_name_label_map[ a_varLabel+m_depth_average_suffix] = a_varName;
             m_var_mesh_map[ a_varLabel+m_surface_state_suffix ]     = m_mesh_2d;
             m_var_mesh_map[ a_varLabel+m_bottom_state_suffix ]      = m_mesh_2d;
             m_var_mesh_map[ a_varLabel+m_depth_average_suffix ]     = m_mesh_2d;
		   }
		   else //3d scalar
		   {
			  a_avtFile->addScalarVarToMetaData(a_metaData,a_varLabel, mesh3d, faceCent);  
              m_var_mesh_map[a_varLabel] = mesh3d;

			  // also add bottom, surface and depth average state option
              a_avtFile->addScalarVarToMetaData(a_metaData,
                                     a_varLabel+m_surface_state_suffix, 
                                     m_mesh_2d, 
                                     faceCent); 
              //debug1<<"add  "<<label+m_surface_state_suffix<<" ";
              a_avtFile->addScalarVarToMetaData(a_metaData,
                                    a_varLabel+m_bottom_state_suffix, 
                                    m_mesh_2d, 
                                    faceCent); 
              //debug1<<"add  "<<label+m_bottom_state_suffix<<" ";
              a_avtFile->addScalarVarToMetaData(a_metaData,
                                    a_varLabel+m_depth_average_suffix, 
                                    m_mesh_2d, 
                                    faceCent); 
              //debug1<<"add  "<< label+m_depth_average_suffix<<" ";
              // all those surface, botootm and average are based on original data set
              m_var_name_label_map[ a_varLabel+m_surface_state_suffix] = a_varName;
              m_var_name_label_map[ a_varLabel+m_bottom_state_suffix]  = a_varName;
              m_var_name_label_map[ a_varLabel+m_depth_average_suffix] = a_varName;

              m_var_mesh_map[ a_varLabel+m_surface_state_suffix ]     = m_mesh_2d;
              m_var_mesh_map[ a_varLabel+m_bottom_state_suffix ]      = m_mesh_2d;
              m_var_mesh_map[ a_varLabel+m_depth_average_suffix ]     = m_mesh_2d;
			  m_var_dim[a_varName]=3;

		   }
		    a_avtFile->addScalarVarToMetaData(a_metaData,  MeshConstants::LEVEL, mesh3d, faceCent);
        }
	   else
	   {

	   }


}
void    avtSCHISMFileFormatImpl::addNodeCenterData(avtDatabaseMetaData * a_metaData,
	                                                SCHISMVar            * a_varPtr,
													 avtSCHISMFileFormat * a_avtFile,
								                    const std::string   & a_varName,
								                    const std::string   & a_varLabel,
								                    const avtCentering  & a_center)
{

	  std::string varName(a_varName);
	  std::string label(a_varLabel);
	  avtCentering avtCenter(a_center);

	 //  scalar var on 2D
	   if (a_varPtr->num_dims()==2)
	   {
		  a_avtFile->addScalarVarToMetaData(a_metaData,label, m_mesh_2d, avtCenter);   
          m_var_mesh_map[label] = m_mesh_2d;
		   m_var_dim[varName]=2;
		  debug1<<"added 2d scalar:"<<label;
	   }
	    //  vector var on 2D
	   else if ((a_varPtr->num_dims()==3) && (!SCHISMVarIs3D(a_varPtr)))
	   {
		  SCHISMDim* comDim = a_varPtr->get_dim(3);
          int ncomps       = comDim->size();
          int ucomps       = (ncomps == 2 ? 3 : ncomps);
		  a_avtFile->addVectorVarToMetaData(a_metaData,label, m_mesh_2d, avtCenter,ucomps);   
          m_var_mesh_map[label] = m_mesh_2d;
		  m_var_dim[varName]=2;
		  
	   }
	    //  scalar var having layer dim
       else if ((a_varPtr->num_dims()==3) && (SCHISMVarIs3D(a_varPtr)))
        {
            if (!(m_external_mesh_provider->provide3DMesh()))
	        {
				stringstream msgStream(stringstream::out);
                msgStream <<"3D variable "<<a_varLabel<<" is not supported by a 2D meshprovider\n";
			    EXCEPTION1(InvalidVariableException,msgStream.str());
			}

		  a_avtFile->addScalarVarToMetaData(a_metaData,  MeshConstants::LEVEL, m_mesh_3d, avtCenter);
          a_avtFile->addScalarVarToMetaData(a_metaData,label, m_mesh_3d, avtCenter);   
          m_var_mesh_map[label] = m_mesh_3d;
          // also add bottom, surface and depth average state option
          a_avtFile->addScalarVarToMetaData(a_metaData,
                                 label+m_surface_state_suffix, 
                                 m_mesh_2d, 
                                 avtCenter); 
          debug1<<"add  "<<label+m_surface_state_suffix<<" ";
          a_avtFile->addScalarVarToMetaData(a_metaData,
                                 label+m_bottom_state_suffix, 
                                 m_mesh_2d, 
                                 avtCenter); 
          debug1<<"add  "<<label+m_bottom_state_suffix<<" ";
          a_avtFile->addScalarVarToMetaData(a_metaData,
                                 label+m_depth_average_suffix, 
                                 m_mesh_2d, 
                                 avtCenter); 
          debug1<<"add  "<< label+m_depth_average_suffix<<" ";
          // all those surface, botootm and average are based on original data set
          m_var_name_label_map[ label+m_surface_state_suffix] = varName;
          m_var_name_label_map[ label+m_bottom_state_suffix]  = varName;
          m_var_name_label_map[ label+m_depth_average_suffix] = varName;

          m_var_mesh_map[ label+m_surface_state_suffix ]     = m_mesh_2d;
          m_var_mesh_map[ label+m_bottom_state_suffix ]      = m_mesh_2d;
          m_var_mesh_map[ label+m_depth_average_suffix ]     = m_mesh_2d;
		  m_var_dim[varName]=3;
      
        }
       else if ((a_varPtr->num_dims()==4) && (SCHISMVarIs3D(a_varPtr)))
        {
           if (!(m_external_mesh_provider->provide3DMesh()))
	        {
				stringstream msgStream(stringstream::out);
                msgStream <<"3D variable "<<a_varLabel<<" is not supported by a 2D meshprovider\n";
			    EXCEPTION1(InvalidVariableException,msgStream.str());
			}
          // last dim is vector component         
          SCHISMDim* comDim = a_varPtr->get_dim(3);
          int ncomps       = comDim->size();
          int ucomps       = (ncomps == 2 ? 3 : ncomps);

          a_avtFile->addVectorVarToMetaData(a_metaData,label, m_mesh_3d, avtCenter,ucomps);   
		  a_avtFile->addScalarVarToMetaData(a_metaData,  MeshConstants::LEVEL, m_mesh_3d, avtCenter);
          m_var_mesh_map[label] = m_mesh_3d;

          // also add bottom, surface and depth average state option
          a_avtFile->addVectorVarToMetaData(a_metaData,
                                 label+m_surface_state_suffix, 
                                 m_mesh_2d, 
                                 avtCenter,
                                 ucomps); 
          debug1<<"add  "<<label+m_surface_state_suffix<<" ";
          a_avtFile->addVectorVarToMetaData(a_metaData,
                                 label+m_bottom_state_suffix, 
                                 m_mesh_2d, 
                                 avtCenter,
                                 ucomps); 
          debug1<<"add  "<<label+m_bottom_state_suffix<<" ";
          a_avtFile->addVectorVarToMetaData(a_metaData,
                                 label+m_depth_average_suffix, 
                                 m_mesh_2d, 
                                 avtCenter,
                                 ucomps); 
          debug1<<"add  "<< label+m_depth_average_suffix<<" ";
          m_var_name_label_map[ label+m_surface_state_suffix] = varName;
          m_var_name_label_map[ label+m_bottom_state_suffix]  = varName;
          m_var_name_label_map[ label+m_depth_average_suffix] = varName;
          m_var_mesh_map[ label+m_surface_state_suffix ]     = m_mesh_2d;
          m_var_mesh_map[ label+m_bottom_state_suffix ]      = m_mesh_2d;
          m_var_mesh_map[ label+m_depth_average_suffix ]     = m_mesh_2d;
		  m_var_dim[varName]=3;

       } 
	   else
	   {

	   }
}

void    avtSCHISMFileFormatImpl::addSideCenterData(avtDatabaseMetaData * a_metaData,
	                                                SCHISMVar            * a_varPtr,
													 avtSCHISMFileFormat * a_avtFile,
								                    const std::string   & a_varName,
								                    const std::string   & a_varLabel,
								                    const avtCentering  & a_center)
{
	  // only add face centered  var now 
       string mesh2d     =  m_side_center_point_2d_mesh;
	   string mesh3d     =  m_side_center_point_3d_mesh;


       if (m_level_center == MeshConstants::HALF_LAYER)
	   {
		   mesh3d = m_face_center_point_3d_mesh;
	   }

       avtCentering  nodeCent(AVT_NODECENT);
       std::string varName(a_varName);
	   std::string label(a_varLabel);
	   avtCentering avtCenter(a_center);

	 //  scalar var on 2D
	   if (a_varPtr->num_dims()==2)
	   {
		  a_avtFile->addScalarVarToMetaData(a_metaData,label, mesh2d, avtCenter);   
          m_var_mesh_map[label] = mesh2d;
		  m_var_dim[varName]=2;
		  debug1<<"added 2d scalar:"<<label;
	   }
	    //  vector var on 2D
	   else if ((a_varPtr->num_dims()==3) && (!SCHISMVarIs3D(a_varPtr)))
	   {
		  SCHISMDim* comDim = a_varPtr->get_dim(3);
          int ncomps       = comDim->size();
          int ucomps       = (ncomps == 2 ? 3 : ncomps);
		  a_avtFile->addVectorVarToMetaData(a_metaData,label, mesh2d, avtCenter,ucomps);   
          m_var_mesh_map[label] = mesh2d;
		  m_var_dim[varName]=2;
		  
	   }
	    //  scalar var having layer dim
       else if ((a_varPtr->num_dims()==3) && (SCHISMVarIs3D(a_varPtr)))
        {
         
		   if (!(m_external_mesh_provider->provide3DMesh()))
	        {
				stringstream msgStream(stringstream::out);
                msgStream <<"3D variable "<<a_varLabel<<" is not supported by a 2D meshprovider\n";
			    EXCEPTION1(InvalidVariableException,msgStream.str());
			}
		  a_avtFile->addScalarVarToMetaData(a_metaData,  MeshConstants::LEVEL, mesh3d, avtCenter);
          a_avtFile->addScalarVarToMetaData(a_metaData,label, mesh3d, avtCenter);   
          m_var_mesh_map[label] = mesh3d;
          // also add bottom, surface and depth average state option
          a_avtFile->addScalarVarToMetaData(a_metaData,
                                 label+m_surface_state_suffix, 
                                 mesh2d, 
                                 avtCenter); 
          debug1<<"add  "<<label+m_surface_state_suffix<<" ";
          a_avtFile->addScalarVarToMetaData(a_metaData,
                                 label+m_bottom_state_suffix, 
                                 mesh2d, 
                                 avtCenter); 
          debug1<<"add  "<<label+m_bottom_state_suffix<<" ";
          a_avtFile->addScalarVarToMetaData(a_metaData,
                                 label+m_depth_average_suffix, 
                                 mesh2d, 
                                 avtCenter); 
          debug1<<"add  "<< label+m_depth_average_suffix<<" ";
          // all those surface, botootm and average are based on original data set
          m_var_name_label_map[ label+m_surface_state_suffix] = varName;
          m_var_name_label_map[ label+m_bottom_state_suffix]  = varName;
          m_var_name_label_map[ label+m_depth_average_suffix] = varName;

          m_var_mesh_map[ label+m_surface_state_suffix ]     = mesh2d;
          m_var_mesh_map[ label+m_bottom_state_suffix ]      = mesh2d;
          m_var_mesh_map[ label+m_depth_average_suffix ]     = mesh2d;
		  m_var_dim[varName]=3;
      
        }
       else if ((a_varPtr->num_dims()==4) && (SCHISMVarIs3D(a_varPtr)))
        {
           if (!(m_external_mesh_provider->provide3DMesh()))
	        {
				stringstream msgStream(stringstream::out);
                msgStream <<"3D variable "<<a_varLabel<<" is not supported by a 2D meshprovider\n";
			    EXCEPTION1(InvalidVariableException,msgStream.str());
			}
          // last dim is vector component         
          SCHISMDim* comDim = a_varPtr->get_dim(3);
          int ncomps       = comDim->size();
          int ucomps       = (ncomps == 2 ? 3 : ncomps);

          a_avtFile->addVectorVarToMetaData(a_metaData,label, mesh3d, avtCenter,ucomps);  
		  a_avtFile->addScalarVarToMetaData(a_metaData,  MeshConstants::LEVEL, mesh3d, avtCenter);
          m_var_mesh_map[label] = mesh3d;

          // also add bottom, surface and depth average state option
          a_avtFile->addVectorVarToMetaData(a_metaData,
                                 label+m_surface_state_suffix, 
                                 mesh2d, 
                                 avtCenter,
                                 ucomps); 
          debug1<<"add  "<<label+m_surface_state_suffix<<" ";
          a_avtFile->addVectorVarToMetaData(a_metaData,
                                 label+m_bottom_state_suffix, 
                                 mesh2d, 
                                 avtCenter,
                                 ucomps); 
          debug1<<"add  "<<label+m_bottom_state_suffix<<" ";
          a_avtFile->addVectorVarToMetaData(a_metaData,
                                 label+m_depth_average_suffix, 
                                 mesh2d, 
                                 avtCenter,
                                 ucomps); 
          debug1<<"add  "<< label+m_depth_average_suffix<<" ";
          m_var_name_label_map[ label+m_surface_state_suffix] = varName;
          m_var_name_label_map[ label+m_bottom_state_suffix]  = varName;
          m_var_name_label_map[ label+m_depth_average_suffix] = varName;
          m_var_mesh_map[ label+m_surface_state_suffix ]     = mesh2d;
          m_var_mesh_map[ label+m_bottom_state_suffix ]      = mesh2d;
          m_var_mesh_map[ label+m_depth_average_suffix ]     = mesh2d;
		  m_var_dim[varName]=3;
       } 
	   else
	   {
	   }

}

// ****************************************************************************
//  Method: avtSCHISMFileFormatImpl::PopulateStateMetaData
//
//  Purpose:
//      Scan the data file and find out all the state variables, 
//      store the name, mesh, and centering information into
//      the input metadata set.
//
//  Programmer: qshu
//  Creation:   Wed Aug 30 08:11:04 PDT 2012
//
// ****************************************************************************

void avtSCHISMFileFormatImpl::PopulateStateMetaData(avtDatabaseMetaData * a_metaData, 
	                                                avtSCHISMFileFormat * a_avtFile,
                                                     int                   a_timeState)
{
  int numVar = m_data_file_ptr->num_vars();
  debug1<<"get vars "<<numVar<<endl;
  for(int iVar = 0;iVar < numVar; iVar++)
   {
     
      debug1<<iVar;
      SCHISMVar*  varPtr  = m_data_file_ptr->get_var(iVar);
      debug1<<" "<<varPtr->num_dims()<<endl;

      std::string varName = varPtr->name();
	  if (m_data_file_ptr->none_data_var(varName))
	  {
		  debug1<<varName<<"is skipped\n";
		  continue;
	  }
      debug1<<varName<<endl;
     
      std::string  location(NODE);
      avtCentering avtCenter(AVT_NODECENT);
     
      if ((varName==m_node_surface) || (varName==m_node_depth))
        {
          continue; 
        }
      std::string  label;
      label = varName;
      // this dic make it easy to find out data set for a visit plot variable
      m_var_name_label_map[label] = varName;

      // handle different for face and node center data
	 location = m_data_center;
     if(location ==FACE)
     {
	   addFaceCenterData(a_metaData,varPtr,a_avtFile,varName,label,avtCenter);
     }
     else if (location ==NODE)
     {
		addNodeCenterData(a_metaData,varPtr,a_avtFile,varName,label,avtCenter);  
     }
	 else if (location ==SIDE)
	 {
		 addSideCenterData(a_metaData,varPtr,a_avtFile,varName,label,avtCenter);  
	 }
    // omit unkown center data
    else
     {
      continue;
     }
   }
}


void   avtSCHISMFileFormatImpl::create2DUnstructuredMesh( vtkUnstructuredGrid *a_uGrid,
	                                                       int                 *a_meshEle,
														   const  int          &a_timeState) 
{
	int   numNodes           = m_num_mesh_nodes;
	vtkPoints *points      = vtkPoints::New();
    points->SetNumberOfPoints(numNodes);
    float * pointPtr       = (float *) points->GetVoidPointer(0);
        
    if (!m_external_mesh_provider->fillPointCoord2D(pointPtr,a_timeState))
    {
        stringstream msgStream(stringstream::out);
        msgStream <<"Fail to retrieve faces nodes coord at step " <<a_timeState;
        EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
    }


    a_uGrid ->SetPoints(points);
    points->Delete();
    a_uGrid ->Allocate( m_num_mesh_faces);
       
    int *  nodePtrTemp = a_meshEle;
    for(int iCell = 0; iCell < m_num_mesh_faces; ++iCell)
        {
			int numberOfNodeInCell = *nodePtrTemp;
			 
			if (numberOfNodeInCell ==3)
			{
            vtkIdType verts[3];
            for(int iNode=0;iNode<3;++iNode)
            {
                verts[iNode] = nodePtrTemp[iNode+1]-1;
				    
            } 
            nodePtrTemp += (MeshConstants::MAX_NUM_NODE_PER_CELL+1) ;
				 
            a_uGrid->InsertNextCell(VTK_TRIANGLE, 3, verts);
			}
			else if (numberOfNodeInCell ==4)
			{
			vtkIdType verts[4];
            for(int iNode=0;iNode<4;++iNode)
            {
                verts[iNode] = nodePtrTemp[iNode+1]-1;
				  
            } 
            nodePtrTemp += (MeshConstants::MAX_NUM_NODE_PER_CELL+1);
				 
            a_uGrid->InsertNextCell(VTK_QUAD, 4, verts);
			}
			else
		    {
			  stringstream msgStream(stringstream::out);
              msgStream <<"invalid cell type with number of nodes: " <<numberOfNodeInCell;
			  EXCEPTION1(InvalidVariableException,msgStream.str());
		    }
             
        }
      
}

void   avtSCHISMFileFormatImpl::createLayerMesh(vtkUnstructuredGrid *a_uGrid,
	                                        int                 *a_meshEle,
										    int                 *a_2DPointto3DPoints,
										    const  int          &a_timeState) 
{
	 vtkPoints *points      = vtkPoints::New();
      points->SetNumberOfPoints(m_total_valid_3D_point);
      float * pointPtr       = (float *) points->GetVoidPointer(0);
	  //debug only
	 
	  if (!m_external_mesh_provider->fillPointCoord3D(pointPtr,a_timeState))
        {
          stringstream msgStream(stringstream::out);
          msgStream <<"Fail to retrieve faces nodes coord at step " <<a_timeState;
          EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
        }
	
      a_uGrid->SetPoints(points);
      points->Delete();
      a_uGrid->Allocate( m_num_mesh_faces*m_num_layers);
      
     
	  int * kbe = new int [m_num_mesh_faces];
	  m_external_mesh_provider->fillKbe(kbe);
	 
	  debug1<<"test layer mesh ele\n";
      for (int iLayer= 0; iLayer<m_num_layers;iLayer++)
        {
		 
          for(int iCell = 0; iCell < m_num_mesh_faces; ++iCell)
            {
			
			  if (iLayer>=(std::max(1,kbe[iCell])-1))
			  {
				
				  int numberOfNodeInCell = a_meshEle[iCell*(NODESPERELE+1)];

				  if (numberOfNodeInCell ==3)
				  {
					 vtkIdType verts[3]; 

					 for(int i=0;i<3;i++)
					 {
						int p = a_meshEle[iCell*(NODESPERELE+1)+i+1]-1;
						int p3d =  a_2DPointto3DPoints[p*m_num_layers+iLayer];
						int valid_bottom = std::max(1,m_kbp00[p])-1;
						if (iLayer<valid_bottom)
						{
							p3d = a_2DPointto3DPoints[p*m_num_layers+valid_bottom];
						}

						verts[i]=p3d;
					 }

					 a_uGrid->InsertNextCell(VTK_TRIANGLE, 3, verts);
				  }
				  else if  (numberOfNodeInCell ==4)
				  {
					 vtkIdType verts[4]; 
					 for(int i=0;i<4;i++)
					 {
						int p = a_meshEle[iCell*(NODESPERELE+1)+i+1]-1;
						int p3d =  a_2DPointto3DPoints[p*m_num_layers+iLayer];
						int valid_bottom = std::max(1,m_kbp00[p])-1;
						if (iLayer<valid_bottom)
						{
							p3d = a_2DPointto3DPoints[p*m_num_layers+valid_bottom];
						}

						verts[i]=p3d;
					 }
					 a_uGrid->InsertNextCell(VTK_QUAD, 4, verts);
				  }
				  else
		          {
			        stringstream msgStream(stringstream::out);
                    msgStream <<"invalid cell type with number of nodes: " <<numberOfNodeInCell;
			        EXCEPTION1(InvalidVariableException,msgStream.str());
		          }
			  }
			  
            }
		 
        }
	 
	  delete kbe;
}


void   avtSCHISMFileFormatImpl::create3DUnstructuredMesh(vtkUnstructuredGrid *a_uGrid,
	                                                 int                 *a_meshEle,
												     int                 *a_2DPointto3DPoints,
												     const  int          &a_timeState) 
{
	 vtkPoints *points      = vtkPoints::New();
      points->SetNumberOfPoints(m_total_valid_3D_point);
      float * pointPtr       = (float *) points->GetVoidPointer(0);

	  if (!m_external_mesh_provider->fillPointCoord3D(pointPtr,a_timeState))
        {
          stringstream msgStream(stringstream::out);
          msgStream <<"Fail to retrieve faces nodes coord at step " <<a_timeState;
          EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
        }
	 
      debug1<<"finish compute and cahce z\n";
      a_uGrid->SetPoints(points);
      points->Delete();
      a_uGrid->Allocate( m_num_mesh_faces*(m_num_layers-1));
      debug1<<"total 2d cell num "<<m_num_mesh_faces<<"\n";
      int *  nodePtrTemp = a_meshEle;
	  
	  m_tri_wedge=0;
      m_tri_pyramid=0;
      m_tri_tetra=0;

      m_quad_hexhedron=0;
      m_quad_wedge=0;
      m_quad_pyramid=0;

      for (int iLayer= 0; iLayer<m_num_layers-1;iLayer++)
        {
          nodePtrTemp    = a_meshEle;
            for(int iCell = 0; iCell < m_num_mesh_faces; ++iCell)
            //for(int iCell = 169309; iCell < 169310; ++iCell)
            {
			  nodePtrTemp = a_meshEle+(MeshConstants::MAX_NUM_NODE_PER_CELL+1)*iCell;
			  int numberOfNodeInCell = *nodePtrTemp;

			  int validTopNode[MeshConstants::MAX_NUM_NODE_PER_CELL];
			  int validBottomNode[MeshConstants::MAX_NUM_NODE_PER_CELL];
			  int validTopNodeNum    =0;
			  int validBottomNodeNum =0;

	          validTopBottomNode(validTopNodeNum,
	                             validBottomNodeNum,
								 validTopNode,
								 validBottomNode,
		                         iLayer,
							     nodePtrTemp);
			  if (numberOfNodeInCell ==3)   
			  {
				 insertTriangle3DCell(a_uGrid,
	                                  validTopNodeNum,
									  validBottomNodeNum,
									  validTopNode,
									  validBottomNode,
									  nodePtrTemp,
									  a_2DPointto3DPoints,
					                  iCell,
	                                  iLayer);
				 //move pointer to next element
				 //nodePtrTemp += (MeshConstants::MAX_NUM_NODE_PER_CELL+1);
			 }
			 else if (numberOfNodeInCell ==4)
			 {
				  insertQuad3DCell(a_uGrid,
	                               validTopNodeNum,
								   validBottomNodeNum,
									  validTopNode,
									  validBottomNode,
									  nodePtrTemp,
									  a_2DPointto3DPoints,
					                  iCell,
	                                  iLayer);
                // nodePtrTemp += (MeshConstants::MAX_NUM_NODE_PER_CELL+1);
			  }
                
			 else
		     {
			   //omit
		     }
		  }
           
        }

	  debug1<<" tri_wedge "<<m_tri_wedge<<" tri_pyramid "<<m_tri_pyramid<<" tri_tetra "<<m_tri_tetra;

	  debug1<<" quad_hexhedron "<<m_quad_hexhedron<<" quad_wedge "<<m_quad_wedge<<" quad_pyramid "<<m_quad_pyramid<<"\n";
}

void    avtSCHISMFileFormatImpl::create2DPointMesh( vtkUnstructuredGrid *a_uGrid,
	                                            int                 *a_meshEle,
										        const  int          &a_timeState) 
{
	int   numNodes           = m_num_mesh_edges;
	vtkPoints *points      = vtkPoints::New();
    points->SetNumberOfPoints(numNodes);
    float * pointPtr       = (float *) points->GetVoidPointer(0);
        
    if (!m_external_mesh_provider->fillSideCenterCoord2D(pointPtr,a_timeState))
    {
        stringstream msgStream(stringstream::out);
        msgStream <<"Fail to retrieve edge center coord at step " <<a_timeState;
        EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
    }


    a_uGrid ->SetPoints(points);
    points->Delete();
	a_uGrid->Allocate(numNodes);
    vtkIdType onevertex;
    for(int i = 0; i < numNodes; ++i)
    {
       onevertex = i;
       a_uGrid->InsertNextCell(VTK_VERTEX, 1, &onevertex);
    }
 
}

void   avtSCHISMFileFormatImpl::create3DPointMesh( vtkUnstructuredGrid *a_uGrid,
	                                                int            *a_meshEle,
										            const  int     &a_timeState) 
{
	int   numNodes           = m_total_valid_3D_side;
	vtkPoints *points      = vtkPoints::New();
    points->SetNumberOfPoints(numNodes);
    float * pointPtr       = (float *) points->GetVoidPointer(0);
        
    if (!m_external_mesh_provider->fillSideCenterCoord3D(pointPtr,a_timeState))
    {
        stringstream msgStream(stringstream::out);
        msgStream <<"Fail to retrieve edge center coord at step " <<a_timeState;
        EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
    }

    a_uGrid ->SetPoints(points);
    points->Delete();
	a_uGrid->Allocate(numNodes);
    vtkIdType onevertex;
    for(int i = 0; i < numNodes; ++i)
     { 
	   onevertex = i;
       a_uGrid->InsertNextCell(VTK_VERTEX, 1, &onevertex);
    }
 
}

void   avtSCHISMFileFormatImpl::create3DPointFaceMesh( vtkUnstructuredGrid *a_uGrid,
	                                                int            *a_meshEle,
										            const  int     &a_timeState) 
{
	int   numNodes           = m_total_valid_3D_side-(m_external_mesh_provider->numberOfSide());
	vtkPoints *points      = vtkPoints::New();
    points->SetNumberOfPoints(numNodes);
    float * pointPtr       = (float *) points->GetVoidPointer(0);
        
    if (!m_external_mesh_provider->fillSideFaceCenterCoord3D(pointPtr,a_timeState))
    {
        stringstream msgStream(stringstream::out);
        msgStream <<"Fail to retrieve edge center coord at step " <<a_timeState;
        EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
    }

    a_uGrid ->SetPoints(points);
    points->Delete();
	a_uGrid->Allocate(numNodes);
    vtkIdType onevertex;
    for(int i = 0; i < numNodes; ++i)
    {
       onevertex = i;
       a_uGrid->InsertNextCell(VTK_VERTEX, 1, &onevertex);
    }
 
}

// ****************************************************************************
//  Method: avtSCHISMFileFormatImpl::GetMesh
//
//  Purpose:
//      Gets the mesh associated with this file.  The mesh is returned as a
//      derived type of vtkDataSet (ie vtkRectilinearGrid, vtkStructuredGrid,
//      vtkUnstructuredGrid, etc).
//
//  Arguments:+
//      timestate   The index of the timestate.  If GetNTimesteps returned
//                  'N' time steps, this is guaranteed to be between 0 and N-1.
//      mesh_name    The name of the mesh of interest.  This can be ignored if
//                  there is only one mesh.
//
//  Programmer: qshu -- generated by xml2avt
//  Creation:   Wed Mar 13 09:13:49 PDT 2013
//
// ****************************************************************************

vtkDataSet *
avtSCHISMFileFormatImpl::GetMesh(int a_timeState, avtSCHISMFileFormat * a_avtFile,const char *mesh_name)
{
  int   nDims              = 3;
  int   numNodes           = m_num_mesh_nodes;
  int   numCells           = m_num_mesh_faces;

  time_t startTicks        = clock();
 
  int   domainID           = 0;
  int   timeState          = 0;
  std::string material("all");
  std::string cacheMeshID(mesh_name);
 // cacheMeshID             += m_data_file;  
  debug1<<" try to find "<<cacheMeshID<<" in cache\n";
  vtkObject * cachedMesh=NULL;
  cachedMesh   = (a_avtFile->get_cache())->GetVTKObject(cacheMeshID.c_str(),
                                                 avtVariableCache::DATASET_NAME,
                                                 a_timeState, 
                                                 domainID, 
                                                 material.c_str());
  if(cachedMesh!=NULL)
    {
      vtkUnstructuredGrid *uGrid = (vtkUnstructuredGrid *)cachedMesh;
      uGrid->Register(NULL);
      debug1<<"get "<<mesh_name<<" from cache"<<endl;
      time_t endTicks      = clock();
      debug1<<"time used in building mesh :"<<endTicks-startTicks<<endl;
      updateMeshZCoordinates(uGrid->GetPoints(),
                             a_timeState,
                             mesh_name);
      return uGrid;
    }
  debug1 << mesh_name << " not in cache. Load from data." << endl;

  // get face nodes
  int    numNodesPerFace        = NODESPERELE;
  int *  faceNodesPtr           = new int  [m_num_mesh_faces*(numNodesPerFace+1)];
 
  if (!m_external_mesh_provider->fillMeshElement(faceNodesPtr))
    {
      stringstream msgStream(stringstream::out);
      msgStream <<"Fail to retrieve faces nodes at step " <<a_timeState;
      EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
    }

  debug1<<"get face nodes\n";


  // need to decide if this mesh is layered or single layer mesh

   int *  m2DPointto3DPoints = new int [m_num_mesh_nodes*m_num_layers];

   for(int i=0;i<m_num_mesh_nodes*m_num_layers;i++)
   {
	   m2DPointto3DPoints[i]= MeshConstants::INVALID_NUM;
   }
   int Index = 0 ;
   for(int iLayer=0;iLayer<m_num_layers;iLayer++)
  {
	  for(int iNode=0;iNode<m_num_mesh_nodes;iNode++)
	  {
		  int bottomLayer = m_kbp00[iNode];
		  if (bottomLayer<=(iLayer+1))
			  {
				  m2DPointto3DPoints[iLayer+iNode*m_num_layers] = Index;
				  Index++;
			  }

	  }

  }

  debug1<<"built mesh "<<mesh_name<<endl;
  vtkUnstructuredGrid *uGrid = vtkUnstructuredGrid::New();
  if (!strcmp(mesh_name,m_mesh_2d.c_str()))
    {	 
		create2DUnstructuredMesh(uGrid,faceNodesPtr,a_timeState);  
    }
  else if (!strcmp(mesh_name,m_layer_mesh.c_str()))
    {	 
		createLayerMesh(uGrid,faceNodesPtr,m2DPointto3DPoints,a_timeState);  
    }
  else if (!strcmp(mesh_name,m_mesh_3d.c_str()))
    {
	    create3DUnstructuredMesh(uGrid,faceNodesPtr,m2DPointto3DPoints,a_timeState); 
    }
   else if (!strcmp(mesh_name,m_side_center_point_2d_mesh.c_str()))
    {
		create2DPointMesh(uGrid,faceNodesPtr,a_timeState);
    }
   else if (!strcmp(mesh_name,m_side_center_point_3d_mesh.c_str()))
    {
		 
		create3DPointMesh(uGrid,faceNodesPtr,a_timeState);
    }
   else if (!strcmp(mesh_name,m_face_center_point_3d_mesh.c_str()))
    {
		 
		create3DPointFaceMesh(uGrid,faceNodesPtr,a_timeState);
    }
  else
    {
        EXCEPTION1(InvalidVariableException, mesh_name);
    }
  
  (a_avtFile->get_cache())->CacheVTKObject(cacheMeshID.c_str(), 
                        avtVariableCache::DATASET_NAME, 
                        a_timeState, 
                        domainID,
                        material.c_str(), 
                        uGrid); 

  
  time_t endTicks      = clock();
  debug1<<"time used in building mesh :"<<endTicks-startTicks<<endl;
  delete    faceNodesPtr;
  delete    m2DPointto3DPoints;
  debug1<<"finish building mesh\n";
  return    uGrid;
}


 void   avtSCHISMFileFormatImpl::insertPyramid(vtkUnstructuredGrid  *     a_uGrid,
								           int                  *     a_validTopNode,
									       int                  *     a_validBottomNode,
										   int                  *     a_faceNodePtr,
										   int                  *     a_2DPointto3DPoints,
								           const int            &     a_Cell,
	                                       const int            &     a_layerID)
 {
	 vtkIdType verts[5];
	//debug1<<"pyramid by triangle cell at layer "<<a_layerID<<", bottom ";
			
	int p1 = a_validBottomNode[0];
	int p2 = a_validBottomNode[1];
	//debug1<<p1<<" "<<p2<<" ";
	verts[0] = a_2DPointto3DPoints[p1*m_num_layers+a_layerID];
	verts[1] = a_2DPointto3DPoints[p2*m_num_layers+a_layerID];
	verts[2] = a_2DPointto3DPoints[p2*m_num_layers+a_layerID+1];
	verts[3] = a_2DPointto3DPoints[p1*m_num_layers+a_layerID+1];
	int p3=MeshConstants::INVALID_NUM;
	//for(int i=0;i<3;i++)
	//{
	//	if((a_validTopNode[i]!=p1)&&(a_validTopNode[i]!=p2))
	//	{
	//		p3= a_validTopNode[i];
	//	}
	//}
	for(int i=0;i<3;i++)
	{
		if(((a_faceNodePtr[i+1]-1)!=p1)&&((a_faceNodePtr[i+1]-1)!=p2))
		{
			p3 = a_faceNodePtr[i+1]-1;
		}
	}

	//debug1<<"top extra:"<<p3<<"\n";
	if (p3==MeshConstants::INVALID_NUM)
	{
			stringstream msgStream(stringstream::out);
            msgStream <<"fail to get pyramid apex for cell " <<a_Cell<<"on layer "<<a_layerID;
			EXCEPTION1(InvalidVariableException,msgStream.str());
	}
	int valid_bottom = std::max(1,m_kbp00[p3])-1;
	if ((a_layerID+1)<valid_bottom)
	{
		verts[4] = a_2DPointto3DPoints[p3*m_num_layers+valid_bottom];
	}
	else
	{
        verts[4] = a_2DPointto3DPoints[p3*m_num_layers+a_layerID+1];
	}
	a_uGrid->InsertNextCell(VTK_PYRAMID,5, verts);

 }


 void   avtSCHISMFileFormatImpl::insertWedge(vtkUnstructuredGrid  *     a_uGrid,
								         int                  *     a_validTopNode,
									     int                  *     a_validBottomNode,
										 int                  *     a_2DPointto3DPoints,
								         const int            &     a_Cell,
	                                     const int            &     a_layerID)
 {
	 vtkIdType verts[2*3];
    //first add bottom face node
	for(int iNode=0;iNode< 3; ++iNode)
	{
	   int p = a_validBottomNode[iNode];
	   verts[iNode] = a_2DPointto3DPoints[p*m_num_layers+a_layerID];
	} 
	//then add top face node
	for(int iNode=3;iNode< 2*3; ++iNode)
	{
	   int p = a_validTopNode[iNode-3];
	   verts[iNode] = a_2DPointto3DPoints[p*m_num_layers+a_layerID+1];
	} 
	a_uGrid->InsertNextCell(VTK_WEDGE,2*3, verts);

 }


  void   avtSCHISMFileFormatImpl::insertTetra(vtkUnstructuredGrid  *     a_uGrid,
								           int                 *     a_validTopNode,
									       int                 *     a_validBottomNode,
										   int                  *     a_faceNodePtr,
										   int                 *     a_2DPointto3DPoints,
								           const int           &     a_Cell,
	                                       const int           &     a_layerID)
 {
	vtkIdType verts[4];
	int p4 = a_validBottomNode[0];
			

	for(int i=0;i<3;i++)
	{
		//int p = a_validTopNode[i];
		int p   = a_faceNodePtr[i+1]-1;
		int valid_bottom = std::max(1,m_kbp00[p])-1;
		if ((a_layerID+1)<valid_bottom)
		{
			verts[i] = a_2DPointto3DPoints[p*m_num_layers+valid_bottom];
		}
		else
		{
            verts[i] = a_2DPointto3DPoints[p*m_num_layers+a_layerID+1];
		}
	}

	verts[3] = a_2DPointto3DPoints[p4*m_num_layers+a_layerID];
	a_uGrid->InsertNextCell(VTK_TETRA,4, verts);
 }


 void   avtSCHISMFileFormatImpl::insertTriangle3DCell(vtkUnstructuredGrid * a_uGrid,
	                                              const int           & a_validTopNodeNum,
									              const int           & a_validBottomNodeNum,
									              int                 * a_validTopNode,
											      int                 * a_validBottomNode,
											      int                 * a_faceNodePtr,
												  int                 * a_2DPointto3DPoints,
											      const int           & a_Cell,
	                                              const int           & a_layerID)
 {
	    if (a_validBottomNodeNum==0) // no 3d cell at all
		{
			return;
		}
	 	

		if (a_validBottomNodeNum ==3) // this is a wedge 
		{
			vtkIdType verts[2*3];
			//first add bottom face node
			for(int iNode=0;iNode< 3; ++iNode)
			{
			  int p = a_validBottomNode[iNode];
			  verts[iNode] = a_2DPointto3DPoints[p*m_num_layers+a_layerID];
			} 
			//then add top face node
			for(int iNode=3;iNode< 2*3; ++iNode)
			{
			  int p = a_validTopNode[iNode-3];
			  verts[iNode] = a_2DPointto3DPoints[p*m_num_layers+a_layerID+1];
			} 
			a_uGrid->InsertNextCell(VTK_WEDGE,2*3, verts);
			m_tri_wedge++;
		}
		else if(a_validBottomNodeNum ==2)// bottom have two node, this is a pyramid
		{
			vtkIdType verts[5];
			
			int p1 = a_validBottomNode[0];
			int p2 = a_validBottomNode[1];
			
			verts[0] = a_2DPointto3DPoints[p1*m_num_layers+a_layerID];
			verts[1] = a_2DPointto3DPoints[p2*m_num_layers+a_layerID];
			verts[2] = a_2DPointto3DPoints[p2*m_num_layers+a_layerID+1];
			verts[3] = a_2DPointto3DPoints[p1*m_num_layers+a_layerID+1];
			int p3=MeshConstants::INVALID_NUM;
			
			for(int i=0;i<3;i++)
			{
				if(((a_faceNodePtr[i+1]-1)!=p1)&&((a_faceNodePtr[i+1]-1)!=p2))
				{
					p3 = a_faceNodePtr[i+1]-1;
				}
			}

			
			if (p3==MeshConstants::INVALID_NUM)
			{
				 stringstream msgStream(stringstream::out);
                 msgStream <<"fail to get pyramid apex for cell " <<a_Cell<<"on layer "<<a_layerID;
			     EXCEPTION1(InvalidVariableException,msgStream.str());
			}
			int valid_bottom = std::max(1,m_kbp00[p3])-1;
			if ((a_layerID+1)<valid_bottom)
			{
				verts[4] = a_2DPointto3DPoints[p3*m_num_layers+valid_bottom];
			}
			else
			{
                verts[4] = a_2DPointto3DPoints[p3*m_num_layers+a_layerID+1];
			}
			a_uGrid->InsertNextCell(VTK_PYRAMID,5, verts);
			m_tri_pyramid++;
		}
		else //tetra
		{
			vtkIdType verts[4];
			int p4 = a_validBottomNode[0];
			

			for(int i=0;i<3;i++)
			{
			
				int p   = a_faceNodePtr[i+1]-1;
				int valid_bottom = std::max(1,m_kbp00[p])-1;
				if ((a_layerID+1)<valid_bottom)
			    {
				   verts[i] = a_2DPointto3DPoints[p*m_num_layers+valid_bottom];
			    }
			    else
			    {
                   verts[i] = a_2DPointto3DPoints[p*m_num_layers+a_layerID+1];
			    }
			}

			verts[3] = a_2DPointto3DPoints[p4*m_num_layers+a_layerID];
			a_uGrid->InsertNextCell(VTK_TETRA,4, verts);
			m_tri_tetra++;
		}

 }

 bool     avtSCHISMFileFormatImpl::fourPointsCoplanar(double p1[3],
	                                              double p2[3],
							                      double p3[3],
							                      double p4[3])
 {
	 double v1[3],v2[3],v3[3];

	 for(int i=0;i<3;i++)
	 {
		 v1[i]=p2[i]-p1[i];
		 v2[i]=p3[i]-p1[i];
		 v3[i]=p4[i]-p1[i];
	 }

	 //double normal[3];
	 //vtkMath::Cross(v1,v2,normal);
	 //double dotvalue = vtkMath::Dot(normal,v3);

	 double dotvalue = v1[0]*(v2[1]*v3[2]-v2[2]*v3[1])-v1[1]*(v2[0]*v3[2]-v2[2]*v3[0])+v1[2]*(v2[0]*v3[1]-v2[1]*v3[0]);
	 
	 if (abs(dotvalue)<1e-6)
	 {
		 return true;
	 }
	 else
	 {
		 return false;
	 }

 }

  void     avtSCHISMFileFormatImpl::insert8NodesPolyhedron(vtkUnstructuredGrid *     a_uGrid,
	                                                   vtkIdType           *     a_verts,
								                       int                 *     a_validTopNode,
									                   int                 *     a_validBottomNode,
													   int                 *     a_2DPointto3DPoints,
								                       const int           &     a_Cell,
	                                                   const int           &     a_layerID,
										               const bool          &     a_bottomCoplane,
										               const bool          &     a_topCoplane)
  {
	  				debug1 <<"8 nodes polyhedron at layer " <<a_layerID<<" cell "
						<<a_Cell<<" "<<a_bottomCoplane<<" "<<a_topCoplane<<"\n";


			  vtkIdType p_t[4];
	
	          for(int inode=0;inode<4;inode++)
		      {
			     int v_t = a_validTopNode[inode];
			
			     // layer in m_kbp00 starts from 1, a_layerID starts from 0
                 int bottomLayer = m_kbp00[v_t];
			   
	            if ((a_layerID+2) >= bottomLayer)
			    {
				    p_t[inode]=a_2DPointto3DPoints[v_t*m_num_layers+a_layerID+1];
		        }
			    else
			    {
                    p_t[inode]=a_2DPointto3DPoints[v_t*m_num_layers+bottomLayer-1];
			    }
		     }
				
				vtkSmartPointer<vtkCellArray> faces = vtkSmartPointer<vtkCellArray>::New();

				for(int iNode =0; iNode<4;++iNode)
				{
					//debug1<<" face "<<iNode<<" ";
					int p1 = a_validTopNode[iNode];
					int iNode2 = iNode+1;
					if(iNode2>3)
					{
						iNode2=0;
					}
					int p2 = a_validTopNode[iNode2];
					
					vtkIdType face[4];
					//face[0]= a_2DPointto3DPoints[p1*m_num_layers+a_layerID+1];
					//face[1]= a_2DPointto3DPoints[p2*m_num_layers+a_layerID+1];
					face[0]=p_t[iNode];
					face[1]=p_t[iNode2];
					face[2]= a_2DPointto3DPoints[p2*m_num_layers+a_layerID];
					face[3]= a_2DPointto3DPoints[p1*m_num_layers+a_layerID];
					faces->InsertNextCell(4,face);
					//debug1<<face[0]<<" "<<face[1]<<" "<<face[2]<<" "<<face[3]<<" \n";	
				}
				int faceNum =4;
			   // add top and bottom face
				 
				if (!a_bottomCoplane)
				{
					vtkIdType faceBottom1[3];
				
					for(int iNode=0;iNode<3;iNode++)
					{
			
						int p1 = a_validBottomNode[iNode];
						faceBottom1[iNode]= a_2DPointto3DPoints[p1*m_num_layers+a_layerID];
						debug1<<" "<<faceBottom1[iNode];
					}
					vtkIdType faceBottom2[3];

					for(int iNode=2;iNode<4;iNode++)
					{
			
						int p1 = a_validBottomNode[iNode];
						faceBottom2[iNode-2]= a_2DPointto3DPoints[p1*m_num_layers+a_layerID];
						debug1<<" "<<faceBottom2[iNode];
					}
					faceBottom2[2]= a_2DPointto3DPoints[a_validBottomNode[0]*m_num_layers+a_layerID];
					faces->InsertNextCell(3,faceBottom1);
				    faces->InsertNextCell(3,faceBottom2);
					faceNum+=2;
				}
				else
				{
                    vtkIdType faceBottom[4];
					for(int iNode=0;iNode<4;iNode++)
					{
			
						int p1 = a_validBottomNode[iNode];
						faceBottom[iNode]= a_2DPointto3DPoints[p1*m_num_layers+a_layerID];
						debug1<<" "<<faceBottom[iNode];
					}
					faces->InsertNextCell(4,faceBottom);
					faceNum++;
				}

				 
				if (!a_topCoplane)
				{
					vtkIdType faceTop1[3];
				
					for(int iNode=0;iNode<3;iNode++)
					{
			
						//int p1 = a_validTopNode[iNode];
						//faceTop1[iNode]= a_2DPointto3DPoints[p1*m_num_layers+a_layerID+1];
						faceTop1[iNode]=p_t[iNode];
						//debug1<<" "<<faceTop1[iNode];
					}
					vtkIdType faceTop2[3];

					for(int iNode=2;iNode<4;iNode++)
					{
			
						//int p1 = a_validTopNode[iNode];
						//faceTop2[iNode-2]= a_2DPointto3DPoints[p1*m_num_layers+a_layerID+1];
						faceTop2[iNode-2]=p_t[iNode];
						//debug1<<" "<<faceTop2[iNode];
					}
					//faceTop2[2]= a_2DPointto3DPoints[a_validTopNode[0]*m_num_layers+a_layerID+1];
					faceTop2[2]=p_t[0];
					faces->InsertNextCell(3,faceTop1);
				    faces->InsertNextCell(3,faceTop2);
					faceNum+=2;
				}
				else
				{
                    vtkIdType faceTop[4];
					for(int iNode=0;iNode<4;iNode++)
					{
			
						//int p1 = a_validTopNode[iNode];
						//faceTop[iNode]= a_2DPointto3DPoints[p1*m_num_layers+a_layerID+1];
						faceTop[iNode]=p_t[iNode];
						//debug1<<" "<<faceTop[iNode];
					}
					faces->InsertNextCell(4,faceTop);
					faceNum++;
				}

			    
				a_uGrid->InsertNextCell(VTK_POLYHEDRON, 8, a_verts,faceNum, faces->GetPointer());

  }

   void    avtSCHISMFileFormatImpl::insert7NodesPolyhedron(vtkUnstructuredGrid *     a_uGrid,
	                                                   vtkIdType           *     a_verts,
								                       int                 *     a_validTopNode,
									                   int                 *     a_validBottomNode,
													   int                 *     a_2DPointto3DPoints,
								                       const int           &     a_Cell,
	                                                   const int           &     a_layerID,
										               const bool          &     a_topCoplane)
   {
	   		
              
            // find out degenerated point
			int degeneratedNode = -9999;
			int degeneratedNodeLoc = -9999;
			for(int iNode=0;iNode<4;++iNode)
			{
				int p1 = a_validTopNode[iNode];
				bool found_in_bottom = false;
				for(int j=0;j<3;++j)
				{
					int p2 = a_validBottomNode[j];
					if(p1==p2) 
					{   
						found_in_bottom = true;
						break;
					}
				}

				if (!found_in_bottom)
				{
					degeneratedNode = p1;
					degeneratedNodeLoc=iNode;
					break;
				}
			 }

			vtkIdType p_t[4];
	
	       for(int inode=0;inode<4;inode++)
		   {
			   int v_t = a_validTopNode[inode];
			
			   // layer in m_kbp00 starts from 1, a_layerID starts from 0
               int bottomLayer = m_kbp00[v_t];
			   
	          if ((a_layerID+2) >= bottomLayer)
			  {
				  p_t[inode]=a_2DPointto3DPoints[v_t*m_num_layers+a_layerID+1];
		      }
			  else
			  {
                  p_t[inode]=a_2DPointto3DPoints[v_t*m_num_layers+bottomLayer-1];
			  }
		    }


			//debug1<<"degenerated node: "<<degeneratedNode<<" at "<<degeneratedNodeLoc<<" \n";

			// vtkCellArray* faces = vtkCellArray::New();
			 vtkSmartPointer<vtkCellArray> faces = vtkSmartPointer<vtkCellArray>::New();

			for(int iNode =0; iNode<4;++iNode)
			{
				//debug1<<" side face "<<iNode<<" ";
                int p1 = a_validTopNode[iNode];
				int iNode2 = iNode+1;
				if(iNode2>3)
				{
					iNode2=0;
				}
				int p2 = a_validTopNode[iNode2];
				int numFaceNode = 4;
				if((p1==degeneratedNode)||(p2==degeneratedNode))
				{
					numFaceNode = 3;
				}

				if (numFaceNode ==3)
				{
					 vtkIdType face[3];
					 //face[0]= a_2DPointto3DPoints[p1*m_num_layers+a_layerID+1];
					 //face[1]= a_2DPointto3DPoints[p2*m_num_layers+a_layerID+1];
					 face[0] = p_t[iNode];
					 face[1] = p_t[iNode2];
					 if(p1==degeneratedNode)
					 {
						face[2]= a_2DPointto3DPoints[p2*m_num_layers+a_layerID];
					 }
					 else
					 {
						face[2]= a_2DPointto3DPoints[p1*m_num_layers+a_layerID];
					 }

					 faces->InsertNextCell(3,face);
					 //debug1<<face[0]<<" "<<face[1]<<" "<<face[2]<<" \n";
				}
				else
				{
					 vtkIdType face[4];
					 //face[0]= a_2DPointto3DPoints[p1*m_num_layers+a_layerID+1];
					 face[0]=p_t[iNode];
					 face[1]= a_2DPointto3DPoints[p1*m_num_layers+a_layerID];
					 face[2]= a_2DPointto3DPoints[p2*m_num_layers+a_layerID];
					 //face[3]= a_2DPointto3DPoints[p2*m_num_layers+a_layerID+1];
					 face[3]=p_t[iNode2];
					 faces->InsertNextCell(4,face);
					 //debug1<<face[0]<<" "<<face[1]<<" "<<face[2]<<" "<<face[3]<<" \n";
				}
			}

			int faceNum = 4;

           // add top and bottom face

			if (!a_topCoplane)
			{
				vtkIdType faceTop1[3];
				//debug1<<"top face1:";
				for(int iNode=0;iNode<3;iNode++)
				{
			
					//int p1 = a_validTopNode[iNode];
					//faceTop1[iNode]= a_2DPointto3DPoints[p1*m_num_layers+a_layerID+1];
					faceTop1[iNode]=p_t[iNode];
					//debug1<<" "<<faceTop1[iNode];
				}
				vtkIdType faceTop2[3];
				//debug1<<"top face2:";
				for(int iNode=2;iNode<4;iNode++)
				{
			
					//int p1 = a_validTopNode[iNode];
					//faceTop2[iNode-2]= a_2DPointto3DPoints[p1*m_num_layers+a_layerID+1];
					faceTop2[iNode-2]=p_t[iNode];
					//debug1<<" "<<faceTop2[iNode];
				}
				//faceTop2[2]=  a_2DPointto3DPoints[a_validTopNode[0]*m_num_layers+a_layerID+1];
				faceTop2[2] = p_t[0];
				faces->InsertNextCell(3,faceTop1);
				faces->InsertNextCell(3,faceTop2);
				faceNum+=2;
			}
			else
			{
                vtkIdType faceTop[4];
				for(int iNode=0;iNode<4;iNode++)
				{
			
					//int p1 = a_validTopNode[iNode];
					//faceTop[iNode]=  a_2DPointto3DPoints[p1*m_num_layers+a_layerID+1];
					faceTop[iNode]=p_t[iNode];
					//debug1<<" "<<faceTop[iNode];
				}
				faces->InsertNextCell(4,faceTop);
				faceNum++;
			}

			//debug1<<" bottom face:";
			vtkIdType faceBottom[3];
			for(int iNode=0;iNode<3;iNode++)
			{
			
				int p1 = a_validBottomNode[iNode];
                faceBottom[iNode]=  a_2DPointto3DPoints[p1*m_num_layers+a_layerID];
				//debug1<<" "<<faceBottom[iNode];
			}
			faces->InsertNextCell(3,faceBottom);
			
			faceNum++;

			// add cut face
			//debug1<<"\n cut face ";
			vtkIdType faceCut[3];
			//faceCut[0]=  a_2DPointto3DPoints[degeneratedNode*m_num_layers+a_layerID+1];
			faceCut[0]=p_t[degeneratedNodeLoc];
			int backNodeLoc = degeneratedNodeLoc -1 ;
			if (backNodeLoc<0)
			{
				backNodeLoc = 3;
			}
			int forNodeLoc = degeneratedNodeLoc +1 ;
			if (forNodeLoc>3)
			{
				forNodeLoc = 0;
			}
			faceCut[1]=  a_2DPointto3DPoints[a_validTopNode[backNodeLoc]*m_num_layers+a_layerID];
			faceCut[2]=  a_2DPointto3DPoints[a_validTopNode[forNodeLoc]*m_num_layers+a_layerID];
			//debug1<<faceCut[0]<<" "<<faceCut[1]<<" "<<faceCut[2]<<"\n";
			faces->InsertNextCell(3,faceCut);
			faceNum++;
		    //debug1<<"total face num:"<<faceNum<<"\n";
			a_uGrid->InsertNextCell(VTK_POLYHEDRON, 7, a_verts,faceNum, faces->GetPointer());
   }

  void    avtSCHISMFileFormatImpl::insertQuad3DCell(vtkUnstructuredGrid *  a_uGrid,
	                                            const int           & a_validTopNodeNum,
									            const int           & a_validBottomNodeNum,
									            int                 * a_validTopNode,
											    int                 * a_validBottomNode,
											    int                 * a_faceNodePtr,
												int                 * a_2DPointto3DPoints,
												const int           & a_Cell,
	                                            const int           & a_layerID)

  {
	    
	    
		if (a_validBottomNodeNum==0)
		{
			return;
		}


		vtkIdType p_t[4];
		//debug1<<"valid 2d top ";
		for(int inode=0;inode<4;inode++)
		{
			int v_t = a_validTopNode[inode];
			
			// layer in m_kbp00 starts from 1, a_layerID starts from 0
            int bottomLayer = m_kbp00[v_t];
			//debug1<<v_t<<" bottom "<<bottomLayer<<" ";
	        if ((a_layerID+2) >= bottomLayer)
			{
				p_t[inode]=a_2DPointto3DPoints[v_t*m_num_layers+a_layerID+1];
			}
			else
			{
                p_t[inode]=a_2DPointto3DPoints[v_t*m_num_layers+bottomLayer-1];
			}
		}

	
		
        //debug1<< "valid top 3d node "<<p_t[0]<<" "<<p_t[1]<<" "<<p_t[2]<<" "<<p_t[3]<<"\n";

		double coord1[3],coord2[3],coord3[3],coord4[3];

		double* coord_ptr = a_uGrid->GetPoint(p_t[0]);
		coord1[0] = coord_ptr[0];
		coord1[1] = coord_ptr[1];
		coord1[2] = coord_ptr[2];

		coord_ptr = a_uGrid->GetPoint(p_t[1]);
		coord2[0] = coord_ptr[0];
		coord2[1] = coord_ptr[1];
		coord2[2] = coord_ptr[2];

		coord_ptr = a_uGrid->GetPoint(p_t[2]);
		coord3[0] = coord_ptr[0];
		coord3[1] = coord_ptr[1];
		coord3[2] = coord_ptr[2];

		coord_ptr = a_uGrid->GetPoint(p_t[3]);
		coord4[0] = coord_ptr[0];
		coord4[1] = coord_ptr[1];
		coord4[2] = coord_ptr[2];

		bool topCoplane= fourPointsCoplanar(coord1,
			                                coord2,
											coord3,
											coord4);
		
		//debug1<<"top Coplane "<<topCoplane<<" "<<a_validBottomNodeNum<<"\n";

		if ((a_validBottomNodeNum ==4)) 
		{
		    
			// find out if bottom four points are coplaner
			int v1 = a_validBottomNode[0];
			int v2 = a_validBottomNode[1];
			int v3 = a_validBottomNode[2];
			int v4 = a_validBottomNode[3];
			
			vtkIdType p1 = a_2DPointto3DPoints[v1*m_num_layers+a_layerID];
		    vtkIdType p2 = a_2DPointto3DPoints[v2*m_num_layers+a_layerID];
		    vtkIdType p3 = a_2DPointto3DPoints[v3*m_num_layers+a_layerID];
		    vtkIdType p4 = a_2DPointto3DPoints[v4*m_num_layers+a_layerID];
		
		    //debug1<<"valid bottom 3d node "<<p1<<" "<<p2<<" "<<p3<<" "<<p4<<"\n";


			coord_ptr = a_uGrid->GetPoint(p1);
		    coord1[0] = coord_ptr[0];
		    coord1[1] = coord_ptr[1];
		    coord1[2] = coord_ptr[2];

     		coord_ptr = a_uGrid->GetPoint(p2);
	    	coord2[0] = coord_ptr[0];
		    coord2[1] = coord_ptr[1];
		    coord2[2] = coord_ptr[2];

    		coord_ptr = a_uGrid->GetPoint(p3);
	    	coord3[0] = coord_ptr[0];
		    coord3[1] = coord_ptr[1];
		    coord3[2] = coord_ptr[2];

		    coord_ptr = a_uGrid->GetPoint(p4);
		    coord4[0] = coord_ptr[0];
		    coord4[1] = coord_ptr[1];
		    coord4[2] = coord_ptr[2];

			bool bottomCoplane = fourPointsCoplanar(coord1,
			                                        coord2,
											        coord3,
											        coord4);
			
			vtkIdType verts[2*4];
			//debug1<<"2D node for this hexahedron ";
			//first add bottom face node
			for(int iNode=0;iNode< 4; ++iNode)
			{
				int p = a_validBottomNode[iNode];
				// debug1<<" "<<p;
				verts[iNode] = a_2DPointto3DPoints[p*m_num_layers+a_layerID];
			} 
			//then add top face node
			for(int iNode=4;iNode< 2*4; ++iNode)
			{
				//int p = a_validTopNode[iNode-4];
				// debug1<<" "<<p;
				//verts[iNode] = a_2DPointto3DPoints[p*m_num_layers+a_layerID+1];
				verts[iNode] = p_t[iNode-4];
			} 

			// this is a HEXAHEDRON /
				
			a_uGrid->InsertNextCell(VTK_HEXAHEDRON,2*4, verts);
			//debug1<<"hexahedron at level (top and bottom coplane) " <<a_layerID<<" cell "<<a_Cell<<endl;
			m_quad_hexhedron++;
		}
		else if (a_validBottomNodeNum ==3) // still do it using hexahedron by using top point of degenerated point twice 
		{
		    
			vtkIdType verts[8];

			//first add bottom face node
			for(int iNode=0;iNode< 3; ++iNode)
			{
			  int p = a_validBottomNode[iNode];
			 // debug1<<" "<<p;
			  verts[iNode] = a_2DPointto3DPoints[p*m_num_layers+a_layerID];
			} 
			verts[3]=p_t[3];
			//then add top face node
			for(int iNode=4;iNode< 8; ++iNode)
			{
			  int p = a_validTopNode[iNode-4];
			 // debug1<<" "<<p;
			 // verts[iNode] = a_2DPointto3DPoints[p*m_num_layers+a_layerID+1];
			  verts[iNode] = p_t[iNode-4];
			} 
			a_uGrid->InsertNextCell(VTK_HEXAHEDRON,2*4, verts);
		    m_quad_hexhedron++;
		}
		else if (a_validBottomNodeNum ==2) // bottom have two node, this is a WEDGE
		{
			//debug1 <<"wedge at layer " <<a_layerID<<" cell "
			//		<<a_Cell<<" "<<topCoplane<<"\n";

			vtkIdType verts[6];
			int cellNodes[4];

			int p1 = a_validBottomNode[0];
			int p2 = a_validBottomNode[1];
			int p1_loc =0;
			int p2_loc =0;
			for( int i=0;i<4;i++)
			{
				cellNodes[i]= (a_faceNodePtr[i+1]-1);
				if(cellNodes[i]==p1)
				{
					p1_loc = i;
				}
				if(cellNodes[i]==p2)
				{
					p2_loc = i;
				}
			}

			
           
			// add first side of wedge
			verts[0] = a_2DPointto3DPoints[p1*m_num_layers+a_layerID];
			verts[1] = a_2DPointto3DPoints[p1*m_num_layers+a_layerID+1];
			

			int p1NeiborFront = p1_loc+1;
			if (p1NeiborFront>=4)
			{
				p1NeiborFront=p1NeiborFront-4;
			}
			int p1NeiborBack = p1_loc-1;
			if (p1NeiborBack<0)
			{
				p1NeiborBack=p1NeiborBack+4;
			}
			int p3 = cellNodes[p1NeiborBack];
			if (p3==p2) 
			{
				p3 = cellNodes[p1NeiborFront];
			}

			int valid_bottom = std::max(1,m_kbp00[p3])-1;
			if ((a_layerID+1)<valid_bottom)
			  {
			     verts[2] = a_2DPointto3DPoints[p3*m_num_layers+valid_bottom];
			  }
			else
			  {
				 verts[2] = a_2DPointto3DPoints[p3*m_num_layers+a_layerID+1];
			  }

			verts[3] = a_2DPointto3DPoints[p2*m_num_layers+a_layerID];
			verts[4] = a_2DPointto3DPoints[p2*m_num_layers+a_layerID+1];

			int p2NeiborFront = p2_loc+1;
			if (p2NeiborFront>=4)
			{
				p2NeiborFront=p2NeiborFront-4;
			}
			int p2NeiborBack = p2_loc-1;
			if (p2NeiborBack<0)
			{
				p2NeiborBack=p2NeiborBack+4;
			}
			
			p3 = cellNodes[p2NeiborBack];
			if (p3==p1) 
			{
				p3 = cellNodes[p2NeiborFront];
			}

			valid_bottom = std::max(1,m_kbp00[p3])-1;
			if ((a_layerID+1)<valid_bottom)
			  {
			     verts[5] = a_2DPointto3DPoints[p3*m_num_layers+valid_bottom];
			  }
			else
			  {
				 verts[5] = a_2DPointto3DPoints[p3*m_num_layers+a_layerID+1];
			  }
			

			a_uGrid->InsertNextCell(VTK_WEDGE,6, verts);
			m_quad_wedge++;
		}
		else // this is pyramid 5 nodes
		{   
			debug1 <<"pyramid at layer " <<a_layerID<<" cell "
					<<a_Cell<<" "<<topCoplane<<"\n";;
			vtkIdType verts[5];
		   
			for (int iNode=0;iNode<4;iNode++)
			{
			   verts[iNode]= p_t[iNode];
			}
			int p3 = a_validBottomNode[0];
			verts[4] = a_2DPointto3DPoints[p3*m_num_layers+a_layerID];
			//debug1<<" bottom: "<<p3<<"\n";
			a_uGrid->InsertNextCell(VTK_PYRAMID,5, verts);
			m_quad_pyramid++;
		}

  }

void    avtSCHISMFileFormatImpl::validTopBottomNode(int       &   a_validTopNodeNum,
	                                            int       &   a_validBottomNodeNum,
											    int       *   a_validTopNode,
											    int       *   a_validBottomNode,
												const int &   a_layerID,
									            int*          a_faceNodePtr) const

{

	 int numberOfNodeInCell = *a_faceNodePtr;

			 
	//get node id indexed in 2d mesh
	int * node2D = new int [numberOfNodeInCell];
	
	for(int iNode=0;iNode< numberOfNodeInCell; iNode++)
    {   
		// faceNodePtr stores node index starting from 1
		// need decrease by one to be conisistent with
		// visit custom
        node2D[iNode] = a_faceNodePtr[iNode+1]-1;
	
    } 
	

	//find out valid number node on bottom face
	int validBottomNode = 0;
	for(int iNode=0;iNode<numberOfNodeInCell; iNode++)
	{
		// layer in m_kbp00 starts from 1, a_layerID starts from 0
		int bottomLayer = m_kbp00[node2D[iNode]];
		if ((a_layerID+1) >= bottomLayer)
		{
			a_validBottomNode[validBottomNode]=node2D[iNode];
			validBottomNode++;
		}
	}

	//find out valid number node on top face
	//int validTopNode = numberOfNodeInCell;
	//for(int iNode=0;iNode<numberOfNodeInCell; ++iNode)
	//{
	//   a_validTopNode[iNode]=node2D[iNode];
		
	//}
	//find out valid number node on top face
	int validTopNode = 0;
	for(int iNode=0;iNode<numberOfNodeInCell; iNode++)
	{
	   // layer in m_kbp00 starts from 1, a_layerID starts from 0
       //int bottomLayer = m_kbp00[node2D[iNode]];
	   //if ((a_layerID+2) >= bottomLayer)
	   	{
	       a_validTopNode[validTopNode]=node2D[iNode];
	       validTopNode++;
		}
		
	}
	a_validTopNodeNum    = validTopNode;
	a_validBottomNodeNum = validBottomNode;
	delete node2D;

}
// ****************************************************************************
//  Method: avtSCHISMFileFormatImpl:: updateMeshZCoordinates
//
//  Purpose:
//      update mesh points z coordinates for surface changes with time
//  
//  Arguments:
//     a_pointSet  the points making up mesh
//     a_timeState time
//     a_meshName  mesh label
//
//  Programmer: qshu
//  Creation:   Mon Sep 24 2012
//
// ****************************************************************************
void  avtSCHISMFileFormatImpl::updateMeshZCoordinates(vtkPoints * a_pointSet,
                                                  const int   a_timeState,
                                                  const char* a_meshName)
{


  if (!strcmp(a_meshName,m_mesh_2d.c_str()))
    {     
      float * pointPtr       = (float *) a_pointSet->GetVoidPointer(0);
      float * zCache = new float [m_num_mesh_nodes];
	  if (!m_external_mesh_provider->zcoords2D(zCache,a_timeState))
	  {
		  stringstream msgStream(stringstream::out);
          msgStream <<"Fail to retrieve node z at step " <<a_timeState;
          EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
	  }
      for(int iNode=0;iNode < m_num_mesh_nodes; iNode++)
        {         
          float z            = zCache[iNode];
          pointPtr++;
          pointPtr++;
          *pointPtr++         = z;
          //debug1<<1<<" "<<iNode<<" "<<x<<" "<<y<<" "<<z<<"\n";
        }
	  delete zCache;
    }
  else if (!strcmp(a_meshName,m_mesh_3d.c_str()))
    {
      float * pointPtr       = (float *) a_pointSet->GetVoidPointer(0);
	  float * nodeZPtr = new float[m_total_valid_3D_point];
	  loadAndCacheZ(a_timeState,nodeZPtr);
      for(int iNode=0;iNode < m_total_valid_3D_point; iNode++)
      {    
            float z            =   nodeZPtr[iNode];
            pointPtr++;
            pointPtr++;
            *pointPtr++         = z;
	  }
	  delete nodeZPtr;
    }
  else if (!strcmp(a_meshName,m_side_center_point_2d_mesh.c_str()))
    {     
      float * pointPtr       = (float *) a_pointSet->GetVoidPointer(0);
      float * zCache = new float [m_num_mesh_edges];
	  if (!m_external_mesh_provider->zSideCenter2D(zCache,a_timeState))
	  {
		  stringstream msgStream(stringstream::out);
          msgStream <<"Fail to retrieve node z at step " <<a_timeState;
          EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
	  }
      for(int i=0;i < m_num_mesh_edges; i++)
        {         
          float z            = zCache[i];
          pointPtr++;
          pointPtr++;
          *pointPtr++         = z;
          //debug1<<1<<" "<<iNode<<" "<<x<<" "<<y<<" "<<z<<"\n";
        }
	  delete zCache;
    }
  else if (!strcmp(a_meshName,m_side_center_point_3d_mesh.c_str()))
    {
      float * pointPtr       = (float *) a_pointSet->GetVoidPointer(0);
	  float * sideCenterZPtr = new float[m_total_valid_3D_side];
	  loadAndCacheZSide(a_timeState,sideCenterZPtr);
      for(int i=0;i < m_total_valid_3D_side; i++)
      {    
            float z            =   sideCenterZPtr[i];
            pointPtr++;
            pointPtr++;
            *pointPtr++         = z;
	  }
	  delete sideCenterZPtr;
    }



}

void avtSCHISMFileFormatImpl::loadAndCacheZSide(const int& a_timeState, float * a_sideCenterZPtr)
{
	 if (!m_external_mesh_provider->zSideCenter3D(a_sideCenterZPtr,a_timeState))
	 {
		  stringstream msgStream(stringstream::out);
          msgStream <<"Fail to retrieve side z at step " <<a_timeState;
          EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
	 }
}

void avtSCHISMFileFormatImpl::loadAndCacheZEle(const int& a_timeState,float * a_eleCenterZPtr)
{
	 if (!m_external_mesh_provider->zEleCenter3D(a_eleCenterZPtr,a_timeState))
	 {
		  stringstream msgStream(stringstream::out);
          msgStream <<"Fail to retrieve ele z at step " <<a_timeState;
          EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
	 }
	
}

void avtSCHISMFileFormatImpl::loadAndCacheZ(const int& a_timeState,float* a_nodeZPtr)
{
	 if (!m_external_mesh_provider->zcoords3D(a_nodeZPtr,a_timeState))
	 {
		  stringstream msgStream(stringstream::out);
          msgStream <<"Fail to retrieve node z at step " <<a_timeState;
         EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
	 }
}


void   avtSCHISMFileFormatImpl::getMeshDimensions(MeshProvider * a_meshProviderPtr)
{
  
  debug1<<"getting mesh node number\n";

  m_num_mesh_nodes = a_meshProviderPtr->numberOfNode();

  debug1<<"number of node "<<m_num_mesh_nodes<<"\n";

  if (m_num_mesh_nodes<0)  //to do: face centered data will get num faces
    {
      EXCEPTION1(InvalidVariableException, ("no nodes found in "+a_meshProviderPtr->file()).c_str());
    }
  
  // SCHISMDim constructor and destructor are private, no need to del.

   m_num_mesh_faces = a_meshProviderPtr->numberOfElement();
  
  if (m_num_mesh_faces<0)
    {
      EXCEPTION1(InvalidVariableException, ("no element found in "+a_meshProviderPtr->file()).c_str());
    }
  
  debug1<<"get face num"<<m_num_mesh_faces<<"\n";

   m_num_mesh_edges = a_meshProviderPtr->numberOfSide();
  
  if (m_num_mesh_edges<0)
    {
      EXCEPTION1(InvalidVariableException, ("no edges found in "+a_meshProviderPtr->file()).c_str());
    }
  
  //initlaize nominal data size per layer for all centering
  m_nominal_size_per_layer[MeshConstants::NODE]= m_num_mesh_nodes;
  m_nominal_size_per_layer[MeshConstants::ELEM]= m_num_mesh_faces;
  m_nominal_size_per_layer[MeshConstants::EDGE]= m_num_mesh_edges;

 m_num_layers= a_meshProviderPtr->numberOfLayer();
  
  if (m_num_layers<0)
    {
      EXCEPTION1(InvalidVariableException, ("no layer found ,one required at least, in "+a_meshProviderPtr->file()).c_str());
    }
   debug1<<"get layers num"<<m_num_layers<<"\n";
}



void    avtSCHISMFileFormatImpl::loadMeshCoordinates(MeshProvider * a_meshProviderPtr)
{
	 // read in x,y coordinates for nodes 

  if (!m_node_x_ptr)
    {
      m_node_x_ptr      = new float [m_num_mesh_nodes];
    }
  if (!m_node_y_ptr)
    {
      m_node_y_ptr      = new float [m_num_mesh_nodes];
    }

  bool state=false;

  float * coordCachePtr = new float [m_num_mesh_nodes*3];
  int timeStep=0;
  state=a_meshProviderPtr->fillPointCoord2D(coordCachePtr,timeStep);
  if(!state)
  {
	   stringstream msgStream(stringstream::out);
       msgStream <<"load mesh x,y from provider \n";
	   EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
  }

  float * coordPoints = coordCachePtr;

  for(int iNode=0;iNode < m_num_mesh_nodes; iNode++)
    {
             
        m_node_x_ptr[iNode] = *coordPoints;
		coordPoints++;
        m_node_y_ptr[iNode] = *coordPoints;
		coordPoints++;
		coordPoints++;
     }
  

  delete coordCachePtr;

  debug1<<"get node x y\n";

  //if (!m_node_z_ptr)
 // {
//	  m_node_z_ptr  = new float [m_num_mesh_nodes*m_num_layers];
 // }

  m_kbp00 = new int [m_num_mesh_nodes];
  a_meshProviderPtr->fillKbp00(m_kbp00);
  
  //count total valid 3d point
  m_total_valid_3D_point = 0;
  for(int iLayer=0;iLayer<m_num_layers;iLayer++)
  {
	  for(int iNode=0;iNode<m_num_mesh_nodes;iNode++)
	  {
		  int bottomLayer = m_kbp00[iNode];
		  if (bottomLayer<=(iLayer+1))
			  {
				  m_total_valid_3D_point++;
			  }

	  }

  }

  int * kbs = new int [m_num_mesh_edges];
  a_meshProviderPtr->fillKbs(kbs);
  
  //count total valid 3d point
  m_total_valid_3D_side = 0;
  for(int iLayer=0;iLayer<m_num_layers;iLayer++)
  {
	  for(int iEdge=0;iEdge<m_num_mesh_edges;iEdge++)
	  {
		  int bottomLayer = kbs[iEdge];
		  if (bottomLayer<=(iLayer+1))
			  {
				  m_total_valid_3D_side++;
			  }

	  }

  }
 

 delete kbs;


  int * kbe = new int [m_num_mesh_faces];
  a_meshProviderPtr->fillKbe(kbe);
  
  //count total valid 3d ele
  m_total_valid_3D_ele = 0;
  for(int iLayer=0;iLayer<m_num_layers;iLayer++)
  {
	  for(int iEle=0;iEle<m_num_mesh_faces;iEle++)
	  {
		  int bottomLayer = kbe[iEle];
		  if (bottomLayer<=(iLayer+1))
			  {
				  m_total_valid_3D_ele++;
			  }
	  }

  }
 

 debug1<<"done side mapper\n";
 delete kbe;
}
// ****************************************************************************
//  Method: avtSCHISMFileFormatImpl::getLayer
//
//  Purpose:
//      Return layer id for each 3d node
//      
//     
//
//  Arguments:
//       
//                  
//      None
//
//  Programmer: qshu -- generated by xml2avt
//  Creation:   Mon DEC 13 09:57:49 PDT 2015
//
// ****************************************************************************

 vtkDataArray*   avtSCHISMFileFormatImpl::getLayer()
 {
  int * kbp00;

  if (m_kbp_data)
  {
	  kbp00=m_kbp_data;
  }
  else
  {
	   stringstream msgStream(stringstream::out);
       msgStream <<"Fail to retrieve data bottom layer index\n";
       EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
  }

  int valid_var_size =0;
  int num_total_layers    =m_num_layers; 
  if(m_level_center == MeshConstants::HALF_LAYER)
  {
	  num_total_layers--;

  }
  int nominal_num_data_per_Layer=m_nominal_size_per_layer[m_data_center];

  for(int iNode=0;iNode<nominal_num_data_per_Layer;iNode++)
  {
	   
	  valid_var_size+=num_total_layers-std::max(1,kbp00[iNode])+1;
  }

  int ntuples        = valid_var_size; 
  vtkIntArray *rv = vtkIntArray::New();
  rv->SetNumberOfTuples(ntuples);
  int idata = 0;    


   for (int iLayer   = 0 ; iLayer < num_total_layers ; iLayer++)
   {
     for( int iNode = 0 ; iNode   < nominal_num_data_per_Layer; iNode++)
      {  
	    int valid_bottom_layer = std::max(1,kbp00[iNode])-1;
		if (iLayer>=valid_bottom_layer)
	    {
           rv->SetTuple1(idata, iLayer);  
           idata++;    
	     }
      }
    }

   return rv;
 
 }

// ****************************************************************************
//  Method: avtSCHISMFileFormatImpl::GetVar
//
//  Purpose:
//      Gets a scalar variable associated with this file.  Although VTK has
//      support for many different types, the best bet is vtkFloatArray, since
//      that is supported everywhere through VisIt.
//
//  Arguments:
//      timestate  The index of the timestate.  If GetNTimesteps returned
//                 'N' time steps, this is guaranteed to be between 0 and N-1.
//      varname    The name of the variable requested.
//
//  Programmer: qshu -- generated by xml2avt
//  Creation:   Wed Mar 13 09:13:49 PDT 2013
//
// ****************************************************************************

vtkDataArray *
avtSCHISMFileFormatImpl::GetVar(int a_timeState, const char *a_varName)
{

   if ( !strcmp(a_varName,( MeshConstants::LEVEL).c_str()))
   {
	   return getLayer();
   }

  //a_varName is the label used by visit. This name
  //needs to be map into varname used in data file.
  
  std::string SCHISMVarName = m_var_name_label_map[a_varName];  
  std::string varMesh      = m_var_mesh_map[a_varName];

  SCHISMVar * SCHISMVarPtr = m_data_file_ptr->get_var(SCHISMVarName);
  
  if (!(SCHISMVarPtr->is_valid()))
    {
      
      EXCEPTION1(InvalidVariableException, a_varName);
    }
  
    if ( (!strcmp(a_varName,m_node_surface_label.c_str())) ||
         (!strcmp(a_varName,m_node_depth_label.c_str()))   ||
		 (!SCHISMVarIs3D(SCHISMVarPtr)))
    {
     float * valBuff;
	 int numData=m_nominal_size_per_layer[m_data_center];
	
     valBuff          = new float  [numData]; 

     getSingleLayerVar (valBuff,
		                m_data_file_ptr,
                        a_timeState,
                        SCHISMVarName);
	

     //total number of data = nodes for a time step
     int ntuples        = numData; 
     vtkDoubleArray *rv = vtkDoubleArray::New();
     rv->SetNumberOfTuples(ntuples);
     int idata = 0;    
     for( int iNode = 0 ; iNode < numData; iNode++)
       {
         float valTemp = valBuff[iNode];
         rv->SetTuple1(idata, valTemp);  
         idata++;             
       }
     delete   valBuff;
     return rv;
    }


 
  debug1<<"get SCHISM var "<<SCHISMVarName<<"\n";

  int num_data_layers    =m_num_layers;
  if(m_level_center == MeshConstants::HALF_LAYER)
  {
	  num_data_layers=m_num_layers-1;

  }
  std::string        varName(a_varName);
  int * layerStarts;
  if (varName.find(m_surface_state_suffix) != string::npos) 
    {
      num_data_layers      = 1;
      layerStarts    = new int [num_data_layers];
      layerStarts[0] =m_num_layers-1;
	  if(m_level_center == MeshConstants::HALF_LAYER)
      {
	     layerStarts[0]--;
      }
    }
  else if (varName.find(m_bottom_state_suffix) != string::npos)
    {
      num_data_layers      = 1;
      layerStarts    = new int [num_data_layers];
	  //this is the layer next to the bottom
      layerStarts[0] = 1;
    } 
  else
    {
      layerStarts   = new int [num_data_layers];
      for(int iLayers=0;iLayers<num_data_layers;iLayers++)
        {
          layerStarts[iLayers] = iLayers;
        }
     }
  
 

  int numDataPerLayer=m_nominal_size_per_layer[m_data_center];
 
  float * valBuff;
  int numTotalLayers    =m_num_layers; // there is no subtract for prismcenter data for
                                       // prism center also maintain total number of layer as other centering
 

  valBuff = new float  [numTotalLayers*numDataPerLayer];

   
  int numOfRecord  = 1;
  int nodeStart    = 0;
  int timeStart    = a_timeState;
   
  SCHISMVarPtr->set_cur(timeStart);
  
  
  if (!(SCHISMVarPtr->get(valBuff)))
  {
       stringstream msgStream(stringstream::out);
       msgStream <<"Fail to retrieve "<<a_varName << " at step " <<a_timeState;
       EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
  }
     
  int valid_var_size =0;
  int * node_start_index  = new int [numDataPerLayer];
  int * num_data_at_layer = new int [numTotalLayers];

  for(int iLayer=0;iLayer<numTotalLayers;iLayer++)
  {
	  num_data_at_layer[iLayer]=0;
  }

  int * kbp00;

  if (m_kbp_data)
  {
	  kbp00=m_kbp_data;
  }
  else
  {
	  kbp00=m_kbp00;
  }

   int num_layers =m_num_layers;
   if(m_level_center == MeshConstants::HALF_LAYER)
   {
	    num_layers--;
   }
 

  for(int iNode=0;iNode<numDataPerLayer;iNode++)
  {
	  node_start_index[iNode]=valid_var_size;
	  valid_var_size+=num_layers-std::max(1,kbp00[iNode])+1;
	  
	  for(int iLayer=0;iLayer<numTotalLayers;iLayer++)
      {
		if(iLayer>=(std::max(1,kbp00[iNode])-1)) 
		{
	       num_data_at_layer[iLayer]++;
		}
      }
  }
 
  
   vtkFloatArray *rv = vtkFloatArray::New(); 
   int idata = 0;
   if (varName.find(m_depth_average_suffix) == string::npos)
     {
       idata = 0;
      
	   // count total number of data 
	   int ntuples       = 0;
       for (int iLayer   = 0 ; iLayer < num_data_layers ; iLayer++)
       {
          int layer = layerStarts[iLayer];
		  ntuples += num_data_at_layer[layer];
		  debug1<<iLayer<<" "<<num_data_at_layer[layer]<<"\n";
	   }
      
	   debug1<<" count total num of data "<<ntuples<<"\n ";
       rv->SetNumberOfTuples(ntuples); 
	   if (varName.find(m_bottom_state_suffix) != string::npos) //for bottom data, same as number of 2D node
	   {
	      rv->SetNumberOfTuples(numDataPerLayer); 
	   }

       for (int iLayer   = 0 ; iLayer < num_data_layers ; iLayer++)
       {
          int layer = layerStarts[iLayer];
          for( int iNode = 0 ; iNode   < numDataPerLayer; iNode++)
          {  
			  float valTemp =MeshConstants::DUMMY_ELEVATION;
			  int start_index = node_start_index[iNode];
			  int valid_bottom_layer = std::max(1,kbp00[iNode])-1;
		      if (layer<valid_bottom_layer)
			  {
				 if(varName.find(m_bottom_state_suffix) != string::npos)
				 {
					int loc = start_index+1; // the layer just above the first valid layer
					int max_above_allowd = numTotalLayers-std::max(1,kbp00[iNode]);
					if (max_above_allowd ==0) // there is only one layer
					{
						loc = start_index;
					}
				    valTemp = valBuff[loc];
				    rv->SetTuple1(idata, valTemp);  
				    idata++;  
				 }

			  }
			  else
			  {
			   int data_loc = start_index+layer-valid_bottom_layer;
               valTemp =  valBuff[data_loc];
			   rv->SetTuple1(idata, valTemp);  
               idata++;    
			  }
          }
        }
     }  
   else
     {
       idata = 0;
       //total number of data =nodes for a time step
       int ntuples       = numDataPerLayer;
       rv->SetNumberOfTuples(ntuples);

       float * averageState = new float [numDataPerLayer];
	   debug1<<"begin averaging state\n";
       depthAverage(averageState,valBuff,node_start_index,a_timeState);

	   debug1<<"total num of averaged data "<<numDataPerLayer<<"\n";
       idata  = 0;
       for( int iNode = 0 ; iNode   < numDataPerLayer; iNode++)
         {
           float valTemp   = averageState[iNode];
           rv->SetTuple1(idata, valTemp);  
           idata++;             
         }
       delete averageState;
     }

   delete valBuff;
   delete layerStarts;
   delete num_data_at_layer;
   debug1<<"done load data for "<<SCHISMVarName<<"\n";
   return rv;
   
}


// ****************************************************************************
//  Method: avtSCHISMFileFormatImpl::GetVectorVar
//
//  Purpose:
//      Gets a vector variable associated with this file.  Although VTK has
//      support for many different types, the best bet is vtkFloatArray, since
//      that is supported everywhere through VisIt.
//
//  Arguments:
//      timestate  The index of the timestate.  If GetNTimesteps returned
//                 'N' time steps, this is guaranteed to be between 0 and N-1.
//      varname    The name of the variable requested.
//
//  Programmer: qshu -- generated by xml2avt
//  Creation:   Wed Mar 13 09:13:49 PDT 2013
//
// ****************************************************************************

vtkDataArray *
avtSCHISMFileFormatImpl::GetVectorVar(int a_timeState, const char *a_varName)
{
 
    std::string SCHISMVarName = m_var_name_label_map[a_varName];  
    std::string varMesh      = m_var_mesh_map[a_varName];

	SCHISMVar * SCHISMVarPtr = m_data_file_ptr->get_var(SCHISMVarName);
    if (!(SCHISMVarPtr->is_valid()))
    {
      
      EXCEPTION1(InvalidVariableException, a_varName);
    }

    // last dim is vector component  
    int      numDim = SCHISMVarPtr->num_dims();       
    SCHISMDim* comDim = SCHISMVarPtr->get_dim(numDim-1);
    int ncomps       = comDim->size();
	int ucomps       = (ncomps == 2 ? 3 : ncomps);
	float *oneEntry  = new float[ucomps];
	int idata=0;

	if (!SCHISMVarIs3D(SCHISMVarPtr))
    {
      float * valBuff;
	  int numData=m_nominal_size_per_layer[m_data_center];
	  
	  int numDataEntry=numData;

	  numData*=ncomps;

      valBuff          = new float  [numData]; 
      getSingleLayerVar (valBuff,
		                m_data_file_ptr,
                        a_timeState,
                        SCHISMVarName);
      //total number of data = nodes for a time step
      int ntuples        = numDataEntry; 
      vtkFloatArray *rv = vtkFloatArray::New();
	  rv->SetNumberOfComponents(ucomps);
      rv->SetNumberOfTuples(ntuples);
	  idata  = 0;
      for( int iNode = 0 ; iNode   < ntuples; iNode++)
         {
           for(int iComp = 0; iComp < ncomps; iComp++)
            {
               oneEntry[iComp]   = valBuff[iNode*ncomps+iComp];
            }
            for(int iComp = ncomps; iComp < ucomps; iComp++)
            {
               oneEntry[iComp]= 0.0;
             }
            
           rv->SetTuple(idata, oneEntry);  
           idata++;             
         }
      delete   valBuff;
      return rv;
    }



    int num_data_layers    =m_num_layers;
	 if(m_level_center == MeshConstants::HALF_LAYER)
    {
	  num_data_layers=m_num_layers-1;
    }
    std::string        varName(a_varName);
    int * layerStarts;
    if (varName.find(m_surface_state_suffix) != string::npos) 
    {
      num_data_layers      = 1;
      layerStarts    = new int [num_data_layers];
      layerStarts[0] =m_num_layers-1;
	  if(m_level_center == MeshConstants::HALF_LAYER)
      {
	    layerStarts[0]--;
      }
    }
    else if (varName.find(m_bottom_state_suffix) != string::npos)
    {
      num_data_layers      = 1;
      layerStarts    = new int [num_data_layers];
	  //this is the layer next to the bottom
	  //for prism center data this is the first valid bottom prim center
      layerStarts[0] = 1;
    } 
    else
    {
      layerStarts   = new int [num_data_layers];
      for(int iLayers=0;iLayers<num_data_layers;iLayers++)
        {
          layerStarts[iLayers] = iLayers;
        }
     }
  

   
    
	int numDataPerLayer=m_nominal_size_per_layer[m_data_center];
	

    float * valBuff;
	int totalNumLayers=m_num_layers;
	
    valBuff = new float  [totalNumLayers*numDataPerLayer*ncomps];

 
    int numOfRecord  = 1;
    int nodeStart    = 0;
    int timeStart    = a_timeState;
   
    SCHISMVarPtr->set_cur(timeStart);

    if (!(SCHISMVarPtr->get(valBuff)))
    {
       stringstream msgStream(stringstream::out);
       msgStream <<"Fail to retrieve "<<a_varName << " at step " <<a_timeState;
       EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
     }
  
	int valid_var_size =0;
    int * node_start_index= new int [numDataPerLayer];
    int * num_data_at_layer = new int [totalNumLayers];

    for(int iLayer=0;iLayer<m_num_layers;iLayer++)
    {
	  num_data_at_layer[iLayer]=0;
    }
	int * kbp00;

    if (m_kbp_data)
    {
	    kbp00=m_kbp_data;
    }
    else
    {
	    kbp00=m_kbp00;
    }
   int num_layers =m_num_layers;
   if(m_level_center == MeshConstants::HALF_LAYER)
   {
	    num_layers--;
   }
    for(int iNode=0;iNode<numDataPerLayer;iNode++)
    {
	  node_start_index[iNode]=valid_var_size;
	  valid_var_size+=((num_layers-std::max(1,kbp00[iNode])+1)*ncomps);
	  for(int iLayer=0;iLayer<totalNumLayers;iLayer++)
      {
		if(iLayer>=(std::max(1,kbp00[iNode])-1))
		{
	       num_data_at_layer[iLayer]++;
		}
      }
    }
 
	debug1<<"size of vector data "<<valid_var_size<<"\n";


     vtkFloatArray *rv = vtkFloatArray::New(); 
     rv->SetNumberOfComponents(ucomps);
     idata = 0;
     
     if (varName.find(m_depth_average_suffix) == string::npos)
     {
       idata = 0;
       //total number of data =layers*nodes for a time step
       int ntuples       = 0;
       for (int iLayer   = 0 ; iLayer < num_data_layers ; iLayer++)
       {
          int layer = layerStarts[iLayer];
		  ntuples += num_data_at_layer[layer];
	   }

       rv->SetNumberOfTuples(ntuples); 
	   if (varName.find(m_bottom_state_suffix) != string::npos) //for surface and bottom data, same as number of 2D node
	   {
	      rv->SetNumberOfTuples(numDataPerLayer); 
	   }

	   debug1<<"totla number of vect tuples "<<ntuples<<" "<<num_data_layers<<" \n";

       for (int iLayer   = 0 ; iLayer < num_data_layers ; iLayer++)
       {
          int layer = layerStarts[iLayer];
           
          for( int iNode = 0 ; iNode < numDataPerLayer; iNode++)
          {  
              
			  int start_index = node_start_index[iNode];

			  int bottom_layer = std::max(1,kbp00[iNode])-1;

			  if(layer>=bottom_layer)
			  {

				  for(int iComp = 0; iComp < ncomps; iComp++)
				  {
					 oneEntry[iComp]= valBuff[start_index+(layer+1-std::max(1,kbp00[iNode]))*ncomps+iComp];
                 
				  }

				  for(int iComp = ncomps; iComp < ucomps; iComp++)
				  {
					 oneEntry[iComp]= 0.0;
				  }
              
				  rv->SetTuple(idata, oneEntry);  
				  idata++; 
			  }
			  else 
			  {
				 if(varName.find(m_bottom_state_suffix) != string::npos)
				 {
					int dist = 1; // the layer just above the first valid layer
					int max_above_allowd = num_data_layers-std::max(1,kbp00[iNode]);
					if (max_above_allowd ==0) // there is only one layer
					{
						dist = 0;
					}
					for(int iComp = 0; iComp < ncomps; iComp++)
				    {
					    oneEntry[iComp]= valBuff[start_index+dist*ncomps+iComp];
                 
				    }

				    for(int iComp = ncomps; iComp < ucomps; iComp++)
				    {
					    oneEntry[iComp]= 0.0;
				    }
				   
				    rv->SetTuple(idata, oneEntry);  
				    idata++;  
				 }

			  }
          }
        
       }  
	 }
     else
     {
       idata = 0;
       //total number of data =nodes for a time step
       int ntuples       = numDataPerLayer;
       rv->SetNumberOfTuples(ntuples);

       float * averageState = new float [numDataPerLayer*ncomps];

       vectorDepthAverage(averageState,valBuff,node_start_index,a_timeState,ncomps);

       idata  = 0;
       for( int iNode = 0 ; iNode   < numDataPerLayer; iNode++)
         {
           for(int iComp = 0; iComp < ncomps; iComp++)
            {
               oneEntry[iComp]   = averageState[iNode*ncomps+iComp];
            }
            for(int iComp = ncomps; iComp < ucomps; iComp++)
            {
               oneEntry[iComp]= 0.0;
             }
            
           rv->SetTuple(idata, oneEntry);  
           idata++;             
         }
       delete averageState;
     }

 
    delete valBuff;
    delete layerStarts;
    delete oneEntry;
    return rv;   
}

 void   avtSCHISMFileFormatImpl::prepare_average(int * & a_kbp, int * & a_mapper, float * & a_zPtr, const int & a_timeState)
 {

  a_kbp    = m_kbp00; 
  
  if(m_data_center == MeshConstants::EDGE)
	{
		a_kbp    = m_kbp_data;
        a_mapper = new int [m_num_mesh_edges*m_num_layers];
        for(int i=0;i<m_num_mesh_edges*m_num_layers;i++)
        {
	        a_mapper[i]= MeshConstants::INVALID_NUM;
        }
       a_zPtr   = new float[m_total_valid_3D_side]; 
	   loadAndCacheZSide(a_timeState,a_zPtr);

	   debug1<<" mapping edge 1850 to 3d id:\n";
       int Index = 0 ;
        for(int iLayer=0;iLayer<m_num_layers;iLayer++)
        {
	       for(int iEdge=0;iEdge<m_num_mesh_edges;iEdge++)
	      {
		     int bottomLayer = a_kbp[iEdge];
		     if (bottomLayer<=(iLayer+1))
			  {
				  a_mapper[iLayer+iEdge*m_num_layers] = Index;
				  if(iEdge == 1850)
				  {
				  debug1<<iLayer<<"->"<<Index<<" "<<a_zPtr[Index]<<" ";
				  }
				  Index++;
			  }
	      }
        }
		debug1<<"\n";
	}
  else if (m_data_center == MeshConstants::ELEM)
  {
	  a_kbp    = m_kbp_data;
      a_mapper = new int [m_num_mesh_faces*m_num_layers];
      for(int i=0;i<m_num_mesh_faces*m_num_layers;i++)
      {
	     a_mapper[i]= MeshConstants::INVALID_NUM;
      }
   
      int Index = 0 ;
      for(int iLayer=0;iLayer<m_num_layers;iLayer++)
      {
	     for(int iEle=0;iEle<m_num_mesh_faces;iEle++)
	     {
		    int bottomLayer = std::max(1,a_kbp[iEle]);
		    if (bottomLayer<=(iLayer+1))
			  { 
				  a_mapper[iLayer+iEle*m_num_layers] = Index;
				  Index++;
			  }
	      }
      }
	  a_zPtr   = new float[m_total_valid_3D_ele];
	  loadAndCacheZEle(a_timeState,a_zPtr);
  }
  else
  {
	   a_mapper = new int [m_num_mesh_nodes*m_num_layers];
       for(int i=0;i<m_num_mesh_nodes*m_num_layers;i++)
       {
	      a_mapper[i]= MeshConstants::INVALID_NUM;
       }
       int Index = 0 ;
       for(int iLayer=0;iLayer<m_num_layers;iLayer++)
       {
	     for(int iNode=0;iNode<m_num_mesh_nodes;iNode++)
	     {
		    int bottomLayer = m_kbp00[iNode];
		    if (bottomLayer<=(iLayer+1))
			  {
				  a_mapper[iLayer+iNode*m_num_layers] = Index;
				  Index++;
			  }
	      }
       }
	  a_zPtr = new float[m_total_valid_3D_point];
	  debug1<<"load z for average size "<<m_total_valid_3D_point<<"\n";
	  loadAndCacheZ(a_timeState,a_zPtr);
	  debug1<<"done load z for average\n";
  }


 }

// ****************************************************************************
//  Method: avtSCHISMFileFormatImpl::depthAverage
//
//  Purpose:
//      Averages layred state over depth and return the result
//      
//      
//
//  Arguments:
//      a_averageState   result average state
//      a_layeredState   state to be averaged
//
//  Programmer: qshu -- generated by xml2avt
//  Creation:   Mon Sep  14:15:00 PDT 2012
//
// ****************************************************************************
void 
avtSCHISMFileFormatImpl:: depthAverage(float         *  a_averageState,
                                   float         *  a_layeredState,
								   int           *  a_nodeDataStart,
                                   const int     &  a_timeState
                                  ) 
{
  
  //int numLayers=m_num_layers;
  int numData=m_nominal_size_per_layer[m_data_center];
 
  int * kbp; 
  int * mapper;
  float * zPtr;
  
  prepare_average(kbp, mapper, zPtr, a_timeState);

  debug1<<"aveages states output \n";

  for( int iNode = 0 ;iNode< numData;iNode++)
    {
	  // this is number of valid data
	  int num_valid_data_layers =m_num_layers-std::max(1,kbp[iNode])+1;

	  
	   // native prism center layer 1 is duplicate of layer2
	   // it still have the same number of layers as the node center data
	  if (m_level_center == MeshConstants::HALF_LAYER)
      {
	    num_valid_data_layers--;
      }

	  if (num_valid_data_layers<1)
	  {
		 stringstream msgStream(stringstream::out);
         msgStream <<"less than one layer at node"<<iNode<<" when averaging\n";
        EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
	  }
	 

      float*  states  = new float [num_valid_data_layers];
	  int     startid = a_nodeDataStart[iNode];
	 
      for(int iLayer  = 0;iLayer<num_valid_data_layers; iLayer++)
        {
          states[iLayer] = a_layeredState[startid+iLayer]; 
        }


      float*  zCoords = new float [m_num_layers-std::max(1,kbp[iNode])+1];  //take all the valid zs for this node
	
	  int iiLayer=0;
	  for(int iLayer  = 0;iLayer<m_num_layers; iLayer++)
       {
		  if (iLayer>=(std::max(1,kbp[iNode])-1))
		  {
		    int index = mapper[iLayer+iNode*m_num_layers];
            zCoords[iiLayer] = zPtr[index]; 
			iiLayer++;
			
		  }
		
       }
	 
	   int num_valid_z_layers = iiLayer;
	   float averageState;
	   if (m_level_center != MeshConstants::HALF_LAYER)
	   {
		
            averageState  = trapezoidAverage(states,
                                             zCoords,
                                             num_valid_data_layers);
	   }
	   else
	   {
		    averageState  = rectAverage(states,
                                        zCoords,
                                        num_valid_z_layers);
	   }
	   
	  
      
      a_averageState[iNode] = averageState;
      delete   states;
      delete   zCoords;
    }  
    delete zPtr;
	delete mapper;
    debug1<<numData<<" center "<<m_data_center<<"\n";
}


// ****************************************************************************
//  Method: avtSCHISMFileFormatImpl::vectorDepthAverage
//
//  Purpose:
//      Averages layred state over depth and return the result
//      
//      
//
//  Arguments:
//      a_averageState   result average state
//      a_layeredState   state to be averaged
//
//  Programmer: qshu -- generated by xml2avt
//  Creation:   Mon Sep  14:15:00 PDT 2012
//
// ****************************************************************************
void 
avtSCHISMFileFormatImpl:: vectorDepthAverage(float        *  a_averageState,
                                         float        *  a_layeredState,
									     int          *  a_nodeDataStart,
                                         const int    &  a_timeState,
                                         const int    &  a_ncomps
                                         ) 
{
  
  int numData=m_nominal_size_per_layer[m_data_center];
 
  int * kbp; 
  int * mapper;
  float * zPtr;
  
  prepare_average(kbp, mapper, zPtr, a_timeState);

  for( int iNode = 0 ;iNode< numData;iNode++)
    { 
	  int num_valid_data_layers =m_num_layers-std::max(1,kbp[iNode])+1;
	 
	  if (m_level_center == MeshConstants::HALF_LAYER)
      {
	    num_valid_data_layers--;
      }

	  if (num_valid_data_layers<1)
	  {
		 stringstream msgStream(stringstream::out);
         msgStream <<"less than one level at node"<<iNode<<" when averaging\n";
        EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
	  }
	 

      float**  states  = new float* [a_ncomps];
      for(int iComp = 0; iComp < a_ncomps; iComp++)
        {
          states[iComp] = new float [num_valid_data_layers]; 
        }
	  int     startid = a_nodeDataStart[iNode];
      for(int iLayer  = 0;iLayer < num_valid_data_layers; iLayer++)
        {
          for(int iComp = 0; iComp < a_ncomps; iComp++)
            {
             states[iComp][iLayer] = a_layeredState[iComp+iLayer*a_ncomps+startid]; 
			 if (iNode==1850)
			 {
				 debug1<<states[iComp][iLayer]<<" ";
			 }
            }
        }

      float*  zCoords = new float [m_num_layers-std::max(1,kbp[iNode])+1];  //take all the zs for the mesh  
	 
	  int iiLayer=0;

	   if(iNode==1850)
	  {
		  debug1<<" z of 1850 : ";
	  }
	  
	 
	  for(int iLayer  = 0;iLayer<m_num_layers; iLayer++)
      {
		  if (iLayer>=(std::max(1,kbp[iNode])-1))
		  {
		    long index        = mapper[iLayer+iNode*m_num_layers];
            zCoords[iiLayer] = zPtr[index]; 
			if(iNode==1850)
	        {
		       debug1<<zCoords[iiLayer]<<" "<<iLayer<<"->"<<index<<" ";
	        }
			iiLayer++;
		  }
		 
      }
	  
	  int num_valid_z_layers = iiLayer;

	  if(iNode==1850)
	  {
		  debug1<<" averaged sate of 1850 : ";
	  }
	  
	  
       for( int iComp = 0; iComp < a_ncomps; iComp++)
       {
		 float averageState;
		 if (m_level_center!=MeshConstants::HALF_LAYER)
		 {
              averageState  = trapezoidAverage(states[iComp],
                                               zCoords,
                                               num_valid_data_layers);
			
			  
		 }
		 else
		 {
			  averageState  = rectAverage(states[iComp],
                                          zCoords,
                                          num_valid_z_layers);
			if(iNode==1850)
	        {
		      debug1<<averageState<<" ";
	        }
	  
		 }
       
         a_averageState[iNode*a_ncomps+iComp] = averageState;
       }
	    if(iNode==1850)
	        {
		      debug1<<" end 1850\n ";
	        }
       for(int iComp = 0; iComp < a_ncomps; iComp++)
        {
          delete  states[iComp]; 
        }
       delete   states;
       delete   zCoords;
    }  

  delete zPtr;
  delete mapper;

}


// ****************************************************************************
//  Method: avtSCHISMFileFormatImpl::GetSingleLayerVar
//
//  Purpose:
//      Gets a scalar variable asscocated with mesh with only one layer.
//      
//      
//
//  Arguments:
//      a_timeState The index of the a_timeState.  If GetNTimesteps returned
//                  'N' time steps, this is guaranteed to be between 0 and N-1.
//      a_varName   The netcdf name of the variable requested.
//
//  Programmer: qshu -- generated by xml2avt
//  Creation:   Wed Aug 21 10:11:04 PDT 2012
//
// ****************************************************************************

void 
avtSCHISMFileFormatImpl::getSingleLayerVar(float    *          a_valBuff,
                                       SCHISMFile*         a_SCHISMOutPtr,
                                       const int &         a_timeState, 
                                       const std::string&  a_varName) const
{
  time_t startTicks      = clock();
  SCHISMVar * SCHISMVarPtr = a_SCHISMOutPtr->get_var(a_varName);
  if (!(SCHISMVarPtr->is_valid()))
    {
      
      EXCEPTION1(InvalidVariableException, a_varName);
    }
  debug1<<"begin to read "<<a_varName<<endl;

  int nodeIndex0  = 0;
  int timeRecord  = a_timeState;
   
  SCHISMVarPtr->set_cur(timeRecord);
    
  debug1<<"set start of  "<<a_varName<<endl;
  int numOfRecord  = 1;
   
  if (!SCHISMVarPtr->get(a_valBuff))
  {
         stringstream msgStream(stringstream::out);
         msgStream <<"Fail to retrieve "<<a_varName << " at step " <<a_timeState;
         EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
   }
   
  time_t endTicks      = clock();
  debug1<<"time used in getting var "<<a_varName<<":"<<endTicks-startTicks<<endl;
}




// ****************************************************************************
//  Method: avtSCHISMFileFormatImpl::ActivateTimestep
//
//  Purpose:
//      
//      
//  Programmer: qshu -- generated by xml2avt
//  Creation:   Mon Aug 6 09:53:36 PDT 2012
//
// ****************************************************************************

void  avtSCHISMFileFormatImpl::ActivateTimestep(const std::string& a_filename)
{
  Initialize(a_filename);
}

// ****************************************************************************
//  Method: avtSCHISMFileFormatImpl::Initialize
//
//  Purpose:
//      Try to open the saved input data filename using ncfile and check
//      if the file is a valid netcdf file. If valid save ncfile pointer.
//     
//
//  Programmer: qshu
//  Creation:   Mon Aug 6 09:53:36 PDT 2012
//
// ****************************************************************************
void avtSCHISMFileFormatImpl::Initialize(const std::string  & a_data_file)
{
  
  if (!m_initialized)
  {
    bool okay = false;
    // Open the file specified by the filename argument here using
    // ncfile API. See if the file has the right things in
    // it. If so, set okay to true.
    debug1<<"file is going to be opened\n";
	m_data_file = a_data_file;
    size_t found = m_data_file.find_last_of("/\\");
    m_data_file_path = m_data_file.substr(0,found);

    size_t startPos       = m_data_file.find_last_of(".");
    std::string suffix;
	debug1<<"done spliting filename";
    if (!(startPos == std::string::npos))
    {
       suffix  = m_data_file.substr(startPos+1,2);
    }
	
    if(suffix=="nc")
    {  
	    debug1<<" nc file\n";
	    m_data_file_ptr=new NetcdfSchismOutput(m_data_file);
	    debug1<<"created netcdf file\n";
    }
	else if ((suffix=="61")||(suffix=="62")||(suffix=="63")||(suffix=="64")||(suffix=="65")
			||(suffix=="66")||(suffix=="67")||(suffix=="68")||(suffix=="69")||(suffix=="70"))
    {
	    debug1<<"native schism file\n";
	    m_data_file_ptr=new NativeSchismOutput(m_data_file);
	    debug1<<"created native file\n";
    }
	else
	{
		EXCEPTION1(InvalidDBTypeException,"The file is not a valid SCHISM file");
	}
	

    okay = m_data_file_ptr->is_valid();
   
    // If your file format API could not open the file then throw   
    // an exception.
     if (!okay)
      {
        EXCEPTION1(InvalidDBTypeException,
                   "The file is not a valid SCHISM output file");
      }
    debug1<<"file is opened\n";
    debug1<<"begin get dim\n";

    char * file_dir = new char [MAX_PATH_LEN];
    char * file_name = new char [MAX_FILE_NAME_LEN];
    char * file_ext = 0;
    decomposePath(m_data_file.c_str(), file_dir, file_name, file_ext);

	std::string file_name_str(file_name);
	std::string file_dir_str(file_dir);

	//found = m_data_file.find_last_of("_");
	found = file_name_str.find_first_of("_");
   // std::string zCorFilePath = m_data_file.substr(0,found)+"_zcor.63";
	std::string zCorFilePath = file_dir_str+file_name_str.substr(0,found)+"_zcor.63";
	if (suffix=="nc")
	{
		//zCorFilePath = m_data_file.substr(0,found)+"_zcor.nc";
		zCorFilePath = file_dir_str+file_name_str.substr(0,found)+"_zcor_63.nc";
	}
	debug1<<"loading zcor/hgrid file";

	found  = m_data_file.find_last_of(".");
	std::string typeStr = m_data_file.substr(found);
	
	
    debug1<<"opening zcor file "<<zCorFilePath;

    m_external_mesh_provider=new ZCoordMeshProvider(zCorFilePath);
    okay = m_external_mesh_provider->isValid();
	
	if (!okay)
	{
		delete m_external_mesh_provider;
		
		if (m_data_file_ptr->data_center() == MeshConstants::NODE)
		{
		 m_external_mesh_provider = new ComputeMeshZProvider(m_data_file);
		}
		else if (m_data_file_ptr->data_center() == MeshConstants::ELEM)
		{
			//try to find hgrid.gr3
			std::string gr3file = m_data_file_path+"/hgrid.gr3";
			debug1<<"try to load mesh from "<<gr3file<<"\n";

			m_external_mesh_provider=new Gr3MeshProvider(gr3file);
			bool ok = m_external_mesh_provider->isValid();
			if(!ok)
			{
				delete m_external_mesh_provider;
				EXCEPTION1(InvalidDBTypeException,
					"The file could not be opened for no zcore or hgrid mesh exist");
			}
		}
		else
		{
			EXCEPTION1(InvalidDBTypeException,
					"The file could not be opened for mesh file exist");
		}
		 
	 }

	debug1<<"getting dim";
	getMeshDimensions(m_external_mesh_provider);
	debug1<<"loading coords";
    loadMeshCoordinates(m_external_mesh_provider);

    debug1<<"got dimension\n";

    getTime();
    debug1<<"got time\n";
	
	m_data_center = m_data_file_ptr->data_center();
	m_level_center=  m_data_file_ptr->level_center();

	debug1<<"got level info\n";

	int kbp_data_size = m_nominal_size_per_layer[MeshConstants::NODE];
	std::string bottom_index_var_name = MeshConstants::NODE_BOTTOM;

	if (m_data_center == MeshConstants::ELEM)
	{
		kbp_data_size = m_nominal_size_per_layer[MeshConstants::ELEM];
		bottom_index_var_name = MeshConstants::FACE_BOTTOM;
	}
	else if (m_data_center == MeshConstants::EDGE)
	{
		kbp_data_size = m_nominal_size_per_layer[MeshConstants::EDGE];
		bottom_index_var_name = MeshConstants::EDGE_BOTTOM;
	}
 
	debug1<<"kbp data size is "<<kbp_data_size<<"\n";

	m_kbp_data = new int [kbp_data_size];
	SCHISMVar * kVarPtr = m_data_file_ptr->get_var(bottom_index_var_name);
    kVarPtr->get(m_kbp_data);
	debug1<<"get kbp\n";

    PopulateVarMap();
	
    m_initialized = true;
	delete file_dir;
	delete file_name;
	debug1<<"done initialize\n";
  }
}




// a reduant populate var map dic, fix a bug in which this dic is 
// lost when switch a different data file group in the same .visit file
// called in initialize()
void avtSCHISMFileFormatImpl::PopulateVarMap()
{
  int numVar = m_data_file_ptr->num_vars();
  std::string  location = m_data_file_ptr->data_center();

  for(int iVar = 0;iVar < numVar; iVar++)
   {
      SCHISMVar*  varPtr  = m_data_file_ptr->get_var(iVar);
      std::string varName = varPtr->name();
     
      if ((varName==m_node_surface) || (varName==m_node_depth))
        {
          continue; 
        }
      std::string  label;
      label = varName;
      // this dic make it easy to find out data set for a visit plot variable
      m_var_name_label_map[label] = varName;

     if( (location ==NODE)||(location==FACE))
     {
          // all those surface, botootm and average are based on original data set
          m_var_name_label_map[ label+m_surface_state_suffix] = varName;
          m_var_name_label_map[ label+m_bottom_state_suffix]  = varName;
          m_var_name_label_map[ label+m_depth_average_suffix] = varName;

          m_var_mesh_map[ label+m_surface_state_suffix ]     = m_mesh_2d;
          m_var_mesh_map[ label+m_bottom_state_suffix ]      = m_mesh_2d;
          m_var_mesh_map[ label+m_depth_average_suffix ]     = m_mesh_2d;
      
     }
    // omit unkown center data
    else
     {
      continue;
     }
   }
}






// ****************************************************************************
//  Method: avtSCHISMFileFormatImpl::GetTimes
//
//  Purpose:
//      overloaded public interface to return time steps
//      
//     
//
//  Programmer: qshu
//  Creation:   Mon Aug 15 03:02:00 PDT 2012
//
// ****************************************************************************
void   avtSCHISMFileFormatImpl::GetTimes(std::vector<double> & a_times)
{

  //copy saved time into a_times
  for(int i=0;i<m_num_time_step;i++)
    {
      a_times.push_back(m_time_ptr[i]);
    }   

}

// ****************************************************************************
//  Method: avtSCHISMFileFormatImpl::getTime
//
//  Purpose:
//      Try to get time steps saved in this file
//      
//     
//
//  Programmer: qshu
//  Creation:   Mon Aug 15 10:11:00 PDT 2012
//
// ****************************************************************************
void  avtSCHISMFileFormatImpl::getTime()
{

  SCHISMDim * dimTimePtr      = m_data_file_ptr->get_dim(m_dim_time);

  if (dimTimePtr->is_valid())
    {
      m_num_time_step       = dimTimePtr->size();
    }
  else
    {
      EXCEPTION1(InvalidVariableException, m_dim_time);
    }
 

  if (m_num_time_step)
    {
      m_time_ptr               = new float [m_num_time_step];
    
      SCHISMVar * SCHISMTimePtr = m_data_file_ptr->get_var(m_time);
    
      if (SCHISMTimePtr->is_valid())
        {
         SCHISMTimePtr->get(m_time_ptr);
        }
      else
        {
          EXCEPTION1(InvalidVariableException, m_time);
        }
     
    }
}



bool  avtSCHISMFileFormatImpl::SCHISMVarIs3D(SCHISMVar*  a_varPtr ) const
{
	int totalNumDim=a_varPtr->num_dims();
	for (int dim=0;dim<totalNumDim;dim++)
	{
		SCHISMDim* dimPtr=a_varPtr->get_dim(dim);
		if (dimPtr->name()==m_dim_layers)
			return true;
	}
	return false;
}
bool  avtSCHISMFileFormatImpl::SCHISMVarIsVector(SCHISMVar* a_varPtr) const
{
	int totalNumDim=a_varPtr->num_dims();
	for (int dim=0;dim<totalNumDim;dim++)
	{
		SCHISMDim* dimPtr=a_varPtr->get_dim(dim);
		if (dimPtr->name()==m_dim_var_component)
			return true;
	}
	return false;
}


static Registrar registrar("native_binary", &avtSCHISMFileFormatImpl::create);
