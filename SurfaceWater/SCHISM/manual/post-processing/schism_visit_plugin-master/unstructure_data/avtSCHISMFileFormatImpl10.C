
#include <avtSCHISMFileFormatImpl10.h>

#include <string>
#include <iostream>
#include <sstream>
#include <time.h>
#include <math.h>
#include <algorithm>

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


#include "ZCoordFileMeshProvider10.h"
#include "SCHISMFileUtil10.h"
#include "Average.h"
#include "MeshConstants10.h"
#include "NetcdfSCHISMOutput10.h"
#include "avtSCHISMFileFormat.h"
#include "Registar.h"

using     std::string;
using     std::stringstream;

const std::string NODE      = MeshConstants10::NODE;
const std::string FACE      = MeshConstants10::ELEM;
const std::string SIDE      = MeshConstants10::EDGE;
const std::string UNKOWN    ="unkown";
const int NODESPERELE       = MeshConstants10::MAX_NUM_NODE_PER_CELL;
const int NODESPERWEDGE     = NODESPERELE*2;


std::map<std::string,std::string> Vector3DVarMapX;
std::map<std::string,std::string> Vector3DVarMapY;

std::map<std::string, std::string> Vector2DVarMapX;
std::map<std::string, std::string> Vector2DVarMapY;

avtSCHISMFileFormatImpl10::avtSCHISMFileFormatImpl10():
      m_initialized(false),
	  m_mesh_is_static(true),
      m_plugin_name("SCHISM_output_plugin"),
      m_data_description("data_description"),
      m_mesh_var("Mesh2"),
      m_var_label_att("long_name"),
      m_var_location_att("location"),
      m_mesh_3d("3D_Mesh"),
	  m_layer_mesh("Layer_Mesh"),
      m_mesh_2d("2D_Mesh"),
	  m_mesh_2d_no_wet_dry("2D_Mesh_no_wet_dry"),
	  m_side_center_point_3d_mesh("side_center_3D"),
	  m_side_center_point_2d_mesh("side_center_2D"),
	  m_face_center_point_3d_mesh("face_center_3D"),
      m_dim_time(MeshConstants10::DIM_TIME),
      m_time(MeshConstants10::TIME),
      m_node_depth(MeshConstants10::NODE_DEPTH),
      m_node_depth_label(MeshConstants10::NODE_DEPTH),
      m_dim_layers(MeshConstants10::DIM_LAYERS),
	  m_dim_var_component(MeshConstants10::DIM_VAR_COMPONENT),
	  m_time_ptr(NULL),
      m_node_x_ptr(NULL),
      m_node_y_ptr(NULL),
	  m_node_z_ptr(NULL),
	  m_kbp00(NULL),
	  m_kbp_data(NULL),
	  m_kbp_node(NULL),
	  m_kbp_side(NULL),
	  m_kbp_ele(NULL),
	  m_node_dry_wet(NULL),
	  m_ele_dry_wet(NULL),
	  m_side_dry_wet(NULL),
	  m_kbp_node_filled(false),
	  m_kbp_side_filled(false),
	  m_kbp_ele_filled(false),
	  m_node_dry_wet_cached_time(-1),
	  m_ele_dry_wet_cached_time(-1),
      m_side_dry_wet_cached_time(-1),
	  m_cache_kbp_id(-1),
	  m_data_file_ptr(NULL),
      m_surface_state_suffix("_surface"),
      m_bottom_state_suffix("_near_bottom"),
      m_depth_average_suffix("_depth_average"),
      m_dry_surface(MeshConstants10::DRY_SURFACE),
	  m_total_valid_3D_point(0),
	  m_total_valid_3D_side(0),
	  m_total_valid_3D_ele(0),
	  m_dry_wet_flag(0),
	  m_scribeIO(false)
{
  // AVT_NODECENT, AVT_ZONECENT, AVT_UNKNOWN_CENT
  m_center_map[NODE]  = AVT_NODECENT;
  m_center_map[FACE]  = AVT_ZONECENT;
  m_center_map[UNKOWN]= AVT_UNKNOWN_CENT;
  m_var_name_label_map[m_node_depth_label]    = m_node_depth;

  Vector3DVarMapX["horizontalVelX"]="horizontalVelX;horizontalVelY";
  Vector3DVarMapY["horizontalVelY"]="horizontalVelX;horizontalVelY";
  Vector3DVarMapX["horizontalSideVelX"]="horizontalSideVelX;horizontalSideVelY";
  Vector3DVarMapY["horizontalSideVelY"]="horizontalSideVelX;horizontalSideVelY";
  Vector3DVarMapX["waveForceX"]="waveForceX;waveForceY";
  Vector3DVarMapY["waveForceY"]="waveForceX;waveForceY";
  Vector3DVarMapX["horzontalViscosityX"]="horzontalViscosityX;horzontalViscosityY";
  Vector3DVarMapY["horzontalViscosityY"]="horzontalViscosityX;horzontalViscosityY";
  Vector3DVarMapX["baroclinicForceX"]="baroclinicForceX;baroclinicForceY";
  Vector3DVarMapY["baroclinicForceY"]="baroclinicForceX;baroclinicForceY";
  Vector3DVarMapX["verticalViscosityX"]="verticalViscosityX;verticalViscosityY";
  Vector3DVarMapY["verticalViscosityY"]="verticalViscosityX;verticalViscosityY";
  Vector3DVarMapX["mommentumAdvectionX"]="mommentumAdvectionX;mommentumAdvectionY";
  Vector3DVarMapY["mommentumAdvectionY"]="mommentumAdvectionX;mommentumAdvectionY";


  Vector2DVarMapX["airPressureGradientX"]="airPressureGradientX;airPressureGradientY";
  Vector2DVarMapY["airPressureGradientY"]="airPressureGradientX;airPressureGradientY";
  Vector2DVarMapX["tidePotentialGradX"]="tidePotentialGradX;tidePotentialGradY";
  Vector2DVarMapY["tidePotentialGradY"]="tidePotentialGradX;tidePotentialGradY";

  Vector2DVarMapX["bottomStressX"]="bottomStressX;bottomStressY";
  Vector2DVarMapY["bottomStressY"]="bottomStressX;bottomStressY";
  Vector2DVarMapX["windSpeedX"]="windSpeedX;windSpeedY";
  Vector2DVarMapY["windSpeedY"]="windSpeedX;windSpeedY";
  Vector2DVarMapX["windStressX"]="windStressX;windStressY";
  Vector2DVarMapY["windStressY"]="windStressX;windStressY";
  Vector2DVarMapX["depthAverageVelX"]="depthAverageVelX;depthAverageVelY";
  Vector2DVarMapY["depthAverageVelY"]="depthAverageVelX;depthAverageVelY";
  Vector2DVarMapX["waveEnergyDirX"] = "waveEnergyDirX;waveEnergyDirY";
  Vector2DVarMapX["waveEnergyDirY"] = "waveEnergyDirX;waveEnergyDirY";
  Vector2DVarMapX["sedBedloadTransportX"] = "sedBedloadTransportX;sedBedloadTransportY";
  Vector2DVarMapX["sedBedloadTransportY"] = "sedBedloadTransportX;sedBedloadTransportY";

  Vector2DVarMapX["iceVelocityX"] = "iceVelocityX;iceVelocityY";
  Vector2DVarMapX["iceVelocityY"] = "iceVelocityX;iceVelocityY";

  for (int i = 0; i < 10; i++) //max support 10 kinds of sediment
  {
	  stringstream x(stringstream::out), y(stringstream::out);
	  x << "sedBedloadX_" << i;
	  y << "sedBedloadY_" << i;
	  stringstream xy(stringstream::out);
	  xy << "sedBedloadX_" << i<<";"<< "sedBedloadY_"<<i;
	  Vector2DVarMapX[x.str()] = xy.str();
	  Vector2DVarMapY[y.str()] = xy.str();
  }

}

FileFormatFavorInterface * avtSCHISMFileFormatImpl10::create()
{
	return new avtSCHISMFileFormatImpl10();
}


// ****************************************************************************
//  Method: avtEMSTDFileFormat::GetNTimesteps
//
//  Purpose:
//      Tells the rest of the code how many timesteps there are in this file.
//
//  Programmer: qshu -- generated by xml2avt
//  Creation:   Wed Mar 13 09:13:49 PDT 2013
//
// ****************************************************************************

int
avtSCHISMFileFormatImpl10::GetNTimesteps(const std::string& a_filename)
{
  Initialize(a_filename);
  return m_num_time_step;
}


// ****************************************************************************
//  Method: avtSCHISMFileFormatImpl10::FreeUpResources
//
//  Purpose:
//      When VisIt is done focusing on a particular timestep, it asks that
//      timestep to free up any resources (memory, file descriptors) that
//      it has associated with it.  This method is the mechanism for doing
//      that.
//
//  Programmer: qshu -- generated by xml2avt
//  Creation:   Wed Mar 13 09:13:49 PDT 2013
//
// ****************************************************************************

void
avtSCHISMFileFormatImpl10::FreeUpResources(void)
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
   if (m_kbp_side)
   {
	   delete m_kbp_data;
   }
   if (m_kbp_node)
   {
	   delete m_kbp_data;
   }
   if (m_kbp_ele)
   {
	   delete m_kbp_data;
   }
   if (m_external_mesh_provider)
   {
	  
		   delete  m_external_mesh_provider;
		
   }

   if(m_node_dry_wet)
   {
	   delete m_node_dry_wet;
   }
   if(m_side_dry_wet)
   {
	   delete m_side_dry_wet;
   }
   if(m_ele_dry_wet)
   {
	   delete m_ele_dry_wet;
   }

  debug1<<"finish free res \n";
}


// ****************************************************************************
//  Method: avtSCHISMFileFormatImpl10::PopulateDatabaseMetaData
//
//  Purpose:
//      This database meta-data object is like a table of contents for the
//      file.  By populating it, you are telling the rest of VisIt what
//      information it can request from you.
//
//  Programmer: qshu -- generated by xml2avt
//  Creation:   Wed Mar 13 09:13:49 PDT 2013
//
// ****************************************************************************

void
avtSCHISMFileFormatImpl10::PopulateDatabaseMetaData(avtDatabaseMetaData *a_metaData, avtSCHISMFileFormat * a_avtFile, int a_timeState)
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
    //add node center 3d mesh
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

	 //add 3d side center point mesh
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

	 //add 3d  face center point mesh
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
	if(!m_scribeIO)
	{
    a_avtFile->addScalarVarToMetaData(a_metaData, m_node_depth_label,   mesh, nodeCent);
	m_var_name_label_map[m_node_depth_label] = "depth";
	}

	//for scribeIO add elev and wind
	//if (m_scribeIO)
	//{

		
	//	a_avtFile->addScalarVarToMetaData(a_metaData, MeshConstants10::NODE_SURFACE_LABEL, mesh, nodeCent);
	//	m_var_name_label_map[MeshConstants10::NODE_SURFACE_LABEL] = "elevation";
	//}


   // m_var_mesh_map[m_node_surface_label] = mesh;
    m_var_mesh_map[m_node_depth_label]   = mesh;

	//add 3D node level label
	string mesh3d     = m_mesh_3d;
    a_avtFile->addScalarVarToMetaData(a_metaData,  MeshConstants10::NODE_LEVEL, mesh3d, nodeCent);

	//add 3D element level label
    mesh3d = m_layer_mesh;
	a_avtFile->addScalarVarToMetaData(a_metaData,  MeshConstants10::ELE_LEVEL, mesh3d, zoneCent);

	//add 3D side level label
	mesh3d =  m_side_center_point_3d_mesh;
	a_avtFile->addScalarVarToMetaData(a_metaData,  MeshConstants10::SIDE_LEVEL, mesh3d, nodeCent);

	//add 3D layer lable (for prism center data)
	mesh3d = m_mesh_3d;
	a_avtFile->addScalarVarToMetaData(a_metaData,  MeshConstants10::LAYER, mesh3d, zoneCent);


	//add node bottom label
	mesh     = m_mesh_2d;
	 //a_avtFile->addScalarVarToMetaData(a_metaData,  MeshConstants10::NODE_BOTTOM, mesh, nodeCent);
	 //a_avtFile->addScalarVarToMetaData(a_metaData,  MeshConstants10::FACE_BOTTOM, mesh, zoneCent);

	//a_avtFile->addScalarVarToMetaData(a_metaData,  MeshConstants10::EDGE_BOTTOM, mesh, nodeCent);

    
    PopulateStateMetaData(a_metaData,a_avtFile,a_timeState);
    debug1<<"finish populate metadata \n";
}

void    avtSCHISMFileFormatImpl10::addFaceCenterData(avtDatabaseMetaData * a_metaData,
	                                                SCHISMVar10           * a_varPtr,
													 avtSCHISMFileFormat * a_avtFile,
								                    const std::string   & a_varName,
								                    const std::string   & a_varLabel,
								                    const avtCentering  & a_center)

{
	  // only add face centered  var now 
       string mesh2d     = m_mesh_2d;
	   string mesh3d     = m_mesh_3d;
	   std::string level_center = a_varPtr->get_vertical_center();

       if (level_center == MeshConstants10::FULL_LAYER)
	   {
		   mesh3d =  m_layer_mesh;
	   }
      
       avtCentering  faceCent(AVT_ZONECENT);
       // scalar data 2d mesh
       if (a_varPtr->num_dims()<=2)
        {     
           a_avtFile->addScalarVarToMetaData(a_metaData, a_varLabel, mesh2d, faceCent);
          m_var_mesh_map[a_varLabel] = mesh2d;
		  m_var_dim[a_varName]=a_varPtr->num_dims();
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
              SCHISMDim10* comDim = a_varPtr->get_dim(3);
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
		    //AddScalarVarToMetaData(a_metaData,  MeshConstants10::LEVEL, mesh3d, faceCent);
        }
	   else
	   {

	   }


}
void    avtSCHISMFileFormatImpl10::addNodeCenterData(avtDatabaseMetaData * a_metaData,
	                                           SCHISMVar10            * a_varPtr,
											    avtSCHISMFileFormat * a_avtFile,
								               const std::string   & a_varName,
								               const std::string   & a_varLabel,
								               const avtCentering  & a_center)
{

	  std::string varName(a_varName);
	  std::string label(a_varLabel);
	  avtCentering avtCenter(a_center);

	 //  scalar var on 2D
	   if (a_varPtr->num_dims()<=2)
	   {
		  a_avtFile->addScalarVarToMetaData(a_metaData,label, m_mesh_2d, avtCenter);   
          m_var_mesh_map[label] = m_mesh_2d;
		  m_var_dim[varName]=a_varPtr->num_dims();
		  debug1<<"added 2d scalar:"<<label;
	   }
	    //  vector var on 2D
	   else if ((a_varPtr->num_dims()==3) && (!SCHISMVarIs3D(a_varPtr)))
	   {
		  SCHISMDim10* comDim = a_varPtr->get_dim(3);
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

		  //AddScalarVarToMetaData(a_metaData,  MeshConstants10::LEVEL, m_mesh_3d, avtCenter);
          a_avtFile->addScalarVarToMetaData(a_metaData,label, m_mesh_3d, avtCenter);   
          m_var_mesh_map[label] = m_mesh_3d;
          // also add bottom, surface and depth average state option
		  if (!(varName==MeshConstants10::ZCOORD))
		  {
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
		  }
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
          SCHISMDim10* comDim = a_varPtr->get_dim(3);
          int ncomps       = comDim->size();
          int ucomps       = (ncomps == 2 ? 3 : ncomps);

          a_avtFile->addVectorVarToMetaData(a_metaData,label, m_mesh_3d, avtCenter,ucomps);   
		  //AddScalarVarToMetaData(a_metaData,  MeshConstants10::LEVEL, m_mesh_3d, avtCenter);
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

void    avtSCHISMFileFormatImpl10::addSideCenterData(avtDatabaseMetaData * a_metaData,
	                                           SCHISMVar10           * a_varPtr,
											    avtSCHISMFileFormat * a_avtFile,
								               const std::string   & a_varName,
								               const std::string   & a_varLabel,
								               const avtCentering  & a_center)
{
	  // only add centered  var now 
       string mesh2d     =  m_side_center_point_2d_mesh;
	   string mesh3d     =  m_side_center_point_3d_mesh;
	   std::string level_center = a_varPtr->get_vertical_center();

       if (level_center == MeshConstants10::HALF_LAYER)
	   {
		   mesh3d = m_face_center_point_3d_mesh;
	   }

       avtCentering  nodeCent(AVT_NODECENT);
       std::string varName(a_varName);
	   std::string label(a_varLabel);
	   avtCentering avtCenter(a_center);

	 //  scalar var on 2D
	   if (a_varPtr->num_dims()<=2)
	   {
		  a_avtFile->addScalarVarToMetaData(a_metaData,label, mesh2d, avtCenter);   
          m_var_mesh_map[label] = mesh2d;
		  m_var_dim[varName]=a_varPtr->num_dims();
		  debug1<<"added 2d scalar:"<<label;
	   }
	    //  vector var on 2D
	   else if ((a_varPtr->num_dims()==3) && (!SCHISMVarIs3D(a_varPtr)))
	   {
		  SCHISMDim10* comDim = a_varPtr->get_dim(3);
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
		  //AddScalarVarToMetaData(a_metaData,  MeshConstants10::LEVEL, mesh3d, avtCenter);
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
          SCHISMDim10* comDim = a_varPtr->get_dim(3);
          int ncomps       = comDim->size();
          int ucomps       = (ncomps == 2 ? 3 : ncomps);

          a_avtFile->addVectorVarToMetaData(a_metaData,label, mesh3d, avtCenter,ucomps);  
		  //AddScalarVarToMetaData(a_metaData,  MeshConstants10::LEVEL, mesh3d, avtCenter);
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
//  Method: avtSCHISMFileFormatImpl10::PopulateStateMetaData
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

void avtSCHISMFileFormatImpl10::PopulateStateMetaData(avtDatabaseMetaData * a_metaData, 
	                                                   avtSCHISMFileFormat * a_avtFile,
                                                     int                   a_timeState)
{
  int numVar = m_data_file_ptr->num_vars();
  debug1<<"get vars "<<numVar<<endl;

  std::vector<std::string> out2d_vector_added;

  for(int iVar = 0;iVar < numVar; iVar++)
   {
     
      debug1<<iVar;
      SCHISMVar10*  varPtr  = m_data_file_ptr->get_var(iVar);
      debug1<<" "<<varPtr->num_dims()<<endl;

      std::string varName = varPtr->name();
	  std::string horizontal_center = varPtr->get_horizontal_center();
	  std::map<std::string,std::string>::iterator itX,itY;
	  itX = Vector3DVarMapX.find(varName);
	  itY = Vector3DVarMapY.find(varName);

	  if ((itX != Vector3DVarMapX.end())||(itY != Vector3DVarMapY.end()))
	  {
		  int ucomps = 2;
		  std::string label = "";
		  std::size_t found = varName.find_last_of("_");
		  if (found != std::string::npos)
		  {
			  label = varName.substr(0, found-1)+varName.substr(found);
		  }
		  else
		  {
			  label = varName.substr(0, varName.length() - 1);
		  }
		  avtCentering avtCenter(AVT_NODECENT);
		  avtCentering avtCenter_elem(AVT_ZONECENT);
		  std::string vars = "";
		  if(itX!=Vector3DVarMapX.end())
		  {
			  vars=Vector3DVarMapX[varName];
		  }
		  else
		  {
			  vars=Vector3DVarMapY[varName];
		  }
		  string mesh2d = m_mesh_2d;
		  string mesh3d = m_mesh_3d;
		  //std::size_t found2 = varName.find("Side");

          if (horizontal_center== MeshConstants10::EDGE)
		  {
			  mesh2d = m_side_center_point_2d_mesh;
			  mesh3d = m_side_center_point_3d_mesh;
		  }

		  if (horizontal_center == MeshConstants10::ELEM)
		  {
			  avtCenter = avtCenter_elem;
		  }
		  std::map<std::string, std::string>::iterator it;
		  it = m_var_name_label_map.find(label);

		  if (SCHISMVarIs3D(varPtr))//&&(it == m_var_name_label_map.end()))
		  {
			  a_avtFile->addVectorVarToMetaData(a_metaData, label, mesh3d, avtCenter, ucomps);
			  //a_avtFile->addScalarVarToMetaData(a_metaData,varName, m_mesh_3d, avtCenter);   
              //m_var_mesh_map[varName] = m_mesh_3d;
			  m_var_name_label_map[label] = vars;
			  m_var_mesh_map[label] = mesh3d;

			  // also add bottom, surface and depth average state option
			  a_avtFile->addVectorVarToMetaData(a_metaData,
				  label + m_surface_state_suffix,
				  mesh2d,
				  avtCenter,
				  ucomps);
			  //debug1 << "add  " << label + m_surface_state_suffix << " ";
			  a_avtFile->addVectorVarToMetaData(a_metaData,
				  label + m_bottom_state_suffix,
				  mesh2d,
				  avtCenter,
				  ucomps);
			  //debug1 << "add  " << label + m_bottom_state_suffix << " ";
			  a_avtFile->addVectorVarToMetaData(a_metaData,
				  label + m_depth_average_suffix,
				  mesh2d,
				  avtCenter,
				  ucomps);
			  // debug1 << "add  " << label + m_depth_average_suffix << " ";
			  m_var_name_label_map[label + m_surface_state_suffix] = vars;
			  m_var_name_label_map[label + m_bottom_state_suffix] = vars;
			  m_var_name_label_map[label + m_depth_average_suffix] = vars;
			  m_var_mesh_map[label + m_surface_state_suffix] = mesh2d;
			  m_var_mesh_map[label + m_bottom_state_suffix] = mesh2d;
			  m_var_mesh_map[label + m_depth_average_suffix] = mesh2d;
			  m_var_dim[varName] = 3;
		  }

		  continue;

	  }

	  itX = Vector2DVarMapX.find(varName);
	  itY = Vector2DVarMapY.find(varName);
	  if ((itX != Vector2DVarMapX.end()) || (itY != Vector2DVarMapY.end()))
	  {
		  int ucomps = 2;
		  std::string label = "";
		  if ((varName[varName.length() - 1] == 'Y') || (varName[varName.length() - 1] == 'X'))
		  {
			  label = varName.substr(0, varName.length() - 1);
		  }
		  else
		  {
			  std::size_t found = varName.find('_');
			  if (found != std::string::npos)
			  {
				  label= varName.substr(0, found - 1)+ varName.substr(found, varName.length());
			  }
			  else
			  {
				  EXCEPTION1(InvalidVariableException, varName);
			  }
		  }
		  avtCentering avtCenter(AVT_NODECENT);
		  avtCentering avtCenter_elem(AVT_ZONECENT);
		  std::string vars = "";
		  if (itX != Vector2DVarMapX.end())
		  {
			  vars = Vector2DVarMapX[varName];
		  }
		  else
		  {
			  vars = Vector2DVarMapY[varName];
		  }
		  string mesh2d = m_mesh_2d;
		  string mesh3d = m_mesh_3d;
		  //std::size_t found2 = varName.find("Side");

		  if (horizontal_center == MeshConstants10::EDGE)
		  {
			  mesh2d = m_side_center_point_2d_mesh;
			  mesh3d = m_side_center_point_3d_mesh;
		  }

		  if (horizontal_center == MeshConstants10::ELEM)
		  {
			  avtCenter = avtCenter_elem;
		  }
		 
		 

		  if (std::find(out2d_vector_added.begin(), out2d_vector_added.end(), label)==out2d_vector_added.end())
		  {
			  a_avtFile->addVectorVarToMetaData(a_metaData, label, mesh2d, avtCenter, ucomps);
			  m_var_name_label_map[label] = vars;
			  m_var_mesh_map[label] = mesh2d;
			  out2d_vector_added.push_back(label);
			  
		  }
		  
		  continue;

	  }

	  if (m_data_file_ptr->none_data_var(varName))
	  {
		  if((varName==m_node_depth)&&(m_scribeIO))
		  {
			  debug1<<"scribe IO depth need to add";
		  }
		  else
		  {
		    debug1<<varName<<"is skipped\n";
		     continue;
		  }
	  }

	  if(!(varPtr->is_defined_over_grid()))
	  {
		  continue;
	  }

	  if(varPtr->is_SCHISM_mesh_parameter())
	  {
		  continue;
	  }

      debug1<<varName<<endl;
     
      std::string  location(NODE);
      avtCentering avtCenter(AVT_NODECENT);
	 
     
      if (((varName==m_node_surface) || (varName==m_node_depth))&&(!m_scribeIO))
        {
          continue; 
        }
      std::string  label;
      label = varName;
      // this dic make it easy to find out data set for a visit plot variable
      m_var_name_label_map[label] = varName;

      // handle different for face and node center data
	 location = varPtr->get_horizontal_center();

	 std::string vertical_center = varPtr->get_vertical_center();

	 m_var_horizontal_center_map[varName] = location;
	 m_var_vertical_center_map[varName] = vertical_center;

     if(location ==FACE)
     {
	   addFaceCenterData(a_metaData,varPtr, a_avtFile,varName,label,avtCenter);
     }
     else if (location ==NODE)
     {
		addNodeCenterData(a_metaData,varPtr, a_avtFile,varName,label,avtCenter);  
     }
	 else if (location ==SIDE)
	 {
		 addSideCenterData(a_metaData,varPtr, a_avtFile,varName,label,avtCenter);  
	 }
    // omit unkown center data
    else
     {
      continue;
     }
   }
}


void   avtSCHISMFileFormatImpl10::create2DUnstructuredMesh( vtkUnstructuredGrid *a_uGrid,
	                                                  long                 *a_meshEle,
												      const  int          &a_timeState) 
{
	long   numNodes           = m_num_mesh_nodes;
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
       
    long *  nodePtrTemp = a_meshEle;
	load_ele_dry_wet(a_timeState);
    for(long iCell = 0; iCell < m_num_mesh_faces; ++iCell)
        {
			int numberOfNodeInCell = *nodePtrTemp;
			
			//if (!(m_ele_dry_wet[iCell]))
			//{
				if (numberOfNodeInCell ==3)
				{
				vtkIdType verts[3];
				for(int iNode=0;iNode<3;++iNode)
				{
					verts[iNode] = nodePtrTemp[iNode+1]-1;
				    
				} 
				nodePtrTemp += (MeshConstants10::MAX_NUM_NODE_PER_CELL+1) ;
				 
				a_uGrid->InsertNextCell(VTK_TRIANGLE, 3, verts);
				}
				else if (numberOfNodeInCell ==4)
				{
				vtkIdType verts[4];
				for(int iNode=0;iNode<4;++iNode)
				{
					verts[iNode] = nodePtrTemp[iNode+1]-1;
				  
				} 
				nodePtrTemp += (MeshConstants10::MAX_NUM_NODE_PER_CELL+1);
				 
				a_uGrid->InsertNextCell(VTK_QUAD, 4, verts);
				}
				else
				{
				  stringstream msgStream(stringstream::out);
				  msgStream <<"invalid cell type with number of nodes: " <<numberOfNodeInCell;
				  EXCEPTION1(InvalidVariableException,msgStream.str());
				}
			//}
             
        }
      
}


void   avtSCHISMFileFormatImpl10::create2DUnstructuredMeshNoDryWet( vtkUnstructuredGrid *a_uGrid,
	                                                  long                 *a_meshEle,
												      const  int          &a_timeState) 
{
	long   numNodes           = m_num_mesh_nodes;
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
       
    long *  nodePtrTemp = a_meshEle;
	 
    for(long iCell = 0; iCell < m_num_mesh_faces; ++iCell)
        {
			int numberOfNodeInCell = *nodePtrTemp;
			
			
			if (numberOfNodeInCell ==3)
			{
			vtkIdType verts[3];
			for(int iNode=0;iNode<3;++iNode)
			{
				verts[iNode] = nodePtrTemp[iNode+1]-1;
				    
			} 
			nodePtrTemp += (MeshConstants10::MAX_NUM_NODE_PER_CELL+1) ;
				 
			a_uGrid->InsertNextCell(VTK_TRIANGLE, 3, verts);
			}
			else if (numberOfNodeInCell ==4)
			{
			vtkIdType verts[4];
			for(int iNode=0;iNode<4;++iNode)
			{
				verts[iNode] = nodePtrTemp[iNode+1]-1;
				  
			} 
			nodePtrTemp += (MeshConstants10::MAX_NUM_NODE_PER_CELL+1);
				 
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


void   avtSCHISMFileFormatImpl10::createLayerMesh(vtkUnstructuredGrid *a_uGrid,
	                                        long                 *a_meshEle,
										    long                *a_2DPointto3DPoints,
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
      
     
	  int * kbe = m_kbp_ele;
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
						int valid_bottom = std::max(1,m_kbp_node[p])-1;
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
						int valid_bottom = std::max(1,m_kbp_node[p])-1;
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
	 
	 
}


void   avtSCHISMFileFormatImpl10::create3DUnstructuredMesh(vtkUnstructuredGrid *a_uGrid,
	                                                 long                 *a_meshEle,
												     long                 *a_2DPointto3DPoints,
												     const  int          &a_timeState) 
{
	 vtkPoints *points      = vtkPoints::New();
      points->SetNumberOfPoints(m_total_valid_3D_point);
	  debug1<<"total valid 3d pts: "<<m_total_valid_3D_point<<"\n";
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
      long *  nodePtrTemp = a_meshEle;
	  
	
	  m_tri_wedge=0;
      m_tri_pyramid=0;
      m_tri_tetra=0;

      m_quad_hexhedron=0;
      m_quad_wedge=0;
      m_quad_pyramid=0;
	  load_ele_dry_wet(a_timeState);

       for (int iLayer= 0; iLayer<m_num_layers-1;iLayer++)
        {
          nodePtrTemp    = a_meshEle;
           for(int iCell = 0; iCell < m_num_mesh_faces; ++iCell)
            {
              {
				  nodePtrTemp = a_meshEle+(MeshConstants10::MAX_NUM_NODE_PER_CELL+1)*iCell;
				  int numberOfNodeInCell = *nodePtrTemp;

				  long validTopNode[MeshConstants10::MAX_NUM_NODE_PER_CELL];
				  long validBottomNode[MeshConstants10::MAX_NUM_NODE_PER_CELL];
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
					 //nodePtrTemp += (MeshConstants10::MAX_NUM_NODE_PER_CELL+1);
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
					// nodePtrTemp += (MeshConstants10::MAX_NUM_NODE_PER_CELL+1);
				  }
                
				 else
				 {
				   //omit
				 }
                
			}
		  }
           
        }

	  debug1<<" tri_wedge "<<m_tri_wedge<<" tri_pyramid "<<m_tri_pyramid<<" tri_tetra "<<m_tri_tetra;

	  debug1<<" quad_hexhedron "<<m_quad_hexhedron<<" quad_wedge "<<m_quad_wedge<<" quad_pyramid "<<m_quad_pyramid<<"\n";
}

void    avtSCHISMFileFormatImpl10::create2DPointMesh( vtkUnstructuredGrid *a_uGrid,
	                                            long                 *a_meshEle,
										        const  int          &a_timeState) 
{

	long   numNodes           = m_num_mesh_edges;
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
    for(long i = 0; i < numNodes; ++i)
    {
       onevertex = i;
       a_uGrid->InsertNextCell(VTK_VERTEX, 1, &onevertex);
    }
 
}

void   avtSCHISMFileFormatImpl10::create3DPointMesh( vtkUnstructuredGrid *a_uGrid,
	                                                long            *a_meshEle,
										            const  int     &a_timeState) 
{
	long   numNodes           = m_total_valid_3D_side;
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
    for(long i = 0; i < numNodes; ++i)
     { 
	   onevertex = i;
       a_uGrid->InsertNextCell(VTK_VERTEX, 1, &onevertex);
    }
 
}

// this is the mesh consisits of center at 3d prism side face, used for var like flux 
void   avtSCHISMFileFormatImpl10::create3DPointFaceMesh( vtkUnstructuredGrid *a_uGrid,
	                                                long               *a_meshEle,
										            const  int         &a_timeState) 
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
//  Method: avtSCHISMFileFormatImpl10::GetMesh
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
avtSCHISMFileFormatImpl10::GetMesh(int a_timeState, avtSCHISMFileFormat * a_avtFile,const char *mesh_name)
{
  int   nDims              = 3;
  long   numNodes           = m_num_mesh_nodes;
  long   numCells           = m_num_mesh_faces;

  time_t startTicks        = clock();
 
  int   domainID           = 0;
  int   timeState          = 0;
  std::string material("all");
  std::string cacheMeshID(mesh_name);
  cacheMeshID             += m_data_file;  
  debug1<<" try to find "<<cacheMeshID<<" in cache\n";
  vtkObject * cachedMesh=NULL;
  cachedMesh   = (a_avtFile->get_cache())->GetVTKObject(cacheMeshID.c_str(),
                                                 avtVariableCache::DATASET_NAME,
                                                 a_timeState, 
                                                 domainID, 
                                                 material.c_str());


 
  if((cachedMesh!=NULL)&&(m_mesh_is_static))
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
  debug1 << mesh_name << " not in cache/or not static. Load from data." << endl;
  
  // get face nodes
  int    numNodesPerFace        = NODESPERELE;
  long *  faceNodesPtr           = new long  [m_num_mesh_faces*(numNodesPerFace+1)];
 
  if (!m_external_mesh_provider->fillMeshElement(faceNodesPtr))
    {
      stringstream msgStream(stringstream::out);
      msgStream <<"Fail to retrieve faces nodes at step " <<a_timeState;
      EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
    }

  if(!m_mesh_is_static)
  {
	  if(m_external_mesh_provider->update_bottom_layer(a_timeState))
	  {
		  this->load_bottom(a_timeState);
	  }
	  else if (cachedMesh!=NULL)
	  {
		   vtkUnstructuredGrid *uGrid = (vtkUnstructuredGrid *)cachedMesh;
           uGrid->Register(NULL);
           updateMeshZCoordinates(uGrid->GetPoints(),
                                 a_timeState,
                                mesh_name);
           return uGrid;
	  }
  }

   
   long *  m2DPointto3DPoints = new long [m_num_mesh_nodes*m_num_layers];
 
   for(long i=0;i<m_num_mesh_nodes*m_num_layers;i++)
   {
	   m2DPointto3DPoints[i]= MeshConstants10::INVALID_NUM;
   }
  
   int Index = 0 ;
   for(int iLayer=0;iLayer<m_num_layers;iLayer++)
  {
	  for(long iNode=0;iNode<m_num_mesh_nodes;iNode++)
	  {
		  int bottomLayer = m_kbp_node[iNode];
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
  else if (!strcmp(mesh_name,m_mesh_2d_no_wet_dry.c_str()))
    {	 
		create2DUnstructuredMeshNoDryWet(uGrid,faceNodesPtr,a_timeState);  
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


 void   avtSCHISMFileFormatImpl10::insertPyramid(vtkUnstructuredGrid  *     a_uGrid,
								           long                  *     a_validTopNode,
									       long                  *     a_validBottomNode,
										   long                  *     a_faceNodePtr,
										   long                  *     a_2DPointto3DPoints,
								           const long            &     a_Cell,
	                                       const int            &     a_layerID)
 {
	 vtkIdType verts[5];
	//debug1<<"pyramid by triangle cell at layer "<<a_layerID<<", bottom ";
			
	long p1 = a_validBottomNode[0];
	long p2 = a_validBottomNode[1];
	//debug1<<p1<<" "<<p2<<" ";
	verts[0] = a_2DPointto3DPoints[p1*m_num_layers+a_layerID];
	verts[1] = a_2DPointto3DPoints[p2*m_num_layers+a_layerID];
	verts[2] = a_2DPointto3DPoints[p2*m_num_layers+a_layerID+1];
	verts[3] = a_2DPointto3DPoints[p1*m_num_layers+a_layerID+1];
	long p3=MeshConstants10::INVALID_NUM;
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
	if (p3==MeshConstants10::INVALID_NUM)
	{
			stringstream msgStream(stringstream::out);
            msgStream <<"fail to get pyramid apex for cell " <<a_Cell<<"on layer "<<a_layerID;
			EXCEPTION1(InvalidVariableException,msgStream.str());
	}
	int valid_bottom = std::max(1,m_kbp_node[p3])-1;
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


 void   avtSCHISMFileFormatImpl10::insertWedge(vtkUnstructuredGrid  *     a_uGrid,
								         long                  *     a_validTopNode,
									     long                  *     a_validBottomNode,
										 long                  *     a_2DPointto3DPoints,
								         const long            &     a_Cell,
	                                     const int            &     a_layerID)
 {
	 vtkIdType verts[2*3];
    //first add bottom face node
	for(int iNode=0;iNode< 3; ++iNode)
	{
	   long p = a_validBottomNode[iNode];
	   verts[iNode] = a_2DPointto3DPoints[p*m_num_layers+a_layerID];
	} 
	//then add top face node
	for(int iNode=3;iNode< 2*3; ++iNode)
	{
	   long p = a_validTopNode[iNode-3];
	   verts[iNode] = a_2DPointto3DPoints[p*m_num_layers+a_layerID+1];
	} 
	a_uGrid->InsertNextCell(VTK_WEDGE,2*3, verts);

 }


  void   avtSCHISMFileFormatImpl10::insertTetra(vtkUnstructuredGrid  *     a_uGrid,
								           long                 *     a_validTopNode,
									       long                 *     a_validBottomNode,
										   long                  *     a_faceNodePtr,
										   long                 *     a_2DPointto3DPoints,
								           const long           &     a_Cell,
	                                       const int           &     a_layerID)
 {
	vtkIdType verts[4];
	long p4 = a_validBottomNode[0];
			

	for(int i=0;i<3;i++)
	{
		//int p = a_validTopNode[i];
		long p   = a_faceNodePtr[i+1]-1;
		int valid_bottom = std::max(1,m_kbp_node[p])-1;
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


 void   avtSCHISMFileFormatImpl10::insertTriangle3DCell(vtkUnstructuredGrid * a_uGrid,
	                                              const int           & a_validTopNodeNum,
									              const int           & a_validBottomNodeNum,
									              long                 * a_validTopNode,
											      long                 * a_validBottomNode,
											      long                 * a_faceNodePtr,
												  long                 * a_2DPointto3DPoints,
											      const long           & a_Cell,
	                                              const int           & a_layerID)
 {
	    if ((a_validBottomNodeNum==0)||((a_validBottomNodeNum==2)&&(a_validTopNodeNum==2))) // no 3d cell at all
		{
			return;
		}
	 	

		if (a_validBottomNodeNum ==3) // this is a wedge 
		{
			vtkIdType verts[2*3];
			//first add bottom face node
			for(int iNode=0;iNode< 3; ++iNode)
			{
			  long p = a_validBottomNode[iNode];
			  verts[iNode] = a_2DPointto3DPoints[p*m_num_layers+a_layerID];
			} 
			//then add top face node
			for(int iNode=3;iNode< 2*3; ++iNode)
			{
			  long p = a_validTopNode[iNode-3];
			  verts[iNode] = a_2DPointto3DPoints[p*m_num_layers+a_layerID+1];
			} 
			a_uGrid->InsertNextCell(VTK_WEDGE,2*3, verts);
			m_tri_wedge++;
		}
		else if(a_validBottomNodeNum ==2)// bottom have two node, this is a pyramid
		{
			vtkIdType verts[5];
			
			long p1 = a_validBottomNode[0];
			long p2 = a_validBottomNode[1];
			
			verts[0] = a_2DPointto3DPoints[p1*m_num_layers+a_layerID];
			verts[1] = a_2DPointto3DPoints[p2*m_num_layers+a_layerID];
			verts[2] = a_2DPointto3DPoints[p2*m_num_layers+a_layerID+1];
			verts[3] = a_2DPointto3DPoints[p1*m_num_layers+a_layerID+1];
			long p3=MeshConstants10::INVALID_NUM;
			
			for(int i=0;i<3;i++)
			{
				if(((a_faceNodePtr[i+1]-1)!=p1)&&((a_faceNodePtr[i+1]-1)!=p2))
				{
					p3 = a_faceNodePtr[i+1]-1;
				}
			}

			
			if (p3==MeshConstants10::INVALID_NUM)
			{
				 stringstream msgStream(stringstream::out);
                 msgStream <<"fail to get pyramid apex for cell " <<a_Cell<<"on layer "<<a_layerID;
			     EXCEPTION1(InvalidVariableException,msgStream.str());
			}
			int valid_bottom = std::max(1,m_kbp_node[p3])-1;
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
			long p4 = a_validBottomNode[0];
			

			for(int i=0;i<3;i++)
			{
			
				long p   = a_faceNodePtr[i+1]-1;
				int valid_bottom = std::max(1,m_kbp_node[p])-1;
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

 bool     avtSCHISMFileFormatImpl10::fourPointsCoplanar(double p1[3],
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

  void     avtSCHISMFileFormatImpl10::insert8NodesPolyhedron(vtkUnstructuredGrid  *     a_uGrid,
	                                                   vtkIdType            *     a_verts,
								                       long                 *     a_validTopNode,
									                   long                 *     a_validBottomNode,
													   long                 *     a_2DPointto3DPoints,
								                       const long           &     a_Cell,
	                                                   const int            &     a_layerID,
										               const bool           &     a_bottomCoplane,
										               const bool           &     a_topCoplane)
  {
	  				debug1 <<"8 nodes polyhedron at layer " <<a_layerID<<" cell "
						<<a_Cell<<" "<<a_bottomCoplane<<" "<<a_topCoplane<<"\n";


			  vtkIdType p_t[4];
	
	          for(int inode=0;inode<4;inode++)
		      {
			     long v_t = a_validTopNode[inode];
			
			     // layer in m_kbp_node starts from 1, a_layerID starts from 0
                 int bottomLayer = m_kbp_node[v_t];
			   
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
					long p1 = a_validTopNode[iNode];
					int iNode2 = iNode+1;
					if(iNode2>3)
					{
						iNode2=0;
					}
					long p2 = a_validTopNode[iNode2];
					
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
			
						long p1 = a_validBottomNode[iNode];
						faceBottom1[iNode]= a_2DPointto3DPoints[p1*m_num_layers+a_layerID];
						debug1<<" "<<faceBottom1[iNode];
					}
					vtkIdType faceBottom2[3];

					for(int iNode=2;iNode<4;iNode++)
					{
			
						long p1 = a_validBottomNode[iNode];
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
			
						long p1 = a_validBottomNode[iNode];
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

   void    avtSCHISMFileFormatImpl10::insert7NodesPolyhedron(vtkUnstructuredGrid  *     a_uGrid,
	                                                   vtkIdType            *     a_verts,
								                       long                 *     a_validTopNode,
									                   long                 *     a_validBottomNode,
													   long                 *     a_2DPointto3DPoints,
								                       const long           &     a_Cell,
	                                                   const int            &     a_layerID,
										               const bool           &     a_topCoplane)
   {
	   		
              
            // find out degenerated point
			long degeneratedNode = -9999;
			long degeneratedNodeLoc = -9999;
			for(int iNode=0;iNode<4;++iNode)
			{
				long p1 = a_validTopNode[iNode];
				bool found_in_bottom = false;
				for(int j=0;j<3;++j)
				{
					long p2 = a_validBottomNode[j];
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
			   long v_t = a_validTopNode[inode];
			
			   // layer in m_kbp_node starts from 1, a_layerID starts from 0
               int bottomLayer = m_kbp_node[v_t];
			   
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
                long p1 = a_validTopNode[iNode];
				long iNode2 = iNode+1;
				if(iNode2>3)
				{
					iNode2=0;
				}
				long p2 = a_validTopNode[iNode2];
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
			
				long p1 = a_validBottomNode[iNode];
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

  void    avtSCHISMFileFormatImpl10::insertQuad3DCell(vtkUnstructuredGrid *  a_uGrid,
	                                            const int           & a_validTopNodeNum,
									            const int           & a_validBottomNodeNum,
									            long                 * a_validTopNode,
											    long                 * a_validBottomNode,
											    long                 * a_faceNodePtr,
												long                 * a_2DPointto3DPoints,
												const long           & a_Cell,
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
			long v_t = a_validTopNode[inode];
			
			// layer in m_kbp_node starts from 1, a_layerID starts from 0
            int bottomLayer = m_kbp_node[v_t];
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
			long v1 = a_validBottomNode[0];
			long v2 = a_validBottomNode[1];
			long v3 = a_validBottomNode[2];
			long v4 = a_validBottomNode[3];
			
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
				long p = a_validBottomNode[iNode];
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
			  long p = a_validBottomNode[iNode];
			 // debug1<<" "<<p;
			  verts[iNode] = a_2DPointto3DPoints[p*m_num_layers+a_layerID];
			} 
			verts[3]=p_t[3];
			//then add top face node
			for(int iNode=4;iNode< 8; ++iNode)
			{
			  long p = a_validTopNode[iNode-4];
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

			long p1 = a_validBottomNode[0];
			long p2 = a_validBottomNode[1];
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

			int valid_bottom = std::max(1,m_kbp_node[p3])-1;
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

			valid_bottom = std::max(1,m_kbp_node[p3])-1;
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
			//debug1 <<"pyramid at layer " <<a_layerID<<" cell "
			//		<<a_Cell<<" "<<topCoplane<<"\n";;
			vtkIdType verts[5];
		   
			for (int iNode=0;iNode<4;iNode++)
			{
			   verts[iNode]= p_t[iNode];
			}
			long p3 = a_validBottomNode[0];
			verts[4] = a_2DPointto3DPoints[p3*m_num_layers+a_layerID];
			//debug1<<" bottom: "<<p3<<"\n";
			a_uGrid->InsertNextCell(VTK_PYRAMID,5, verts);
			m_quad_pyramid++;
		}

  }

void    avtSCHISMFileFormatImpl10::validTopBottomNode(int       &   a_validTopNodeNum,
	                                            int       &   a_validBottomNodeNum,
											    long      *   a_validTopNode,
											    long      *   a_validBottomNode,
												const int &   a_layerID,
									            long      *   a_faceNodePtr) const

{

	 int numberOfNodeInCell = *a_faceNodePtr;

			 
	//get node id indexed in 2d mesh
	long * node2D = new long [numberOfNodeInCell];
	
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
		// layer in m_kbp_node starts from 1, a_layerID starts from 0
		int bottomLayer = m_kbp_node[node2D[iNode]];
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
	long validTopNode = 0;
	for(int iNode=0;iNode<numberOfNodeInCell; iNode++)
	{
	   // layer in m_kbp_node starts from 1, a_layerID starts from 0
       //int bottomLayer = m_kbp_node[node2D[iNode]];
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
//  Method: avtSCHISMFileFormatImpl10:: updateMeshZCoordinates
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
void  avtSCHISMFileFormatImpl10::updateMeshZCoordinates(vtkPoints * a_pointSet,
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

void avtSCHISMFileFormatImpl10::loadAndCacheZSide(const int& a_timeState, float * a_sideCenterZPtr)
{
	 if (!m_external_mesh_provider->zSideCenter3D(a_sideCenterZPtr,a_timeState))
	 {
		  stringstream msgStream(stringstream::out);
          msgStream <<"Fail to retrieve side z at step " <<a_timeState;
          EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
	 }
}

void avtSCHISMFileFormatImpl10::loadAndCacheZEle(const int& a_timeState,float * a_eleCenterZPtr)
{
	 if (!m_external_mesh_provider->zEleCenter3D(a_eleCenterZPtr,a_timeState))
	 {
		  stringstream msgStream(stringstream::out);
          msgStream <<"Fail to retrieve ele z at step " <<a_timeState;
          EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
	 }
	
}

void avtSCHISMFileFormatImpl10::loadAndCacheZ(const int& a_timeState,float* a_nodeZPtr)
{
	 if (!m_external_mesh_provider->zcoords3D(a_nodeZPtr,a_timeState))
	 {
		  stringstream msgStream(stringstream::out);
          msgStream <<"Fail to retrieve node z at step " <<a_timeState;
         EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
	 }
}


void avtSCHISMFileFormatImpl10::load_node_dry_wet(const int & a_time)
{
 
  long num=m_num_mesh_nodes;

  if (m_node_dry_wet_cached_time==a_time)
  {
     return;
  }
  else
  {
	if (!(m_node_dry_wet))
	{
		m_node_dry_wet= new int [num];
	}

 
   m_external_mesh_provider->fill_node_dry_wet(m_node_dry_wet,m_ele_dry_wet);

   long nominal_data_size_per_layer=0;

   
   for(int i=0;i<num;i++)
   {
	   if(!(m_node_dry_wet[i]))
	   {
		   nominal_data_size_per_layer++;
	   }
   }
  
   m_nominal_size_per_layer[MeshConstants10::NODE]=nominal_data_size_per_layer;
   m_node_dry_wet_cached_time=a_time;
  }

  return;
}


void avtSCHISMFileFormatImpl10::load_ele_dry_wet(const int & a_time)
{
   long num=m_num_mesh_faces;

  if (m_ele_dry_wet_cached_time==a_time)
  {
     return;
  }
  else
  {
	if (!(m_ele_dry_wet))
	{
		m_ele_dry_wet= new int [num];
	}

	try
	{
		m_external_mesh_provider->fill_ele_dry_wet(m_ele_dry_wet, a_time);
    }
	catch (...)
	{
		stringstream msgStream(stringstream::out);
		msgStream << "Fail to retrieve ele dry/wet flag at step " << a_time;
		EXCEPTION3(DBYieldedNoDataException, m_data_file, m_plugin_name, msgStream.str());
	}

 /* std::string SCHISMVarName = MeshConstants10::ELEM_DRYWET;
  SCHISMVar10* SCHISMVarPtr=NULL;
  try
  {
     SCHISMVarPtr = m_data_file_ptr->get_var(SCHISMVarName);
  }
  catch(...)
  {
	  EXCEPTION1(InvalidVariableException, ("no "+SCHISMVarName+" in "+m_data_file_ptr->file()).c_str());
  }

   SCHISMVarPtr->set_cur(a_time);
   if (!(SCHISMVarPtr->get(m_ele_dry_wet)))
   {
	   stringstream msgStream(stringstream::out);
       msgStream <<"Fail to retrieve "<<SCHISMVarName << " at step " <<a_time;
       EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
   }*/
    long nominal_data_size_per_layer=0;

   for(int i=0;i<num;i++)
   {
	   if(!(m_ele_dry_wet[i]))
	   {
		   nominal_data_size_per_layer++;
	   }
   }

   m_nominal_size_per_layer[MeshConstants10::ELEM]=nominal_data_size_per_layer;
    m_ele_dry_wet_cached_time=a_time;
  }

  return;
}


void avtSCHISMFileFormatImpl10::load_side_dry_wet(const int & a_time)
{
  long num=m_num_mesh_edges;

  if (m_side_dry_wet_cached_time==a_time)
  {
     return;
  }
  else
  {
	if (!(m_side_dry_wet))
	{
		m_side_dry_wet= new int [num];
	}

  
  m_external_mesh_provider->fill_side_dry_wet(m_side_dry_wet,m_ele_dry_wet);

    long nominal_data_size_per_layer=0;

   for(int i=0;i<num;i++)
   {
	   if(!(m_side_dry_wet[i]))
	   {
		   nominal_data_size_per_layer++;
	   }
   }

   m_nominal_size_per_layer[MeshConstants10::EDGE]=nominal_data_size_per_layer;

   m_side_dry_wet_cached_time=a_time;
  }

  return;
}
void   avtSCHISMFileFormatImpl10::getMeshDimensions(MeshProvider10 * a_meshProviderPtr)
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
  m_nominal_size_per_layer[MeshConstants10::NODE]= m_num_mesh_nodes;
  m_nominal_size_per_layer[MeshConstants10::ELEM]= m_num_mesh_faces;
  m_nominal_size_per_layer[MeshConstants10::EDGE]= m_num_mesh_edges;

 m_num_layers= a_meshProviderPtr->numberOfLayer();
  
  if (m_num_layers<0)
    {
      EXCEPTION1(InvalidVariableException, ("no layer found ,one required at least, in "+a_meshProviderPtr->file()).c_str());
    }
   debug1<<"get layers num"<<m_num_layers<<"\n";
}

void    avtSCHISMFileFormatImpl10::load_bottom(const int& a_time)
{

  if ( m_cache_kbp_id==a_time)
  {
	  return;
  }
  if(!m_kbp_node)
  {
     m_kbp_node = new int [m_num_mesh_nodes];
  }
  m_external_mesh_provider->fillKbp00(m_kbp_node,a_time);
  
  //count total valid 3d point
  m_total_valid_3D_point = 0;
  for(int iLayer=0;iLayer<m_num_layers;iLayer++)
  {
	  for(int iNode=0;iNode<m_num_mesh_nodes;iNode++)
	  {
		  int bottomLayer = m_kbp_node[iNode];
		  if (bottomLayer<=(iLayer+1))
			  {
				  m_total_valid_3D_point++;
			  }

	  }
  }

  // int * kbs=0;
  if(!m_kbp_side)
  {
	m_kbp_side = new int [m_num_mesh_edges];
	m_external_mesh_provider->fillKbs(m_kbp_side,a_time);
  }
 

  
  //count total valid 3d point
  m_total_valid_3D_side = 0;
  for(int iLayer=0;iLayer<m_num_layers;iLayer++)
  {
	  for(int iEdge=0;iEdge<m_num_mesh_edges;iEdge++)
	  {
		  int bottomLayer = m_kbp_side[iEdge];
		  if (bottomLayer<=(iLayer+1))
			  {
				  m_total_valid_3D_side++;
			  }

	  }

  }
 
  

  if(!m_kbp_ele)
  {
	  m_kbp_ele = new int [m_num_mesh_faces];
	  m_external_mesh_provider->fillKbe(m_kbp_ele,a_time);
  }
  
  //count total valid 3d ele
  m_total_valid_3D_ele = 0;
  for(int iLayer=0;iLayer<m_num_layers;iLayer++)
  {
	  for(int iEle=0;iEle<m_num_mesh_faces;iEle++)
	  {
		  int bottomLayer =  m_kbp_ele[iEle];
		  if (bottomLayer<=(iLayer+1))
		 {
			 m_total_valid_3D_ele++;
	     }
	  }
  }
 
 m_cache_kbp_id=a_time;
 
 
}


void    avtSCHISMFileFormatImpl10::loadMeshCoordinates(MeshProvider10 * a_meshProviderPtr)
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

 }
// ****************************************************************************
//  Method: avtSCHISMFileFormatImpl10::getLayer
//
//  Purpose:
//      Return layer id for each 3d prsim
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

 vtkDataArray*   avtSCHISMFileFormatImpl10::getLayer()
 {
  int * kbp00=m_kbp_ele;
  std::string data_center = MeshConstants10::ELEM;

  int valid_var_size =0;
  int num_total_layers    =m_num_layers-1; 

  long nominal_num_data_per_Layer=m_num_mesh_faces;

  for(long iNode=0;iNode<nominal_num_data_per_Layer;iNode++)
  {
	  //if(!(m_ele_dry_wet[iNode]))
	  {
	   valid_var_size+=num_total_layers-std::max(1,kbp00[iNode])+1;
	  }
  }

  long ntuples        = valid_var_size; 
  vtkIntArray *rv = vtkIntArray::New();
  rv->SetNumberOfTuples(ntuples);
  long idata = 0;    


   for (int iLayer   = 0 ; iLayer < num_total_layers ; iLayer++)
   {
     for( long iEle = 0 ; iEle   < nominal_num_data_per_Layer; iEle++)
      { 
		int valid_bottom_layer = std::max(1,kbp00[iEle])-1;
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
//  Method: avtSCHISMFileFormatImpl10::getLayer
//
//  Purpose:
//      Return layer id for 3d node, ele or side according to name
//      
//     
//
//  Arguments:
//       
//                  
//      level id in string
//
//  Programmer: qshu -- generated by xml2avt
//  Creation:   Mon DEC 13 09:57:49 PDT 2015
//
// ****************************************************************************

vtkDataArray*   avtSCHISMFileFormatImpl10::getLevel(const string& a_level_name)
 {
  int * kbp00;
  std::string data_center(MeshConstants10::NODE);
   long nominal_num_data_per_Layer=0;
  if (!(a_level_name.compare(MeshConstants10::NODE_LEVEL)))
  {
	  //kbp00=m_kbp_data;
	  kbp00=m_kbp_node;
	  nominal_num_data_per_Layer=m_num_mesh_nodes;
  }
  else if (!(a_level_name.compare(MeshConstants10::ELE_LEVEL)))
  {
	  kbp00=m_kbp_ele;
	  data_center = MeshConstants10::ELEM;
	   nominal_num_data_per_Layer=m_num_mesh_faces;
  }
  else if (!(a_level_name.compare(MeshConstants10::SIDE_LEVEL)))
  {
	  kbp00=m_kbp_side;
	  data_center = MeshConstants10::EDGE;
	   nominal_num_data_per_Layer=m_num_mesh_edges;
  }
  else
  {
	   stringstream msgStream(stringstream::out);
       msgStream <<a_level_name<<" is tot a valid mesh level variable\n";
       EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
  }

  int valid_var_size =0;
  int num_total_layers    =m_num_layers; 

  for(long iNode=0;iNode<nominal_num_data_per_Layer;iNode++)
  {
	   
	  valid_var_size+=num_total_layers-std::max(1,kbp00[iNode])+1;
  }

  long ntuples        = valid_var_size; 
  vtkIntArray *rv = vtkIntArray::New();
  rv->SetNumberOfTuples(ntuples);
  long idata = 0;    


   for (int iLayer   = 0 ; iLayer < num_total_layers ; iLayer++)
   {
     for( long iNode = 0 ; iNode   < nominal_num_data_per_Layer; iNode++)
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




vtkDataArray*   avtSCHISMFileFormatImpl10::getBottom(const string& a_bottom_name)
 {
  int * kbp00;
  std::string data_center(MeshConstants10::NODE);
   long nominal_num_data_per_Layer=0;
  if (!(a_bottom_name.compare(MeshConstants10::NODE_BOTTOM)))
  {
	  //kbp00=m_kbp_data;
	  kbp00=m_kbp_node;
	   nominal_num_data_per_Layer=m_num_mesh_nodes;
  }
  else if (!(a_bottom_name.compare(MeshConstants10::FACE_BOTTOM)))
  {
	  kbp00=m_kbp_ele;
	  data_center = MeshConstants10::ELEM;
	   nominal_num_data_per_Layer=m_num_mesh_faces;
  }
  else if (!(a_bottom_name.compare(MeshConstants10::EDGE_BOTTOM)))
  {
	  kbp00=m_kbp_side;
	  data_center = MeshConstants10::EDGE;
	   nominal_num_data_per_Layer=m_num_mesh_edges;
  }
  else
  {
	   stringstream msgStream(stringstream::out);
       msgStream <<a_bottom_name<<" is not a valid mesh bottom variable\n";
       EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
  }

 
  //long nominal_num_data_per_Layer=m_nominal_size_per_layer[data_center];


  long ntuples        = nominal_num_data_per_Layer; 
  vtkIntArray *rv = vtkIntArray::New();
  rv->SetNumberOfTuples(ntuples);
  long idata = 0;    


  
	for( long iNode = 0 ; iNode   < nominal_num_data_per_Layer; iNode++)
	{  
	
		rv->SetTuple1(idata, kbp00[iNode]);  
		idata++;    

	}
  

   return rv;
 }

// ****************************************************************************
//  Method: avtSCHISMFileFormatImpl10::GetVar
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
avtSCHISMFileFormatImpl10::GetVar(int a_timeState, const char *a_varName)
{

     //a_varName is the label used by visit. This name
  //needs to be map into varname used in data file.

   if ( !strcmp(a_varName,( MeshConstants10::LAYER).c_str()))
   {
	   return getLayer();
   }
   else  if ( !strcmp(a_varName,( MeshConstants10::NODE_LEVEL).c_str()))
   {
	   load_node_dry_wet(a_timeState);
	   return getLevel(MeshConstants10::NODE_LEVEL);
   }
   else  if ( !strcmp(a_varName,( MeshConstants10::ELE_LEVEL).c_str()))
   {
	   load_ele_dry_wet(a_timeState);
	   return getLevel(MeshConstants10::ELE_LEVEL);
   }
   else  if ( !strcmp(a_varName,( MeshConstants10::SIDE_LEVEL).c_str()))
   {
	   load_side_dry_wet(a_timeState);
	   return getLevel(MeshConstants10::SIDE_LEVEL);
   }
   else  if ( !strcmp(a_varName,( MeshConstants10::NODE_BOTTOM).c_str()))
   {
	   return getBottom(MeshConstants10::NODE_BOTTOM);
   }
      else  if ( !strcmp(a_varName,( MeshConstants10::FACE_BOTTOM).c_str()))
   {
	   return getBottom(MeshConstants10::FACE_BOTTOM);
   }
	   else  if ( !strcmp(a_varName,( MeshConstants10::EDGE_BOTTOM).c_str()))
   {
	   return getBottom(MeshConstants10::EDGE_BOTTOM);
   }

  
  std::string SCHISMVarName = m_var_name_label_map[a_varName];  
  std::string varMesh      = m_var_mesh_map[a_varName];

  SCHISMVar10 * SCHISMVarPtr = NULL;
  
 
  //if (m_scribeIO && ((!strcmp(a_varName, MeshConstants10::NODE_SURFACE_LABEL.c_str())) || (!strcmp(a_varName, m_node_depth_label.c_str()))))
  //{
	 //
	 // SCHISMVarPtr = m_external_mesh_provider->get_mesh_data_ptr()->get_var(SCHISMVarName);
  //}
  //else
  //{
	  SCHISMVarPtr= m_data_file_ptr->get_var(SCHISMVarName);
  //}
  
  std::string level_center = SCHISMVarPtr->get_vertical_center();
  std::string data_center =SCHISMVarPtr->get_horizontal_center();
  
  int * drywet;
  debug1<<"getting "<<a_varName<<" "<<level_center<<" "<<data_center <<"\n";

  long numDataPerLayer=m_num_mesh_nodes;
  if (data_center ==MeshConstants10::ELEM)
  {
	  load_ele_dry_wet(a_timeState);
	  numDataPerLayer=m_num_mesh_faces;
	  drywet=m_ele_dry_wet;
  }
  else if (data_center ==MeshConstants10::EDGE)
  {
	  load_side_dry_wet(a_timeState);
	  numDataPerLayer=m_num_mesh_edges;
	  drywet=m_side_dry_wet;
  }
  else
  {
	  debug1<<"try to load node dry wet\n";
	  load_node_dry_wet(a_timeState);
	  drywet=m_node_dry_wet;
  }


  long record_size_adjust=0;
  int layer_size_adjust=0;
  bool is_half_layer=false;

  if(level_center==MeshConstants10::HALF_LAYER)
  {
	  record_size_adjust=-1;
	  layer_size_adjust=-1;
	  is_half_layer=true;
  }

  if (!(SCHISMVarPtr->is_valid()))
    {
      
      EXCEPTION1(InvalidVariableException, a_varName);
    }
  

  

    if ( (!strcmp(a_varName,m_node_depth_label.c_str()))   || (!strcmp(a_varName, MeshConstants10::NODE_SURFACE_LABEL.c_str())) ||
		 (!SCHISMVarIs3D(SCHISMVarPtr)))
    {
     float * valBuff;
	 int numData=numDataPerLayer;
	
     valBuff          = new float  [numData]; 


		 getSingleLayerVar(valBuff,
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
		   //if((!(drywet[iNode]))||(!strcmp(a_varName,m_node_depth_label.c_str())) || (!strcmp(a_varName, MeshConstants10::NODE_BOTTOM2.c_str())))
		   //{
             
		     rv->SetTuple1(idata, valTemp); 
		   //}
		   //else
		   //{
			// if(m_dry_wet_flag)
			//    rv->SetTuple1(idata, MeshConstants10::DRY_STATE); 
			// else
			//	rv->SetTuple1(idata, valTemp);  
		   //}
         idata++;             
       }
     delete   valBuff;
     return rv;
    }


 
  debug1<<"get SCHISM var "<<SCHISMVarName<<"\n";

  int num_data_layers    =m_num_layers+layer_size_adjust;

  std::string        varName(a_varName);
  int * layerStarts;
  bool is_bottom=false;
  bool is_surface =false;
  if (varName.find(m_surface_state_suffix) != string::npos) 
    {
      num_data_layers      = 1;
      layerStarts    = new int [num_data_layers];
      layerStarts[0] =m_num_layers-1+layer_size_adjust;
	  is_surface=true;
    }
  else if (varName.find(m_bottom_state_suffix) != string::npos)
    {
      num_data_layers      = 1;
      layerStarts    = new int [num_data_layers];
	  //this is the layer next to the bottom assumed,it will change
	  //for each node later
      layerStarts[0] = 1;
	  is_bottom=true;
    } 
  else
    {
      layerStarts   = new int [num_data_layers];
      for(int iLayers=0;iLayers<num_data_layers;iLayers++)
        {
          layerStarts[iLayers] = iLayers;
        }
     }
  

  float * valBuff;
  int numTotalLayers    =m_num_layers+layer_size_adjust; 
  valBuff = new float  [numTotalLayers*numDataPerLayer];

   
  int numOfRecord  = 1;
  int nodeStart    = 0;
  int timeStart    = a_timeState;
  debug1<<"before schism var "<<SCHISMVarName<<" set current\n";
  SCHISMVarPtr->set_cur(timeStart);
  
  debug1<<"schism var "<<SCHISMVarName<<" set current\n";
  if (!m_scribeIO)
  {
	  if (!(SCHISMVarPtr->get(valBuff)))
	  {
		  stringstream msgStream(stringstream::out);
		  msgStream << "Fail to retrieve " << a_varName << " at step " << a_timeState;
		  EXCEPTION3(DBYieldedNoDataException, m_data_file, m_plugin_name, msgStream.str());
	  }
  }
  else
  {
	  int * bottom = new int[numDataPerLayer];

	  if (data_center == MeshConstants10::ELEM)
	  {
		  m_external_mesh_provider->fillKbe(bottom, timeStart);
	  }
	  else if (data_center == MeshConstants10::EDGE)
	  {
		  m_external_mesh_provider->fillKbs(bottom, timeStart);
	  }
	  else
	  {
		  m_external_mesh_provider->fillKbp00(bottom, timeStart);
	  }

	  if (!(SCHISMVarPtr->get(valBuff,bottom)))
	  {
		  stringstream msgStream(stringstream::out);
		  msgStream << "Fail to retrieve " << a_varName << " at step " << a_timeState;
		  EXCEPTION3(DBYieldedNoDataException, m_data_file, m_plugin_name, msgStream.str());
	  }
	  delete bottom;
  }
  debug1<<"schism var "<<SCHISMVarName<<" data loaded\n";  
  long valid_var_size =0;
  long * node_start_index  = new long [numDataPerLayer];
  long * num_data_at_layer = new long [numTotalLayers];

  for(int iLayer=0;iLayer<numTotalLayers;iLayer++)
  {
	  num_data_at_layer[iLayer]=0;
  }

  int * kbp00=m_kbp_node;

  if (!(data_center.compare(MeshConstants10::EDGE)))
  {
	  kbp00=m_kbp_side;
  }
  else if (!(data_center.compare(MeshConstants10::ELEM)))
  {
	  kbp00=m_kbp_ele;
  }


   int num_layers =m_num_layers+layer_size_adjust;

  long half_remove_cell =0;
  for(int iNode=0;iNode<numDataPerLayer;iNode++)
  {

	 node_start_index[iNode]=valid_var_size;
	 //if (!(drywet[iNode]))
	 {
		 long record_len =num_layers-std::max(1,kbp00[iNode])+1;
		 //if((drywet[iNode])&&(record_len>0)) half_remove_cell+=record_len;
		 //if(record_len<0) record_len=0; // only happened for dry element
		 valid_var_size+=record_len;

	  }
  }
 
  
   vtkFloatArray *rv = vtkFloatArray::New(); 
   int idata = 0;
   if (varName.find(m_depth_average_suffix) == string::npos)
     {
       idata = 0;
      
	   // count total number of data 
	   int ntuples       = 0;
      
	   ntuples=valid_var_size;
       if (is_bottom||is_surface) //for surface and bottom data, same as number of 2D node
	   {
	      ntuples=numDataPerLayer; 
	   }
	   if(is_half_layer) ntuples-=half_remove_cell;
	   rv->SetNumberOfTuples(ntuples); 
	   debug1<<" count total num of data "<<ntuples<<"\n ";
       for (int iLayer   = 0 ; iLayer < num_data_layers ; iLayer++)
       {
          int layer = layerStarts[iLayer];
          for( int iNode = 0 ; iNode   < numDataPerLayer; iNode++)
          {  
			   int valid_bottom_layer = std::max(1,kbp00[iNode])-1;
			   if(is_bottom)
			   {
				   layer=valid_bottom_layer+1;
				   if(layer>(numTotalLayers-1))
				   {
					   layer=numTotalLayers-1;
				   }
			   }
			  float valTemp =MeshConstants10::DUMMY_ELEVATION;
			  int start_index = node_start_index[iNode];
			 
			  if(layer>=valid_bottom_layer)
			  {
				  long data_loc = start_index+layer-valid_bottom_layer;
                  valTemp =  valBuff[data_loc]; 
			   if (!(drywet[iNode])) 
				{

			      rv->SetTuple1(idata, valTemp); 
			    }
			   else
			    {
				  if(m_dry_wet_flag)
				  {
				     rv->SetTuple1(idata, MeshConstants10::DRY_STATE); 
				  }
				  else
				  {
					 rv->SetTuple1(idata, valTemp); 
				  }
			    }
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
       depthAverage(averageState,valBuff,node_start_index,a_timeState,data_center,level_center);
	   debug1<<"total num of averaged data "<<numDataPerLayer<<"\n";
       idata  = 0;
       for( int iNode = 0 ; iNode   < numDataPerLayer; iNode++)
         {
		   //if (!(drywet[iNode]))
		   if((!drywet[iNode])|| (drywet[iNode]&&(!(m_dry_wet_flag))))
		   {
             float valTemp   = averageState[iNode];
             rv->SetTuple1(idata, valTemp);  
		   }
		   else
		   {
             rv->SetTuple1(idata, MeshConstants10::DRY_STATE); 
		   }

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

vtkDataArray  * avtSCHISMFileFormatImpl10::GetVector2d(int          a_timeState,std::string a_varName)
{
	std::string xcom, ycom;
	size_t      found = a_varName.find_last_of("_");
	if (found != std::string::npos)
	{
		xcom = a_varName.substr(0, found) + "X" + a_varName.substr(found);
		ycom = a_varName.substr(0, found) + "Y" + a_varName.substr(found);
	}
	else
	{
		xcom = a_varName + "X";
		ycom = a_varName + "Y";
	}
	SCHISMVar10 * SCHISMVarPtr = m_external_mesh_provider->get_mesh_data_ptr()->get_var(xcom);
	SCHISMVar10 * SCHISMVarPtr2 = m_external_mesh_provider->get_mesh_data_ptr()->get_var(ycom);
	long numDataPerLayer = m_num_mesh_nodes;

	int      numDim = SCHISMVarPtr->num_dims();
	SCHISMDim10* comDim = SCHISMVarPtr->get_dim(numDim - 1);
	int ncomps = 2;
	int ucomps = 3;


	float *oneEntry = new float[ucomps];
	int idata = 0;

	
	float * valBuff;
	long numData = numDataPerLayer;

	long numDataEntry = numData;
	if (!SCHISMVarPtr2)
	{
		numData *= ncomps;
	}

	valBuff = new float[numData];
	getSingleLayerVar(valBuff,
		m_external_mesh_provider->get_mesh_data_ptr(),
		a_timeState,
		xcom);
	float * valBuff2 = NULL;

	valBuff2 = new float[numData];
		getSingleLayerVar(valBuff2,
			m_external_mesh_provider->get_mesh_data_ptr(),
			a_timeState,
			ycom);


	float * valBuffAll = valBuff;

		valBuffAll = new float[ncomps * numData];
		for (long iNode = 0; iNode < numData; iNode++)
		{
			valBuffAll[ncomps*iNode] = valBuff[iNode];
			valBuffAll[ncomps * iNode + 1] = valBuff2[iNode];
		}


	//total number of data = nodes for a time step
	long ntuples = numDataEntry;
	vtkFloatArray *rv = vtkFloatArray::New();
	rv->SetNumberOfComponents(ucomps);
	rv->SetNumberOfTuples(ntuples);
	idata = 0;
	for (long iNode = 0; iNode < ntuples; iNode++)
	{


			for (int iComp = 0; iComp < ncomps; iComp++)
			{
				oneEntry[iComp] = valBuffAll[iNode*ncomps + iComp];
			}
			for (int iComp = ncomps; iComp < ucomps; iComp++)
			{
				oneEntry[iComp] = 0.0;
			}
		


		rv->SetTuple(idata, oneEntry);
		idata++;
	}
	delete   valBuff;
    delete valBuff2;
	delete valBuffAll;
	
	return rv;
	
}


// ****************************************************************************
//  Method: avtSCHISMFileFormatImpl10::GetVectorVar
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
avtSCHISMFileFormatImpl10::GetVectorVar(int a_timeState, const char *a_varName)
{
	//decide if it is 2d scribe io format
	std::string varName(a_varName);
	std::map<std::string, std::string>::iterator itX;
	std::size_t found = varName.find('_');
	std::string x_com_name = "";
	if (found != std::string::npos)
	{
		x_com_name = varName.substr(0, found) +"X"+ varName.substr(found, varName.length());
	}
	else
	{
		x_com_name = varName + "X";

	}
	itX = Vector2DVarMapX.find(x_com_name);
	if (itX != Vector2DVarMapX.end())
	{
		return GetVector2d(a_timeState, varName);
	}


    std::string SCHISMVarName = m_var_name_label_map[a_varName];  
	
	std::vector<std::string> varNames;

	std::istringstream f(SCHISMVarName);
	std::string s;

	while (getline(f, s, ';'))
	{
		varNames.push_back(s);
	}

    std::string varMesh      = m_var_mesh_map[a_varName];

	
	SCHISMVar10 * SCHISMVarPtr = m_data_file_ptr->get_var(varNames[0]);
    if (!(SCHISMVarPtr->is_valid()))
    {
      
      EXCEPTION1(InvalidVariableException, a_varName);
    }
	
	SCHISMVar10 * SCHISMVarPtr2 = NULL;
	if (varNames.size() == 2)
	{
		if (m_data_file_ptr2)
		{
			SCHISMVarPtr2 = m_data_file_ptr2->get_var(varNames[1]);
	    }
		else
		{
			stringstream msgStream(stringstream::out);
			msgStream << " " << a_varName << " miss another component data source file \n";
			EXCEPTION1(InvalidFilesException, msgStream.str());
		}
	}

	std::string level_center = SCHISMVarPtr->get_vertical_center();
    std::string data_center =SCHISMVarPtr->get_horizontal_center();

	int * drywet;

  long numDataPerLayer=m_num_mesh_nodes;
  if (data_center ==MeshConstants10::ELEM)
  {
	  load_ele_dry_wet(a_timeState);
	  numDataPerLayer=m_num_mesh_faces;
	  drywet=m_ele_dry_wet;
  }
  else if (data_center ==MeshConstants10::EDGE)
  {
	  load_side_dry_wet(a_timeState);
	  numDataPerLayer=m_num_mesh_edges;
	  drywet=m_side_dry_wet;
  }
  else
  {
	  debug1<<"try to load node dry wet\n";
	  load_node_dry_wet(a_timeState);
	  drywet=m_node_dry_wet;
  }

    debug1 << "load vector dry wet\n";
    // last dim is vector component  
    
    int      numDim = SCHISMVarPtr->num_dims();       
    SCHISMDim10* comDim = SCHISMVarPtr->get_dim(numDim-1);
    int ncomps       = comDim->size();
	int ucomps       = (ncomps == 2 ? 3 : ncomps);

	if (SCHISMVarPtr2) //for scribe format
	{
		ucomps = 3;
		ncomps = 2;
	}

	float *oneEntry  = new float[ucomps];
	int idata=0;

	if (!SCHISMVarIs3D(SCHISMVarPtr))
    {
      float * valBuff;
	  long numData=numDataPerLayer;
	  
	  long numDataEntry=numData;
	  if (!SCHISMVarPtr2)
	  {
		  numData *= ncomps;
	  }

      valBuff          = new float  [numData]; 
      getSingleLayerVar (valBuff,
		                m_data_file_ptr,
                        a_timeState,
		                varNames[0]);
	  float * valBuff2 = NULL;

	  if (SCHISMVarPtr2)
	  {
		  valBuff2 = new float[numData];
		  getSingleLayerVar(valBuff2,
			                m_data_file_ptr2,
			                a_timeState,
			                 varNames[1]);
	  }

	  float * valBuffAll = valBuff;
	  if (valBuff2)
	  {
		  valBuffAll =new float [ncomps * numData];
		  for (long iNode = 0; iNode < numData; iNode++)
		  {
			  valBuffAll[ncomps*iNode] = valBuff[iNode];
			  valBuffAll[ncomps * iNode + 1] = valBuff2[iNode];
		  }
	  }

      //total number of data = nodes for a time step
      long ntuples        = numDataEntry; 
      vtkFloatArray *rv = vtkFloatArray::New();
	  rv->SetNumberOfComponents(ucomps);
      rv->SetNumberOfTuples(ntuples);
	  idata  = 0;
      for( long iNode = 0 ; iNode   < ntuples; iNode++)
         {

		   //if((!drywet[iNode])|| (drywet[iNode]&&(!(m_dry_wet_flag))))
		   {
             for(int iComp = 0; iComp < ncomps; iComp++)
             {
                oneEntry[iComp]   = valBuffAll[iNode*ncomps+iComp];
             }
             for(int iComp = ncomps; iComp < ucomps; iComp++)
             {
                oneEntry[iComp]= 0.0;
             }
		   }
		  // else
		  // {
		 //   for(int iComp = 0; iComp < ucomps; iComp++)
         //    {
          //      oneEntry[iComp]   = MeshConstants10::DRY_STATE;
          //   }
   
		  // }
            
           rv->SetTuple(idata, oneEntry);  
           idata++;             
         }
      delete   valBuff;
	  if (valBuff2)
	  {
		  delete valBuff2;
		  delete valBuffAll;
	  }
      return rv;
    }



    int num_data_layers    =m_num_layers;
	 if(level_center == MeshConstants10::HALF_LAYER)
    {
	  num_data_layers=m_num_layers-1;
    }
    
    int * layerStarts;
	bool is_bottom=false;
	bool is_surface=false;
    if (varName.find(m_surface_state_suffix) != string::npos) 
    {
      num_data_layers      = 1;
      layerStarts    = new int [num_data_layers];
      layerStarts[0] =m_num_layers-1;
	  if(level_center == MeshConstants10::HALF_LAYER)
      {
	    layerStarts[0]--;
      }
	  is_surface=true;
    }
    else if (varName.find(m_bottom_state_suffix) != string::npos)
    {
      num_data_layers      = 1;
      layerStarts    = new int [num_data_layers];
	  is_bottom=true;
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
  

   
    
	//int numDataPerLayer=m_nominal_size_per_layer[data_center];
	

    float * valBuff = NULL;
	float * valBuff2 = NULL;
	float * valBuffAll = NULL;

	int totalNumLayers=m_num_layers;
	
	if (!SCHISMVarPtr2)
	{
		valBuff = new float[totalNumLayers*numDataPerLayer*ncomps];
		valBuffAll = valBuff;
	}
	else
	{
		valBuff = new float[totalNumLayers*numDataPerLayer];
		valBuff2 = new float[totalNumLayers*numDataPerLayer];
		valBuffAll = new float[totalNumLayers*numDataPerLayer*ncomps];
	}

 
    int numOfRecord  = 1;
    int nodeStart    = 0;
    int timeStart    = a_timeState;
   
    SCHISMVarPtr->set_cur(timeStart);
	int * bottom = NULL;
	if (m_scribeIO)
	{
		bottom= new int[numDataPerLayer];

		if (data_center == MeshConstants10::ELEM)
		{
			m_external_mesh_provider->fillKbe(bottom, timeStart);
		}
		else if (data_center == MeshConstants10::EDGE)
		{
			m_external_mesh_provider->fillKbs(bottom, timeStart);
		}
		else
		{
			m_external_mesh_provider->fillKbp00(bottom, timeStart);
		}
		if (!(SCHISMVarPtr->get(valBuff,bottom)))
		{
			stringstream msgStream(stringstream::out);
			msgStream << "Fail to retrieve " << a_varName << " at step " << a_timeState;
			EXCEPTION3(DBYieldedNoDataException, m_data_file, m_plugin_name, msgStream.str());
		}
	}
	else
	{
		if (!(SCHISMVarPtr->get(valBuff)))
		{
			stringstream msgStream(stringstream::out);
			msgStream << "Fail to retrieve " << a_varName << " at step " << a_timeState;
			EXCEPTION3(DBYieldedNoDataException, m_data_file, m_plugin_name, msgStream.str());
		}
	}


    if (SCHISMVarPtr2)
	{
		SCHISMVarPtr2->set_cur(timeStart);

		if (m_scribeIO)
		{
			if (!(SCHISMVarPtr2->get(valBuff2,bottom)))
			{
				stringstream msgStream(stringstream::out);
				msgStream << "Fail to retrieve vector " << a_varName << "another component at step " << a_timeState;
				EXCEPTION3(DBYieldedNoDataException, m_data_file, m_plugin_name, msgStream.str());
			}
		}
		else
		{
			if (!(SCHISMVarPtr2->get(valBuff2)))
			{
				stringstream msgStream(stringstream::out);
				msgStream << "Fail to retrieve vector " << a_varName << "another component at step " << a_timeState;
				EXCEPTION3(DBYieldedNoDataException, m_data_file, m_plugin_name, msgStream.str());
			}
		}
		for (long iNode = 0; iNode < totalNumLayers*numDataPerLayer; iNode++)
		{
			valBuffAll[ncomps*iNode] = valBuff[iNode];
			valBuffAll[ncomps * iNode + 1] = valBuff2[iNode];
		}
	}
	if (bottom) delete bottom;
	debug1 << "vector all buff filled\n";
  
	long valid_var_size =0;
    long * node_start_index= new long [numDataPerLayer];
    long * num_data_at_layer = new long [totalNumLayers];

    for(int iLayer=0;iLayer<m_num_layers;iLayer++)
    {
	  num_data_at_layer[iLayer]=0;
    }
	int * kbp00=m_kbp_node;

    if (!(data_center.compare(MeshConstants10::EDGE)))
    {
	    kbp00=m_kbp_side;
    }
    else if (!(data_center.compare(MeshConstants10::ELEM)))
    {
	    kbp00=m_kbp_ele;
    }

   int num_layers =m_num_layers;
   if(level_center == MeshConstants10::HALF_LAYER)
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

      
	   if (is_bottom||is_surface) //for surface and bottom data, same as number of 2D node
	   {
	      ntuples=numDataPerLayer; 
	   }
	    rv->SetNumberOfTuples(ntuples); 

	   debug1<<"totla number of vect tuples "<<ntuples<<" "<<num_data_layers<<" \n";

       for (int iLayer   = 0 ; iLayer < num_data_layers ; iLayer++)
       {
          int layer = layerStarts[iLayer];
           
          for( int iNode = 0 ; iNode < numDataPerLayer; iNode++)
          {  

			  int bottom_layer = std::max(1,kbp00[iNode])-1;
			   if(is_bottom)
			   {
				   layer=bottom_layer+1;
				   if(layer>(num_layers-1))
				   {
					   layer=num_layers-1;
				   }
			   }
              
			  int start_index = node_start_index[iNode];

			  if(layer>=bottom_layer) 
			  {
				  //if(!drywet[iNode])
				  if((!drywet[iNode])|| (drywet[iNode]&&(!(m_dry_wet_flag))))
				  {

					  for(int iComp = 0; iComp < ncomps; iComp++)
					  {
						 oneEntry[iComp]= valBuffAll[start_index+(layer+1-std::max(1,kbp00[iNode]))*ncomps+iComp];
                 
					  }

					  for(int iComp = ncomps; iComp < ucomps; iComp++)
					  {
						 oneEntry[iComp]= 0.0;
					  }
				  }
				  else
				  {
					  for(int iComp = 0; iComp < ucomps; iComp++)
					  {
						 oneEntry[iComp]= MeshConstants10::DRY_STATE;
					  }
				  }
				  
				  rv->SetTuple(idata, oneEntry); 
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

       float * averageState = new float [numDataPerLayer*ncomps];

       vectorDepthAverage(averageState,valBuffAll,node_start_index,a_timeState,ncomps,data_center,level_center);

       idata  = 0;
       for( int iNode = 0 ; iNode   < numDataPerLayer; iNode++)
         {
				
		   //if(!drywet[iNode])
		   if((!drywet[iNode])|| (drywet[iNode]&&(!(m_dry_wet_flag))))
		   {
				for(int iComp = 0; iComp < ncomps; iComp++)
				{
				   oneEntry[iComp]   = averageState[iNode*ncomps+iComp];
				}
				for(int iComp = ncomps; iComp < ucomps; iComp++)
				{
				   oneEntry[iComp]= 0.0;
				}
			 }
			else
			{
				for(int iComp = 0; iComp < ucomps; iComp++)
				{
					oneEntry[iComp]= MeshConstants10::DRY_STATE;
				}
			}
            
           rv->SetTuple(idata, oneEntry);  
           idata++;             
         }
       delete averageState;
     }

 
    delete valBuff;
	if (valBuff2)
	{
		delete valBuff2;
		delete valBuffAll;
	}
    delete layerStarts;
    delete oneEntry;
    return rv;   
}

 void   avtSCHISMFileFormatImpl10::prepare_average(int * & a_kbp, 
	                                         int * & a_mapper, 
											 float * & a_zPtr, 
											 const int & a_timeState,
											 const std::string & a_horizontal_center,
							                 const std::string & a_vertical_center)
 {

  a_kbp    = m_kbp_node; 
  
  if(a_horizontal_center == MeshConstants10::EDGE)
	{
		a_kbp    = m_kbp_side;
        a_mapper = new int [m_num_mesh_edges*m_num_layers];
        for(int i=0;i<m_num_mesh_edges*m_num_layers;i++)
        {
	        a_mapper[i]= MeshConstants10::INVALID_NUM;
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
  else if (a_horizontal_center == MeshConstants10::ELEM)
  {
	  a_kbp    = m_kbp_ele;
      a_mapper = new int [m_num_mesh_faces*m_num_layers];
      for(int i=0;i<m_num_mesh_faces*m_num_layers;i++)
      {
	     a_mapper[i]= MeshConstants10::INVALID_NUM;
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
	      a_mapper[i]= MeshConstants10::INVALID_NUM;
       }
       int Index = 0 ;
       for(int iLayer=0;iLayer<m_num_layers;iLayer++)
       {
	     for(int iNode=0;iNode<m_num_mesh_nodes;iNode++)
	     {
		    int bottomLayer = a_kbp[iNode];
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
//  Method: avtSCHISMFileFormatImpl10::depthAverage
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
avtSCHISMFileFormatImpl10:: depthAverage(float         *  a_averageState,
                                   float         *  a_layeredState,
								   long           *  a_nodeDataStart,
                                   const int     &  a_timeState,
								   const std::string & a_horizontal_center,
							       const std::string & a_vertical_center
                                  ) 
{
 
 long numData=m_num_mesh_nodes;
  if (a_horizontal_center ==MeshConstants10::ELEM)
  {
	  numData=m_num_mesh_faces;
  }
  else if (a_horizontal_center ==MeshConstants10::EDGE)
  {
	  numData=m_num_mesh_edges;
  }
  else
  {
  }
 
  int * kbp; 
  int * mapper;
  float * zPtr;
  
  prepare_average(kbp, mapper, zPtr, a_timeState,a_horizontal_center,a_vertical_center);

  debug1<<"aveages states output \n";

  for( int iNode = 0 ;iNode< numData;iNode++)
    {
	  if(kbp[iNode]<=m_num_layers)
	  {
		  // this is number of valid data
		  int num_valid_data_layers =m_num_layers-std::max(1,kbp[iNode])+1;

	  
		   // native prism center layer 1 is duplicate of layer2
		   // it still have the same number of layers as the node center data
		  if (a_vertical_center== MeshConstants10::HALF_LAYER)
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

		   //fix for possible inactive z layers filled with 0.0 elevation 
		   if(kbp[iNode]==0)
		   {
			   int last_level_be_removed=0;
			   for(int ilevel=0;ilevel<num_valid_z_layers-1;ilevel++)
			   {
				   if(zCoords[ilevel+1]<zCoords[ilevel])
				   {
					   last_level_be_removed=ilevel;
					   //debug1<<iNode<<" has done inactive z removal to "<<last_level_be_removed<<"\n";
					   break;
				   }
			   }
			   num_valid_z_layers=num_valid_z_layers-last_level_be_removed-1;
			   num_valid_data_layers=num_valid_z_layers;
			   for(int ilevel=0;ilevel<num_valid_z_layers;ilevel++)
			   {
				   states[ilevel]=states[ilevel+last_level_be_removed+1];
				   zCoords[ilevel]=zCoords[ilevel+last_level_be_removed+1];
			   }

		   }

		   float averageState;
		   if (a_vertical_center != MeshConstants10::HALF_LAYER)
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
	   
		   if (iNode==20002)
		   {
			   debug1<<"var vals for "<<iNode<<" averageing:\n";
			   debug1<<"level center "<<a_vertical_center<<"\n";
			   debug1<<" bottom level:"<<kbp[iNode]<<"\n";
			   debug1<<" valid num of level:"<<num_valid_data_layers<<"\n";
				for(int iLayer  = 0;iLayer<num_valid_data_layers; iLayer++)
			   {
				  debug1<<states[iLayer]<<" "<<zCoords[iLayer]<<"\n"; 
				}
				debug1<<"averaged to "<<averageState<<"\n";
		   }
		   delete   states;
		   delete   zCoords;
	  }
	  else
	  {
         a_averageState[iNode] = MeshConstants10::DRY_STATE;
	  }
    }  
    delete zPtr;
	delete mapper;
    //debug1<<numData<<" center "<<m_data_center<<"\n";
}


// ****************************************************************************
//  Method: avtSCHISMFileFormatImpl10::vectorDepthAverage
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
avtSCHISMFileFormatImpl10:: vectorDepthAverage(float        *  a_averageState,
                                         float        *  a_layeredState,
									     long          *  a_nodeDataStart,
                                         const int    &  a_timeState,
                                         const int    &  a_ncomps,
										 const std::string & a_horizontal_center,
							             const std::string & a_vertical_center
                                         ) 
{
  
   long numData=m_num_mesh_nodes;
  if (a_horizontal_center ==MeshConstants10::ELEM)
  {
	  numData=m_num_mesh_faces;
  }
  else if (a_horizontal_center ==MeshConstants10::EDGE)
  {
	  numData=m_num_mesh_edges;
  }
  else
  {
  }
 
  int * kbp; 
  int * mapper;
  float * zPtr;
  
  prepare_average(kbp, mapper, zPtr, a_timeState,a_horizontal_center,a_vertical_center);

  for( int iNode = 0 ;iNode< numData;iNode++)
    { 
	  if(kbp[iNode]<=m_num_layers)
	  {
		  int num_valid_data_layers =m_num_layers-std::max(1,kbp[iNode])+1;
	 
		  if (a_vertical_center == MeshConstants10::HALF_LAYER)
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
		   //fix for possible inactive z layers filled with 0.0 elevation 
		   if(kbp[iNode]==0)
		   {
			   int last_level_be_removed=0;
			   for(int ilevel=0;ilevel<num_valid_z_layers-1;ilevel++)
			   {
				   if(zCoords[ilevel+1]<zCoords[ilevel])
				   {
					   last_level_be_removed=ilevel;
					   break;
				   }
			   }
			   num_valid_z_layers=num_valid_z_layers-last_level_be_removed-1;
				num_valid_data_layers=num_valid_z_layers;
			   for(int ilevel=0;ilevel<num_valid_z_layers;ilevel++)
			   {
				   for(int iComp=0;iComp<a_ncomps;iComp++)
				   {
					 states[iComp][ilevel]=states[iComp][ilevel+last_level_be_removed+1];
				   }
				   zCoords[ilevel]=zCoords[ilevel+last_level_be_removed+1];
			   }

		   }

		  if(iNode==1850)
		  {
			  debug1<<" averaged sate of 1850 : ";
		  }
	  
	  
		   for( int iComp = 0; iComp < a_ncomps; iComp++)
		   {
			 float averageState;
			 if (a_vertical_center!=MeshConstants10::HALF_LAYER)
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
	 else
	 {
		 for( int iComp = 0; iComp < a_ncomps; iComp++)
        {
		  a_averageState[iNode*a_ncomps+iComp] = MeshConstants10::DRY_STATE;
       }
	 }
   }  

  delete zPtr;
  delete mapper;

}


// ****************************************************************************
//  Method: avtSCHISMFileFormatImpl10::GetSingleLayerVar
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
avtSCHISMFileFormatImpl10::getSingleLayerVar(float    *          a_valBuff,
                                             SCHISMFile10*         a_SCHISMOutPtr,
                                             const int &         a_timeState, 
                                             const std::string&  a_varName) const
{
  time_t startTicks      = clock();
  SCHISMVar10 * SCHISMVarPtr = a_SCHISMOutPtr->get_var(a_varName);
  if (!(SCHISMVarPtr->is_valid()))
    {
      
      EXCEPTION1(InvalidVariableException, a_varName);
    }
  debug1<<"begin to read "<<a_varName<<endl;

  int nodeIndex0  = 0;
  int timeRecord  = a_timeState;
  std::string data_center = SCHISMVarPtr->get_horizontal_center();
  SCHISMVarPtr->set_cur(timeRecord);
    
  debug1<<"set start of  "<<a_varName<<endl;
  int numOfRecord  = 1;
  if (m_scribeIO)
  {
	  int * bottom=NULL; 

	  if (data_center == MeshConstants10::ELEM)
	  {
		  bottom = new int[m_num_mesh_faces];
		  m_external_mesh_provider->fillKbe(bottom, timeRecord);
	  }
	  else if (data_center == MeshConstants10::EDGE)
	  {
		  bottom = new int[m_num_mesh_edges];
		  m_external_mesh_provider->fillKbs(bottom, timeRecord);
	  }
	  else
	  {
		  bottom = new int[m_num_mesh_nodes];
		  m_external_mesh_provider->fillKbp00(bottom, timeRecord);
	  }

	  if (!(SCHISMVarPtr->get(a_valBuff, bottom)))
	  {
		  stringstream msgStream(stringstream::out);
		  msgStream << "Fail to retrieve " << a_varName << " at step " << a_timeState;
		  EXCEPTION3(DBYieldedNoDataException, m_data_file, m_plugin_name, msgStream.str());
	  }
	  delete bottom;

  }
  else
  {
	  if (!SCHISMVarPtr->get(a_valBuff))
	  {
		  stringstream msgStream(stringstream::out);
		  msgStream << "Fail to retrieve " << a_varName << " at step " << a_timeState;
		  EXCEPTION3(DBYieldedNoDataException, m_data_file, m_plugin_name, msgStream.str());
	  }
  }
   
  time_t endTicks      = clock();
  debug1<<"time used in getting var "<<a_varName<<":"<<endTicks-startTicks<<endl;
}




// ****************************************************************************
//  Method: avtSCHISMFileFormatImpl10::ActivateTimestep
//
//  Purpose:
//      
//      
//  Programmer: qshu -- generated by xml2avt
//  Creation:   Mon Aug 6 09:53:36 PDT 2012
//
// ****************************************************************************

void  avtSCHISMFileFormatImpl10::ActivateTimestep(const std::string& a_filename)
{
  Initialize(a_filename);
}

// ****************************************************************************
//  Method: avtSCHISMFileFormatImpl10::Initialize
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
void avtSCHISMFileFormatImpl10::Initialize(std::string a_data_file)
{
  
  debug1 << a_data_file << " is initialized:" << m_initialized<<"\n";
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

	try
	{ 
	   std::string file_name = m_data_file.substr(found+1);
	   size_t      found_underline = file_name.find_last_of("_");
	   std::string  var_name = file_name.substr(0, found_underline);

	   std::map<std::string,std::string>::iterator itX,itY;
	   itX = Vector3DVarMapX.find(var_name);
	   itY = Vector3DVarMapY.find(var_name);

	   if (itX != Vector3DVarMapX.end())
	   {
		   
		   m_data_file_ptr = new NetcdfSchismOutput10(m_data_file);
#ifdef _WIN32  
		   std::string extra_file = m_data_file_path + "\\" + file_name.replace(found_underline-1, 1 , "Y");
#else
	       std::string extra_file = m_data_file_path + "/" + file_name.replace(found_underline-1,1 , "Y");
#endif
		   try 
		   {
              m_data_file_ptr2 = new NetcdfSchismOutput10(extra_file);
		   }
		   catch (...)
		   {
			   stringstream msgStream(stringstream::out);
			   msgStream << extra_file<<" is not valid\n";
			   EXCEPTION1(InvalidDBTypeException, msgStream.str().c_str());
		   }
	   }
	   else if (itY != Vector3DVarMapY.end())
	   {

		   m_data_file_ptr2 = new NetcdfSchismOutput10(m_data_file);
#ifdef _WIN32 
		   std::string extra_file = m_data_file_path + "\\" + file_name.replace(found_underline-1, 1, "X");
#else
	       std::string extra_file = m_data_file_path + "/" + file_name.replace(found_underline-1, 1, "X");
#endif
		   try
		   {
			   m_data_file_ptr = new NetcdfSchismOutput10(extra_file);
		   }
		   catch (...)
		   {
			   stringstream msgStream(stringstream::out);
			   msgStream << extra_file << " is not valid\n";
			   EXCEPTION1(InvalidDBTypeException, msgStream.str().c_str());
		   }
	   }
	   else
	   {
		   m_data_file_ptr = new NetcdfSchismOutput10(m_data_file);
	   }
	}
	catch(SCHISMFileException10 e)
	{
		EXCEPTION1(InvalidDBTypeException,e.what());
	}
	catch(...)
	{
		EXCEPTION1(InvalidDBTypeException,"not valid schsim NC output");
	}

	//find out format of output by reading global attribute source
	//if "V10" in the string 5.8 format, otherwise scribeIo format (no atribute or other values)
	try
	{
		std::string source = m_data_file_ptr->global_att_as_string(MeshConstants10::source);
		std::size_t found_v10= source.find(MeshConstants10::SCHISM58_OUTPUT_FORMAT);
		if (found_v10 != std::string::npos)
		{
			m_scribeIO = false;
		}
		else
		{
			m_scribeIO = true;
		}
	}
	catch (...)
	{
		m_scribeIO = true;
	}



    //here a simple and temp way to decided if data file is the latest scriber format
	std::string meshFilePath = m_data_file;
	//std::size_t found2 = m_data_file.find("schout");
	
	// if not name by schout then treated as scriber format, mesh file should
	// be out2d*.nc + zcor*.nc
    //if (found2==std::string::npos)
	if(m_scribeIO == true)
	{
		//m_scribeIO = true;
		size_t found3 = m_data_file.find_last_of("_");
	    std::string suffix=m_data_file.substr(found3);
#ifdef _WIN32
		meshFilePath = m_data_file_path+"\\out2d"+suffix;
#else
	   meshFilePath = m_data_file_path+"/out2d"+suffix;
#endif
		//debug1 << "found2:" << found2 << " "<<meshFilePath<<"\n";
	}


		try
		{
			m_external_mesh_provider = new ZCoordMeshProvider10(meshFilePath);
			
		}
		catch (SCHISMFileException10 e)
		{
			EXCEPTION1(InvalidDBTypeException, e.what());
		}
		catch (...)
		{
			EXCEPTION1(InvalidDBTypeException, "no valid mesh exist");
		}

		
	//if (found2 == std::string::npos)
	//{
	//	
	//	m_data_file_ptr->set_mesh_bottom(m_external_mesh_provider->get_mesh_data_ptr(),0);
	//	if(m_data_file_ptr2)
	//	{
	//		m_data_file_ptr2->set_mesh_bottom(m_external_mesh_provider->get_mesh_data_ptr(),0);
	//	}
	//}
    okay = m_data_file_ptr->is_valid();
	
   
    // If your file format API could not open the file then throw   
    // an exception.
     if (!okay)
      {
        EXCEPTION1(InvalidDBTypeException,
                    (m_data_file_path+"is not a valid SCHISM output file").c_str());
      }
    debug1<<"file is opened\n";
	
	if (!m_scribeIO)
	{
		m_dry_wet_flag = m_data_file_ptr->get_dry_wet_val_flag();
	}
	else
	{
		m_dry_wet_flag = 1;
	}
	debug1<<"wet_dry_flag is "<<m_dry_wet_flag<<"\n";
    debug1<<"begin get dim\n";


    

	debug1<<"loading zcor/hgrid file";

	found  = m_data_file.find_last_of(".");
	std::string typeStr = m_data_file.substr(found);
	
	
    //debug1<<"opening zcor file "<<zCorFilePath;

   
    okay = m_external_mesh_provider->isValid();
	
	if (!okay)
	{
		 EXCEPTION1(InvalidDBTypeException,
					 (m_data_file_path+"doesn't contain valid mesh infomation").c_str());

	}
	
    getTime();
    debug1<<"got time\n";
	
	debug1<<"getting dim";
	getMeshDimensions(m_external_mesh_provider);
	debug1<<"got dimension\n";
	
	debug1<<"loading coords";
    loadMeshCoordinates(m_external_mesh_provider);


	m_mesh_is_static=m_external_mesh_provider->mesh3d_is_static();
	debug1 << "static mesh:" << m_mesh_is_static << "\n";
    PopulateVarMap();
	debug1 << "var mesh loaded\n";
	int current_time=0;
	this->load_ele_dry_wet(current_time);
	debug1 << "dry/wet loaded\n";
	this->load_bottom(current_time);
	debug1 << "bottom loaded\n";
    m_initialized = true;

	debug1<<"done initialize\n";
  }
}




// a reduant populate var map dic, fix a bug in which this dic is 
// lost when switch a different data file group in the same .visit file
// called in initialize()
void avtSCHISMFileFormatImpl10::PopulateVarMap()
{
  int numVar = m_data_file_ptr->num_vars();
  //std::string  location = m_data_file_ptr->data_center();

  for(int iVar = 0;iVar < numVar; iVar++)
   {
      SCHISMVar10*  varPtr  = m_data_file_ptr->get_var(iVar);
      std::string varName = varPtr->name();
      std::string  location = varPtr->get_horizontal_center();
      if ((varName==m_node_surface) || (varName==m_node_depth))
        {
          continue; 
        }

	  std::map<std::string,std::string>::iterator itX,itY;
	  itX = Vector3DVarMapX.find(varName);
	  itY = Vector3DVarMapY.find(varName);

	  if ((itX != Vector3DVarMapX.end())||(itY != Vector3DVarMapY.end()))
	  {
		 
		  std::string label = "";
		  std::size_t found = varName.find_last_of("_");
		  if (found != std::string::npos)
		  {
			  label = varName.substr(0, found-1)+varName.substr(found);
		  }
		  else
		  {
			  label = varName.substr(0, varName.length() - 1);
		  }
		  
		  std::string vars = "";
		  if(itX!=Vector3DVarMapX.end())
		  {
			  vars=Vector3DVarMapX[varName];
		  }
		  else
		  {
			  vars=Vector3DVarMapY[varName];
		  }

		  std::map<std::string, std::string>::iterator it;
		  m_var_name_label_map[label] = vars;
          m_var_name_label_map[label + m_surface_state_suffix] = vars;
		  m_var_name_label_map[label + m_bottom_state_suffix] = vars;
	      m_var_name_label_map[label + m_depth_average_suffix] = vars;
		  continue;
		  
	  }

	  itX = Vector2DVarMapX.find(varName);
	  itY = Vector2DVarMapY.find(varName);
	  if ((itX != Vector2DVarMapX.end()) || (itY != Vector2DVarMapY.end()))
	  {
		  
		  std::string label = varName.substr(0, varName.length() - 1);
		  std::string vars = "";
		  if (itX != Vector2DVarMapX.end())
		  {
			  vars = Vector2DVarMapX[varName];
		  }
		  else
		  {
			  vars = Vector2DVarMapY[varName];
		  }

	      m_var_name_label_map[label] = vars;	  
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
//  Method: avtSCHISMFileFormatImpl10::GetTimes
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
void   avtSCHISMFileFormatImpl10::GetTimes(std::vector<double> & a_times)
{

  //copy saved time into a_times
  for(int i=0;i<m_num_time_step;i++)
    {
      a_times.push_back(m_time_ptr[i]);
    }   

}

// ****************************************************************************
//  Method: avtSCHISMFileFormatImpl10::getTime
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
void  avtSCHISMFileFormatImpl10::getTime()
{

  SCHISMDim10 * dimTimePtr      = m_data_file_ptr->get_dim(m_dim_time);

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
    
      SCHISMVar10* SCHISMTimePtr = m_data_file_ptr->get_var(m_time);
    
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



bool  avtSCHISMFileFormatImpl10::SCHISMVarIs3D(SCHISMVar10*  a_varPtr ) const
{
	int totalNumDim=a_varPtr->num_dims();
	for (int dim=0;dim<totalNumDim;dim++)
	{
		SCHISMDim10* dimPtr=a_varPtr->get_dim(dim);
		if (dimPtr->name()==m_dim_layers)
			return true;
	}
	return false;
}
bool  avtSCHISMFileFormatImpl10::SCHISMVarIsVector(SCHISMVar10* a_varPtr) const
{
	int totalNumDim=a_varPtr->num_dims();
	for (int dim=0;dim<totalNumDim;dim++)
	{
		SCHISMDim10* dimPtr=a_varPtr->get_dim(dim);
		if (dimPtr->name()==m_dim_var_component)
			return true;
	}
	return false;
}

static Registrar registrar("combine10_nc4", &avtSCHISMFileFormatImpl10::create);
