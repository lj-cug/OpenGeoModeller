#ifndef AVTMDSCHISMFILEFORMATIMPL_H
#define AVTMDSCHISMFILEFORMATIMPL_H
#include <vector>
#include <map>
#include <vtkUnstructuredGrid.h>
// has to put mpi.h before any netcdf.h include to avoid redefining of mpi_comm,... such as.
#ifdef PARALLEL
#include <mpi.h>
#endif
//#include "SCHISMFile10.h"
#include "MDSCHISMOutput.h"
#include "MeshProvider10.h"
#include "MDSCHISMMeshProvider.h"
#include "FileFormatFavorInterface.h"



class avtMDSCHISMFileFormatImpl : public FileFormatFavorInterface
{
  public:
                    avtMDSCHISMFileFormatImpl();
  virtual           ~avtMDSCHISMFileFormatImpl();

  static           FileFormatFavorInterface * create();
  //
  // This is used to return unconvention data -- ranging from material
  // information to information about block connectivity.
  //
  // virtual void      *GetAuxiliaryData(const char *var, int timestep, 
  //                                     const char *type, void *args, 
  //                                     DestructorFunction &);
  //
  
  //
  // If you know the times and cycle numbers, overload this function.
  // Otherwise, VisIt will make up some reasonable ones for you.
  //
  // virtual void        GetCycles(std::vector<int> &);

   void           GetTimes(std::vector<double> & a_times);
   int            GetNTimesteps(const std::string& a_filename);
   //void           ActivateTimestep(const std::string& a_filename);
   const char    *GetType(void)   { return "MDUGrid"; };
   void           FreeUpResources(void); 
  
   vtkDataSet    *GetMesh(int          a_timeState, 
                          int          a_domainID,
	                      avtMDSCHISMFileFormat * a_avtFile,
                          const char * a_meshName);

   vtkDataArray  *GetVar(int           a_timeState,
                         int           a_domainID,
                         const char *  a_varName);

   vtkDataArray  *GetVectorVar(int          a_timeState, 
                               int          a_domainID,
                               const char * a_varName);
  


   void   PopulateDatabaseMetaData(avtDatabaseMetaData * a_metaData, 
	                               const std::string     a_datafile,
                                   int                   a_timeState);
								   
    void  BroadcastGlobalInfo(avtDatabaseMetaData *metadata);

	//both return just first data file and mesh provider
	MDSchismOutput * get_a_data_file();
	MDSCHISMMeshProvider * get_a_mesh_provider();
	int num_domain();
	bool           SCHISMVarIs3D(SCHISMVar10*  a_varPtr) const;
	bool           SCHISMVarIsVector(SCHISMVar10* a_varPtr) const;
	void set_var_name_label_map(const std::map<std::string, std::string> a_map);
	void set_var_mesh_map(const std::map<std::string, std::string> a_map);
	void set_var_horizontal_center_map(const std::map<std::string, std::string> a_map);
	void set_var_vertical_center_map(const std::map<std::string, std::string> a_map);
	
protected:

  void           Initialize(std::string a_data_file);
  
//private:

  void           PopulateStateMetaData(avtDatabaseMetaData * a_metaData, 
                                       int                   a_timeState);

  void           addFaceCenterData(avtDatabaseMetaData * a_metaData,
	                               SCHISMVar10            * a_varPtr,
								   const std::string   & a_varName,
								   const std::string   & a_varLabel,
								   const avtCentering  & a_center);

  void           addNodeCenterData(avtDatabaseMetaData * a_metaData,
	                               SCHISMVar10            * a_varPtr,
								   const std::string   & a_varName,
								   const std::string   & a_varLabel,
								   const avtCentering  & a_center);

  void           addSideCenterData(avtDatabaseMetaData * a_metaData,
	                               SCHISMVar10            * a_varPtr,
								   const std::string   & a_varName,
								   const std::string   & a_varLabel,
								    const avtCentering  & a_center);

  void           create2DUnstructuredMesh( vtkUnstructuredGrid *a_uGrid,
	                                       long                 *a_meshEle,
                                           const  int          &a_domainID,
										   const  int          &a_timeState);

  void           create2DUnstructuredMeshNoDryWet( vtkUnstructuredGrid *a_uGrid,
	                                                  long                 *a_meshEle,
	                                                  const  int          &a_domainID,
												      const  int          &a_timeState);

  void           create3DUnstructuredMesh(vtkUnstructuredGrid *a_uGrid,
	                                      long                 *a_meshEle,
										  long                 *a_2DPointto3DPoints,
	                                      const  int          &a_domainID,
										  const  int          &a_timeState);

  void           createLayerMesh(vtkUnstructuredGrid *a_uGrid,
	                             long                 *a_meshEle,
								 long                 *a_2DPointto3DPoints,
	                             const  int          &a_domainID,
							     const  int          &a_timeState);

  void           create2DPointMesh( vtkUnstructuredGrid *a_uGrid,
	                                long                 *a_meshEle,
	                                const  int          &a_domainID,
								    const  int          &a_timeState);

  void           create3DPointMesh( vtkUnstructuredGrid *a_uGrid,
	                                long                 *a_meshEle,
	                                const  int          &a_domainID,
								    const  int          &a_timeState);

   void          create3DPointFaceMesh( vtkUnstructuredGrid *a_uGrid,
	                                    long                *a_meshEle,
	                                    const  int          &a_domainID,
								        const  int          &a_timeState);

  


  vtkDataArray*   getLayer(const int& a_domain);

  vtkDataArray*   getLevel(const string& a_level_name,const int& a_domain);

  vtkDataArray*   getBottom(const string& a_bottom_name,const int& a_domain);

  void           PopulateVarMap();
  void           getTime();

  

  void           getMeshDimensions(const int& a_dominID, MeshProvider10 * a_meshProviderPtr);

  //void           loadMeshCoordinates(MeshProvider10 * a_meshProviderPtr);

   

  void           updateMeshZCoordinates(vtkPoints  * a_pointSet,
                                        const int    a_timeState,
	                                    const int    a_domainID,
                                        const char * a_meshName);
  
 // void           retrieve1DArray(float             * a_valBuff,
//	                             SCHISMFile10         * a_selfeFilePtr,
  //                               const std::string & a_varToken,
   //                              const int     & a_numVar) const;
  //void           retrieve1DArray(int           * a_valBuff,
	                             //SCHISMFile10     * a_selfeFilePtr,
                                 //const std::string & a_varToken,
                                 //const int     & a_numVar) const;

  void            loadAndCacheZ(const int& a_timeState,float * a_cache);
  void            loadAndCacheZSide(const int& a_timeState,float * a_cache);
  void            loadAndCacheZEle(const int& a_timeState,float * a_cache);

  void            getSingleLayerVar(float   *          a_valBuff,
	                                SCHISMFile10*         a_selfeOutPtr,
                                    const int&         a_timeState,
                                    const std::string& a_varname) const;

  void            depthAverage(float        *  a_averageState,
                               float        *  a_layeredState,
							   long          *  a_nodeDataStart,
                               const int    &  a_timeState,
							   const int    &  a_domain,
							   const std::string & a_horizontal_center,
							   const std::string & a_vertical_center
                               ) ;

  void            vectorDepthAverage(float         *  a_averageState,
                                     float         *  a_layeredState,
									 long           *  a_nodeDataStart,
                                     const int     &  a_timeState,
									 const int    &  a_domain,
                                     const int     &  a_ncomps,
									 const std::string & a_horizontal_center,
							         const std::string & a_vertical_center
                                    ) ;



  
 

  void           validTopBottomNode(int       &   a_validTopNodeNum,
	                                int       &   a_validBottomNodeNum,
									long       *   a_validTopNode,
									long       *   a_validBottomNode,
									const int &   a_layerID,
									long*          a_faceNodePtr,
	                                int       *   a_kbp_node) const;

  void            insertTriangle3DCell(vtkUnstructuredGrid * a_uGrid,
	                                   const int           & a_validTopNodeNum,
									   const int           & a_validBottomNodeNum,
									         long          * a_validTopNode,
											 long          * a_validBottomNode,
											 long          * a_faceNodePtr,
											 long          * a_2DPointto3DPoints,
	                                         int           * a_kbp_node,
									    const long         & a_Cell,
	                                    const int           & a_layerID,
	                                    const int           & a_numLayers);

  void            insertQuad3DCell(vtkUnstructuredGrid  *     a_uGrid,
	                               const int            &     a_validTopNodeNum,
								   const int            &     a_validBottomNodeNum,
									     long           *     a_validTopNode,
										 long           *     a_validBottomNode,
										 long           *     a_faceNodePtr,
										 long           *     a_2DPointto3DPoints,
	                                     int            *     a_kbp_node,
								   const long           &     a_Cell,
	                               const int            &     a_layerID,
	                               const int            &     a_numLayers);

   

  bool            fourPointsCoplanar(double p1[3],
	                                 double p2[3],
									 double p3[3],
									 double p4[3]);

  void            prepare_average(int * & a_kbp, 
	                             int * & a_mapper, 
								 float * & a_zPtr,
								 const int & a_timeState, 
								 const int & a_domainID,
								 const std::string & a_horizontal_center,
							     const std::string & a_vertical_center);

  void           load_node_dry_wet(const int & a_time,const int & a_domain);
  void           load_side_dry_wet(const int & a_time,const int & a_domain);
  void           load_ele_dry_wet(const int & a_time,const int & a_domain);
  void           load_bottom(const int & a_time,const int & a_domainID);
  //broadcast string map from rank 0 to all other ranks
  void           broadCastStringMap(std::map<std::string, std::string>& m_map,int myrank);
  int            load_per_proc_file(const std::string& a_path,int & num_node,int & num_side,int & num_ele) const;
  void           count_node_side_num_domain(const std::string& a_path, const int& num_proc);
  vtkDataArray*   get_ele_global_id(const int& a_domain);
  vtkDataArray*   get_node_global_id(const int& a_domain);
  vtkDataArray*   get_node_depth(const int& a_domain);

private:
  bool         m_initialized;
  bool         m_mesh_is_static;
  std::string  m_data_file;
  // path where  m_data_file is under
  std::string  m_data_file_path; 
  std::string  m_plugin_name;
  std::map<int,MDSchismOutput*>   m_data_files;
  

  // element centered data use mesh from other file
  std::map<int,MDSCHISMMeshProvider*> m_external_mesh_providers;
  std::vector<int>  m_domain_list;
  
  // a number of token of saved vars and attributes
  std::string  m_data_description;
  std::string  m_mesh_var;
  std::string  m_var_label_att;
  std::string  m_var_location_att; 
  
  // name for meshes
  std::string  m_mesh_3d;
  std::string  m_layer_mesh;
  std::string  m_mesh_2d;
  std::string  m_mesh_2d_no_wet_dry;
  std::string  m_side_center_point_3d_mesh;
  std::string  m_side_center_point_2d_mesh;
  // vertical side-faces center points, 3d flux data uses it
  std::string  m_face_center_point_3d_mesh;
  
  // dimension size of single layer of 2D Mesh 
  std::map<int,long>          m_num_mesh_nodes;
  std::map<int,long>          m_num_mesh_edges;
  std::map<int,long>          m_num_mesh_faces;
  std::map<int,long>          m_total_valid_3D_point;
  std::map<int,long>          m_total_valid_3D_side;
  std::map<int,long>          m_total_valid_3D_ele;
  std::map<int, std::map<long, int>> m_3d_node_to_2d_node;

  // number of time step 
  std::string  m_dim_time;
  long         m_num_time_step;
  float *      m_time_ptr;
  std::string  m_time;

  //coordinates and depth
  std::string  m_node_depth;
  std::string  m_node_depth_label;
  std::string  m_node_surface;
  std::string  m_node_surface_label;
  std::string  m_layerSCoord;
  std::string  m_dim_layers;
  std::string  m_dim_var_component;
  // this is number of level in schsim
  int          m_num_layers;
  float     *  m_node_x_ptr;
  float     *  m_node_y_ptr;
  float     *  m_node_z_ptr;
  // this is kbp read from mesh provider, always node center
  //int       *  m_kbp00;
  std::map<int,int *>  m_kbp_node;
  std::map<int,int >  m_kbp_node_time;
  bool         m_kbp_node_filled;
  std::map<int,int *>  m_kbp_ele;
  std::map<int,int >   m_kbp_ele_time;
  std::map<int, int*>   m_kbp_prism;
  std::map<int, int>    m_kbp_prism_time;
  bool         m_kbp_ele_filled;
  std::map<int,int *>  m_kbp_side;
  std::map<int,int>    m_kbp_side_time;
  bool         m_kbp_side_filled;
  int          m_cache_kbp_id;
  //  this is the kbp read from data file itself
  //  it is different from m_kbp00 for element/prism centered data
  int       *  m_kbp_data;

  int          m_dry_wet_flag; // 0: dry cell/ele/side filled with last wetting val, 1: junk
  std::map<int,int*>         m_node_dry_wet;
  std::map<int,int*>         m_ele_dry_wet;
  std::map<int,int*>         m_side_dry_wet;
  std::map<int,int>          m_node_dry_wet_cached_time;
  std::map<int,int>          m_side_dry_wet_cached_time;
  std::map<int,int>          m_ele_dry_wet_cached_time;
 

  
  int           m_number_domain;
  std::string  m_surface_state_suffix;
  std::string  m_bottom_state_suffix;
  std::string  m_depth_average_suffix;

  //map convert netcdf centering into avt centering
  std::map<std::string, avtCentering> m_center_map;

  //map convert var label into netcdf varname token
  std::map<std::string, std::string>  m_var_name_label_map;


  //map convert var label into netcdf varname token
  std::map<std::string, std::string>  m_var_vertical_center_map;


  //map convert var label into netcdf varname token
  std::map<std::string, std::string>  m_var_horizontal_center_map;


  // maping var label to its mesh name
  std::map<std::string,std::string> m_var_mesh_map;

  // caching SCHISM var (not visit labeled var)  dim 3D/2D only
  std::map<std::string,int> m_var_dim;

 // norminal num data per layer for different centering, node,side,element
  std::map<std::string,int> m_nominal_size_per_layer;

  // dry state
  float       m_dry_surface;  

  std::string m_data_center;

  //half or full level, only meaningfull for 3d data
  std::string m_level_center;

  int m_rank;

  //debug var del after done
  int m_tri_wedge;
  int m_tri_pyramid;
  int m_tri_tetra;

  int m_quad_hexhedron;
  int m_quad_wedge;
  int m_quad_pyramid;

  //store num of domain a side/node resides to figure out inter-facial node/side and mark them as ghost
  std::vector<int> m_side_num_domain;
  std::vector<int> m_node_num_domain;

  int m_global_num_node;
  int m_global_num_side;
  int m_global_num_ele;
};


#endif
