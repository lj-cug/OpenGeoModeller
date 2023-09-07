#ifndef AVTSCHISMFILEFORMATIMPL10_H
#define AVTSCHISMFILEFORMATIMPL10_H
#include <vector>
#include <map>
#include <vtkUnstructuredGrid.h>
#include "SCHISMFile10.h"
#include "SCHISMMeshProvider10.h"
#include "FileFormatFavorInterface.h"


class avtSCHISMFileFormatImpl10 : public FileFormatFavorInterface
{
  public:
                    avtSCHISMFileFormatImpl10();
  virtual           ~avtSCHISMFileFormatImpl10() {;};

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
   void           ActivateTimestep(const std::string& a_filename);
   const char    *GetType(void)   { return "UGrid"; };
   void           FreeUpResources(void); 
  
   vtkDataSet    *GetMesh(int          a_timeState, 
	                             avtSCHISMFileFormat * a_avtFile,
                                 const char * a_meshName);

   vtkDataArray  *GetVar(int           a_timeState,
                                const char *  a_varName);

   vtkDataArray  *GetVectorVar(int          a_timeState, 
                                      const char * a_varName);
  


   void   PopulateDatabaseMetaData(avtDatabaseMetaData * a_metaData, 
	                                      avtSCHISMFileFormat * a_avtFile,
                                          int                   a_timeState);
protected:

  void           Initialize(std::string a_data_file);

 // for scribe io 2d vector data, all coms saved in out2d file 
  vtkDataArray  *GetVector2d(int          a_timeState,const std::string a_varName);
  
  
//private:

  void           PopulateStateMetaData(avtDatabaseMetaData * a_metaData, 
	                                    avtSCHISMFileFormat * a_avtFile,
                                       int                   a_timeState);

  void           addFaceCenterData(avtDatabaseMetaData * a_metaData,
	                               SCHISMVar10            * a_varPtr,
								    avtSCHISMFileFormat * a_avtFile,
								   const std::string   & a_varName,
								   const std::string   & a_varLabel,
								   const avtCentering  & a_center);

  void           addNodeCenterData(avtDatabaseMetaData * a_metaData,
	                               SCHISMVar10            * a_varPtr,
								    avtSCHISMFileFormat * a_avtFile,
								   const std::string   & a_varName,
								   const std::string   & a_varLabel,
								   const avtCentering  & a_center);

  void           addSideCenterData(avtDatabaseMetaData * a_metaData,
	                               SCHISMVar10            * a_varPtr,
								    avtSCHISMFileFormat * a_avtFile,
								   const std::string   & a_varName,
								   const std::string   & a_varLabel,
								    const avtCentering  & a_center);

 virtual void           create2DUnstructuredMesh( vtkUnstructuredGrid *a_uGrid,
	                                       long                 *a_meshEle,
										   const  int          &a_timeState);

 virtual void           create2DUnstructuredMeshNoDryWet( vtkUnstructuredGrid *a_uGrid,
	                                                  long                 *a_meshEle,
												      const  int          &a_timeState);

 virtual  void           create3DUnstructuredMesh(vtkUnstructuredGrid *a_uGrid,
	                                      long                 *a_meshEle,
										  long                 *a_2DPointto3DPoints,
										  const  int          &a_timeState);

 virtual void           createLayerMesh(vtkUnstructuredGrid *a_uGrid,
	                             long                 *a_meshEle,
								 long                 *a_2DPointto3DPoints,
							     const  int          &a_timeState);

 virtual void           create2DPointMesh( vtkUnstructuredGrid *a_uGrid,
	                                long                 *a_meshEle,
								    const  int          &a_timeState);

 virtual void           create3DPointMesh( vtkUnstructuredGrid *a_uGrid,
	                                long                 *a_meshEle,
								    const  int          &a_timeState);

 virtual  void          create3DPointFaceMesh( vtkUnstructuredGrid *a_uGrid,
	                                    long                *a_meshEle,
								        const  int          &a_timeState);

  


  vtkDataArray*   getLayer();

  vtkDataArray*   getLevel(const string& a_level_name);

  vtkDataArray*   getBottom(const string& a_bottom_name);

  void           PopulateVarMap();
  void           getTime();

  

  void           getMeshDimensions(MeshProvider10 * a_meshProviderPtr);

  void           loadMeshCoordinates(MeshProvider10 * a_meshProviderPtr);

   

  void           updateMeshZCoordinates(vtkPoints  * a_pointSet,
                                        const int    a_timeState,
                                        const char * a_meshName);
  
  //void           retrieve1DArray(float             * a_valBuff,
	 //                            SCHISMFile10         * a_selfeFilePtr,
  //                               const std::string & a_varToken,
  //                               const int     & a_numVar) const;



  //void           retrieve1DArray(int           * a_valBuff,
	 //                            SCHISMFile10     * a_selfeFilePtr,
  //                               const std::string & a_varToken,
  //                               const int     & a_numVar) const;

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
							   const std::string & a_horizontal_center,
							   const std::string & a_vertical_center
                               ) ;

  void            vectorDepthAverage(float         *  a_averageState,
                                     float         *  a_layeredState,
									 long           *  a_nodeDataStart,
                                     const int     &  a_timeState,
                                     const int     &  a_ncomps,
									 const std::string & a_horizontal_center,
							         const std::string & a_vertical_center
                                    ) ;


  bool           SCHISMVarIs3D(SCHISMVar10*  a_varPtr ) const;
  bool           SCHISMVarIsVector(SCHISMVar10* a_varPtr) const;
  
 

  void           validTopBottomNode(int       &   a_validTopNodeNum,
	                                int       &   a_validBottomNodeNum,
									long       *   a_validTopNode,
									long       *   a_validBottomNode,
									const int &   a_layerID,
									long*          a_faceNodePtr) const;

  void            insertTriangle3DCell(vtkUnstructuredGrid * a_uGrid,
	                                   const int           & a_validTopNodeNum,
									   const int           & a_validBottomNodeNum,
									         long          * a_validTopNode,
											 long          * a_validBottomNode,
											 long          * a_faceNodePtr,
											 long          * a_2DPointto3DPoints,
									    const long         & a_Cell,
	                                   const int           & a_layerID);

  void            insertQuad3DCell(vtkUnstructuredGrid  *     a_uGrid,
	                               const int            &     a_validTopNodeNum,
								   const int            &     a_validBottomNodeNum,
									     long           *     a_validTopNode,
										 long           *     a_validBottomNode,
										 long           *     a_faceNodePtr,
										 long           *     a_2DPointto3DPoints,
								   const long           &     a_Cell,
	                               const int            &     a_layerID);

   void            insert8NodesPolyhedron(vtkUnstructuredGrid   *     a_uGrid,
	                                       vtkIdType            *     a_verts,
								           long                 *     a_validTopNode,
									       long                 *     a_validBottomNode,
										   long                 *     a_2DPointto3DPoints,
								           const long           &     a_Cell,
	                                       const int            &     a_layerID,
										   const bool           &     a_bottomCoplane,
										   const bool           &     a_topCoplane);

   void            insert7NodesPolyhedron(vtkUnstructuredGrid   *     a_uGrid,
	                                       vtkIdType            *     a_verts,
								           long                 *     a_validTopNode,
									       long                 *     a_validBottomNode,
										   long                 *     a_2DPointto3DPoints,
								           const long           &     a_Cell,
	                                       const int            &     a_layerID,
										   const bool           &     a_topCoplane);

   void            insertPyramid(          vtkUnstructuredGrid  *    a_uGrid,
								           long                 *     a_validTopNode,
									       long                 *     a_validBottomNode,
										   long                 *     a_faceNodePtr,
										   long                 *     a_2DPointto3DPoints,
								           const long           &     a_Cell,
	                                       const int            &     a_layerID);


   void            insertWedge(            vtkUnstructuredGrid  *     a_uGrid,
								           long                 *     a_validTopNode,
									       long                 *     a_validBottomNode,
										   long                 *     a_2DPointto3DPoints,
								           const long           &     a_Cell,
	                                       const int           &     a_layerID);


   void            insertTetra(            vtkUnstructuredGrid  *     a_uGrid,
								           long                  *     a_validTopNode,
									       long                  *     a_validBottomNode,
										   long                  *     a_faceNodePtr,
										   long                  *     a_2DPointto3DPoints,
								           const long            &     a_Cell,
	                                       const int            &     a_layerID);


  bool            fourPointsCoplanar(double p1[3],
	                                 double p2[3],
									 double p3[3],
									 double p4[3]);

  void            prepare_average(int * & a_kbp, 
	                             int * & a_mapper, 
								 float * & a_zPtr,
								 const int & a_timeState, 
								 const std::string & a_horizontal_center,
							     const std::string & a_vertical_center);

  void           load_node_dry_wet(const int & a_time);
  void           load_side_dry_wet(const int & a_time);
  void           load_ele_dry_wet(const int & a_time);
  void           load_bottom(const int & a_time);


protected:
  bool         m_initialized;
  bool         m_mesh_is_static;
  bool         m_scribeIO;
  std::string  m_data_file;
  // path where  m_data_file is under
  std::string  m_data_file_path; 
  std::string  m_plugin_name;
  SCHISMFile10*   m_data_file_ptr;
  SCHISMFile10*   m_data_file_ptr2; // in scribe format, vector another component

  // element centered data use mesh from other file
  SCHISMMeshProvider10* m_external_mesh_provider;
 
  
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
  long          m_num_mesh_nodes;
  long          m_num_mesh_edges;
  long          m_num_mesh_faces;
  

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
  int       *  m_kbp00;
  int       *  m_kbp_node;
  bool         m_kbp_node_filled;
  int       *  m_kbp_ele;
  bool         m_kbp_ele_filled;
  int       *  m_kbp_side;
  bool         m_kbp_side_filled;
  int          m_cache_kbp_id;
  //  this is the kbp read from data file itself
  //  it is differnt from m_kbp00 for element/prism centred data
  int       *  m_kbp_data;

  int          m_dry_wet_flag; // 0: dry cell/ele/side filled with last wetting val, 1: junk
  int       *  m_node_dry_wet;
  int       *  m_ele_dry_wet;
  int       *  m_side_dry_wet;
  int          m_node_dry_wet_cached_time;
  int          m_side_dry_wet_cached_time;
  int          m_ele_dry_wet_cached_time;
 
  long          m_total_valid_3D_point;
  long          m_total_valid_3D_side;
  long          m_total_valid_3D_ele;
  

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


  //debug var del after done
  int m_tri_wedge;
  int m_tri_pyramid;
  int m_tri_tetra;

  int m_quad_hexhedron;
  int m_quad_wedge;
  int m_quad_pyramid;
};


#endif
