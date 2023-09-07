
#ifndef AVTSCHISMFILEFORMATIMPL_H
#define AVTSCHISMFILEFORMATIMPL_H

#include <vector>
#include <map>
#include <vtkUnstructuredGrid.h>
#include "FileFormatFavorInterface.h"
#include "SCHISMFile.h"
#include "MeshProvider.h"


class avtSCHISMFileFormatImpl : public FileFormatFavorInterface
{
  public:
                    avtSCHISMFileFormatImpl();
  virtual           ~avtSCHISMFileFormatImpl() {;};
  static FileFormatFavorInterface * create();

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

   void           Initialize(const std::string  & a_data_file);

  
private:

  void           PopulateStateMetaData(avtDatabaseMetaData * a_metaData, 
	                                    avtSCHISMFileFormat * a_avtFile,
                                       int                   a_timeState);

  void           addFaceCenterData(avtDatabaseMetaData * a_metaData,
	                               SCHISMVar            * a_varPtr,
								    avtSCHISMFileFormat * a_avtFile,
								   const std::string   & a_varName,
								   const std::string   & a_varLabel,
								   const avtCentering  & a_center);

  void           addNodeCenterData(avtDatabaseMetaData * a_metaData,
	                               SCHISMVar            * a_varPtr,
								    avtSCHISMFileFormat * a_avtFile,
								   const std::string   & a_varName,
								   const std::string   & a_varLabel,
								   const avtCentering  & a_center);

  void           addSideCenterData(avtDatabaseMetaData * a_metaData,
	                               SCHISMVar            * a_varPtr,
								    avtSCHISMFileFormat * a_avtFile,
								   const std::string   & a_varName,
								   const std::string   & a_varLabel,
								    const avtCentering  & a_center);

  void           create2DUnstructuredMesh( vtkUnstructuredGrid *a_uGrid,
	                                       int                 *a_meshEle,
										   const  int          &a_timeState);

  void           create3DUnstructuredMesh(vtkUnstructuredGrid *a_uGrid,
	                                      int                 *a_meshEle,
										  int                 *a_2DPointto3DPoints,
										  const  int          &a_timeState);

  void           createLayerMesh(vtkUnstructuredGrid *a_uGrid,
	                             int                 *a_meshEle,
								 int                 *a_2DPointto3DPoints,
							     const  int          &a_timeState);

  void           create2DPointMesh( vtkUnstructuredGrid *a_uGrid,
	                                int                 *a_meshEle,
								    const  int          &a_timeState);

  void           create3DPointMesh( vtkUnstructuredGrid *a_uGrid,
	                                int                 *a_meshEle,
								    const  int          &a_timeState);

   void          create3DPointFaceMesh( vtkUnstructuredGrid *a_uGrid,
	                                    int                 *a_meshEle,
								        const  int          &a_timeState);

  


  vtkDataArray*   getLayer();



  void           PopulateVarMap();
  void           getTime();

  

  void           getMeshDimensions(MeshProvider * a_meshProviderPtr);

  void           loadMeshCoordinates(MeshProvider * a_meshProviderPtr);

   

  void           updateMeshZCoordinates(vtkPoints  * a_pointSet,
                                        const int    a_timeState,
                                        const char * a_meshName);
  
  void           retrieve1DArray(float             * a_valBuff,
	                             SCHISMFile         * a_selfeFilePtr,
                                 const std::string & a_varToken,
                                 const int     & a_numVar) const;
  void           retrieve1DArray(int           * a_valBuff,
	                             SCHISMFile     * a_selfeFilePtr,
                                 const std::string & a_varToken,
                                 const int     & a_numVar) const;

  void            loadAndCacheZ(const int& a_timeState,float * a_cache);
  void            loadAndCacheZSide(const int& a_timeState,float * a_cache);
  void            loadAndCacheZEle(const int& a_timeState,float * a_cache);

  void            getSingleLayerVar(float   *          a_valBuff,
	                                SCHISMFile*         a_selfeOutPtr,
                                    const int&         a_timeState,
                                    const std::string& a_varname) const;

  void            depthAverage(float        *  a_averageState,
                               float        *  a_layeredState,
							   int          *  a_nodeDataStart,
                               const int    &  a_timeState
                               ) ;

  void            vectorDepthAverage(float         *  a_averageState,
                                     float         *  a_layeredState,
									 int           *  a_nodeDataStart,
                                     const int     &  a_timeState,
                                     const int     &  a_ncomps
                                    ) ;


  bool           SCHISMVarIs3D(SCHISMVar*  a_varPtr ) const;
  bool           SCHISMVarIsVector(SCHISMVar* a_varPtr) const;
  
 

  void           validTopBottomNode(int       &   a_validTopNodeNum,
	                                int       &   a_validBottomNodeNum,
									int       *   a_validTopNode,
									int       *   a_validBottomNode,
									const int &   a_layerID,
									int*          a_faceNodePtr) const;

  void            insertTriangle3DCell(vtkUnstructuredGrid * a_uGrid,
	                                   const int           & a_validTopNodeNum,
									   const int           & a_validBottomNodeNum,
									         int           * a_validTopNode,
											 int           * a_validBottomNode,
											 int           * a_faceNodePtr,
											 int           * a_2DPointto3DPoints,
									    const int          & a_Cell,
	                                   const int           & a_layerID);

  void            insertQuad3DCell(vtkUnstructuredGrid *     a_uGrid,
	                               const int           &     a_validTopNodeNum,
								   const int           &     a_validBottomNodeNum,
									     int           *     a_validTopNode,
										 int           *     a_validBottomNode,
										 int           *     a_faceNodePtr,
										 int           *     a_2DPointto3DPoints,
								   const int           &     a_Cell,
	                               const int           &     a_layerID);

   void            insert8NodesPolyhedron(vtkUnstructuredGrid  *     a_uGrid,
	                                       vtkIdType           *     a_verts,
								           int                 *     a_validTopNode,
									       int                 *     a_validBottomNode,
										   int                 *     a_2DPointto3DPoints,
								           const int           &     a_Cell,
	                                       const int           &     a_layerID,
										   const bool          &     a_bottomCoplane,
										   const bool          &     a_topCoplane);

   void            insert7NodesPolyhedron(vtkUnstructuredGrid  *     a_uGrid,
	                                       vtkIdType           *     a_verts,
								           int                 *     a_validTopNode,
									       int                 *     a_validBottomNode,
										   int                 *     a_2DPointto3DPoints,
								           const int           &     a_Cell,
	                                       const int           &     a_layerID,
										   const bool          &     a_topCoplane);

   void            insertPyramid(          vtkUnstructuredGrid  *    a_uGrid,
								           int                 *     a_validTopNode,
									       int                 *     a_validBottomNode,
										    int                *     a_faceNodePtr,
										   int                 *     a_2DPointto3DPoints,
								           const int           &     a_Cell,
	                                       const int           &     a_layerID);


   void            insertWedge(            vtkUnstructuredGrid  *     a_uGrid,
								           int                 *     a_validTopNode,
									       int                 *     a_validBottomNode,
										   int                 *     a_2DPointto3DPoints,
								           const int           &     a_Cell,
	                                       const int           &     a_layerID);


   void            insertTetra(            vtkUnstructuredGrid  *     a_uGrid,
								           int                  *     a_validTopNode,
									       int                  *     a_validBottomNode,
										   int                  *     a_faceNodePtr,
										   int                  *     a_2DPointto3DPoints,
								           const int            &     a_Cell,
	                                       const int            &     a_layerID);


  bool            fourPointsCoplanar(double p1[3],
	                                 double p2[3],
									 double p3[3],
									 double p4[3]);

  void            prepare_average(int * & a_kbp, int * & a_mapper, float * & a_zPtr, const int & a_timeState);


private:
  bool         m_initialized;
  std::string  m_data_file;
  // path where  m_data_file is under
  std::string  m_data_file_path; 
  std::string  m_plugin_name;
  SCHISMFile*   m_data_file_ptr;
  

  // element centered data use mesh from other file
  MeshProvider* m_external_mesh_provider;
 
  
  // a number of token of saved vars and attributes
  std::string  m_data_description;
  std::string  m_mesh_var;
  std::string  m_var_label_att;
  std::string  m_var_location_att; 
  
  // name for meshes
  std::string  m_mesh_3d;
  std::string  m_layer_mesh;
  std::string  m_mesh_2d;
  std::string  m_side_center_point_3d_mesh;
  std::string  m_side_center_point_2d_mesh;
  // vertical side faces center points
  std::string  m_face_center_point_3d_mesh;
  
  // dimension size of single layer of 2D Mesh 
  int          m_num_mesh_nodes;
  int          m_num_mesh_edges;
  int          m_num_mesh_faces;
  

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
  //  this is the kbp read from data file itself
  //  it is differnt from m_kbp00 for element/prism centred data
  int       *  m_kbp_data;
  int          m_total_valid_3D_point;
  int          m_total_valid_3D_side;
  int          m_total_valid_3D_ele;
  

  std::string  m_surface_state_suffix;
  std::string  m_bottom_state_suffix;
  std::string  m_depth_average_suffix;

  //map convert netcdf centering into avt centering
  std::map<std::string, avtCentering> m_center_map;

  //map convert var label into netcdf varname token
  std::map<std::string, std::string>  m_var_name_label_map;


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
