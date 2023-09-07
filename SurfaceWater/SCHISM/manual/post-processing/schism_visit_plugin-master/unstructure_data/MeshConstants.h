#ifndef _MESHcONSTANTS_H_
#define _MESHcONSTANTS_H_
#include <string>
using namespace std;

namespace MeshConstants{
const std::string      DIM_MESH_NODES("nSCHISM_hgrid_node");
const std::string      DIM_MESH_EDGES("nSCHISM_hgrid_edge");
const std::string      DIM_MESH_FACES("nSCHISM_hgrid_face");
const std::string      DIM_MESH_FACE_NODES("nMaxSCHISM_hgrid_face_nodes");
const std::string      DIM_TWO("two");
const std::string      MESH_FACE_NODES("SCHISM_hgrid_face_nodes");
const std::string      MESH_EDGE_NODES("SCHISM_hgrid_edge_nodes");
const std::string      DIM_TIME("time");
const std::string      TIME("time");
const std::string      NODE_X("SCHISM_hgrid_node_x");
const std::string      NODE_Y("SCHISM_hgrid_node_y"); 
const std::string      FACE_X("SCHISM_hgrid_face_x");
const std::string      FACE_Y("SCHISM_hgrid_face_y"); 
const std::string      EDGE_X("SCHISM_hgrid_edge_x");
const std::string      EDGE_Y("SCHISM_hgrid_edge_y"); 
const std::string      NODE_BOTTOM("node_bottom_index");
const std::string      FACE_BOTTOM("ele_bottom_index");
const std::string      EDGE_BOTTOM("edge_bottom_index");
const std::string      NODE_DEPTH("depth");
const std::string      NODE_DEPTH_LABEL("depth");
const std::string      NODE_SURFACE("eta");
const std::string      NODE_SURFACE_LABEL("surface");
const std::string      LAYER("layer");
const std::string      LEVEL("level");
const std::string      LAYER_SCOORD("sigma");
const std::string      DIM_LAYERS("nSCHISM_vgrid_layers");
const std::string      DIM_SIGMA_LAYERS("nSCHISM_sigma_layers");
const std::string      DIM_KZ_LAYERS("nSCHISM_kz_layers");
const std::string	   DIM_VAR_COMPONENT("nComponent");
const std::string      HS("h_s");
const std::string      HC("h_c");
const std::string      THETAB("theta_b");
const std::string      THETAF("theta_f");
const std::string      ZCOORD     ="zcor";
const std::string      kbp00         ="kbp00";
const std::string      CENTER("data_horizontal_center");
const std::string      NODE("node");
const std::string      ELEM("elem");
const std::string      EDGE("edge");
const std::string      LAYER_CENTER("data_vertical_center");
const std::string      FULL_LAYER("full");
const std::string      HALF_LAYER("half");
const int INVALID_NUM   = -9999;
const int MAX_NUM_NODE_PER_CELL = 4;
const float  DRY_SURFACE= -9999.0;
const float  DUMMY_ELEVATION   = 0.0;
const float  DEGENERATED_Z      = -99999.0;
};
#endif