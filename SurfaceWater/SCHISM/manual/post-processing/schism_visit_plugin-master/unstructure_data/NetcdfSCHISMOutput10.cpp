
#include "NetcdfSCHISMOutput10.h"
#include "SchismGeometry10.h"
#include "ncFloat.h"
#include "ncDouble.h"
#include "ncDouble.h"
#include "ncInt.h"
#include "ncShort.h"
#include "ncChar.h"

#include <sstream>
#include <vector>
#include <algorithm> 
#include <cmath>
#include <map>



std::map<int, std::string> i23d_horizontal_center;
typedef pair<int, std::string> IS_Pair;

//i23d_horizontal_center={ {1,MeshConstants10::NODE}, {2,MeshConstants10::NODE}, {3,MeshConstants10::NODE}, \
//                                                   {4, MeshConstants10::ELEM}, {5,MeshConstants10::ELEM }, {6,MeshConstants10::ELEM},\
//                                                   {7, MeshConstants10::EDGE}, {8,MeshConstants10::EDGE}, {9,MeshConstants10::EDGE},};

std::map<int, std::string> i23d_vertical_center;
//{ {1,MeshConstants10::FULL_LAYER}, {2,MeshConstants10::FULL_LAYER}, {3,MeshConstants10::HALF_LAYER},\
//                                                {4, MeshConstants10::FULL_LAYER}, {5,MeshConstants10::FULL_LAYER }, {6,MeshConstants10::HALF_LAYER },\
//                                                {7, MeshConstants10::FULL_LAYER}, {8,MeshConstants10::FULL_LAYER }, {9,MeshConstants10::HALF_LAYER }, };



NetcdfSchismOutput10::NetcdfSchismOutput10(const std::string a_outputFile):SCHISMFile10(a_outputFile),
	                                                                m_face_bottom(NULL),
                                                                    m_node_bottom(NULL),
                                                                    m_edge_bottom(NULL),
                                                                    m_face_bottom_time_id(-1),
                                                                    m_node_bottom_time_id(-1),
                                                                    m_edge_bottom_time_id(-1)
{
	m_outputNcFilePtr=new NcFile(a_outputFile.c_str(),NcFile::ReadOnly);
	if (m_outputNcFilePtr->is_valid()==0)
	{
		m_is_valid=false;
		throw SCHISMFileException10( a_outputFile+" is not a valid NC file\n");
	}
	else
	{
		m_is_valid=true;
	}

	std::size_t found3 = a_outputFile.find("schout");
	if (found3 != std::string::npos)
	{
		std::string source_id = "source";
		bool find_source_att = false;
		for (int i = 0; i < m_outputNcFilePtr->num_atts(); i++)
		{
			NcAtt * temp_ptr = m_outputNcFilePtr->get_att(i);
			std::string att_id = temp_ptr->name();
			if (att_id == source_id)
			{
				find_source_att = true;
				std::string source = temp_ptr->as_string(0);
				if ((source.find("SCHISM model output")) == std::string::npos)
				{
					throw SCHISMFileException10(a_outputFile + "is not a valid SCHISM NC output file\n");
				}
			}
		}
	}


	//if (!(find_source_att))
	//{
	//	throw SCHISMFileException10( a_outputFile+"is not a valid SCHSIM NC output file\n");
	//}


	
	i23d_horizontal_center.insert(IS_Pair(1,MeshConstants10::NODE));
	i23d_horizontal_center.insert(IS_Pair(2,MeshConstants10::NODE));
	i23d_horizontal_center.insert(IS_Pair(3,MeshConstants10::NODE));
	i23d_horizontal_center.insert(IS_Pair(4,MeshConstants10::ELEM));
	i23d_horizontal_center.insert(IS_Pair(5,MeshConstants10::ELEM));
	i23d_horizontal_center.insert(IS_Pair(6,MeshConstants10::ELEM));
	i23d_horizontal_center.insert(IS_Pair(7,MeshConstants10::EDGE));
	i23d_horizontal_center.insert(IS_Pair(8,MeshConstants10::EDGE));
	i23d_horizontal_center.insert(IS_Pair(9,MeshConstants10::EDGE));

	i23d_vertical_center.insert(IS_Pair(1,MeshConstants10::FULL_LAYER));
	i23d_vertical_center.insert(IS_Pair(2,MeshConstants10::FULL_LAYER));
	i23d_vertical_center.insert(IS_Pair(4,MeshConstants10::FULL_LAYER));
	i23d_vertical_center.insert(IS_Pair(5,MeshConstants10::FULL_LAYER));
	i23d_vertical_center.insert(IS_Pair(7,MeshConstants10::FULL_LAYER));
	i23d_vertical_center.insert(IS_Pair(8,MeshConstants10::FULL_LAYER));
	i23d_vertical_center.insert(IS_Pair(3,MeshConstants10::FULL_LAYER));
	i23d_vertical_center.insert(IS_Pair(6,MeshConstants10::FULL_LAYER));
	i23d_vertical_center.insert(IS_Pair(9,MeshConstants10::FULL_LAYER));

	try
	{
	this->load_dim_var();
	}
	catch( const std::invalid_argument& e )
	{
         throw SCHISMFileException10(a_outputFile + "has long int attribute which is not supported by current netcdf c++ lib\n");
     }
	this->fill_node_bottom();
    this->fill_edge_bottom();
	this->fill_ele_bottom();
	
}

NetcdfSchismOutput10::~NetcdfSchismOutput10()
{
 
   if(m_total_num_vars)
	{
		for(int ivar=0;ivar<m_total_num_vars;ivar++)
		{
			delete m_variables[ivar];
		}
		delete [] m_variables;
	}
	if(m_total_num_dims)
	{
		for(int idim=0;idim<m_total_num_dims;idim++)
		{
			delete m_dimensions[idim];
		}
		delete [] m_dimensions;
	}

	if (m_face_bottom)
	{
		delete m_face_bottom;
	 }
    if (m_node_bottom)
	{
		delete m_node_bottom;
	}
  
	if(m_edge_bottom)
	{
		delete m_edge_bottom;
	}
 
	close();
	
}
void  NetcdfSchismOutput10::close()
{
  
 if((m_outputNcFilePtr)&&(m_outputNcFilePtr->is_valid()))
 {
   m_outputNcFilePtr->close();
 }
}
int    NetcdfSchismOutput10::get_dry_wet_val_flag()// 0: filled with last wetting val 1: junk
{
	if (!(inquire_var(MeshConstants10::DRY_VALUE_FLAG.c_str())))
	{
		return 1;
	}
	try
	{
	  NcVar * ncvar = m_outputNcFilePtr->get_var(MeshConstants10::DRY_VALUE_FLAG.c_str());
	  int val;
	  int count[1];
	  count[0]=1;
	  ncvar->get(&val,count[0]);
      return val;
	}
	catch(...) 
	{
       return 1;
    }
}

int  NetcdfSchismOutput10::global_att_as_int(const std::string& a_att_name) const
{
	int num_att = m_outputNcFilePtr->num_atts();
	for(int i=0;i<num_att;i++)
	{
		NcAtt* a_att = m_outputNcFilePtr->get_att(i);
		std::string att_name =a_att->name();
		if (att_name==a_att_name)
		{
			return a_att->as_int(0);
		}
	}
	
	return 0;
}

std::string NetcdfSchismOutput10::global_att_as_string(const std::string& a_att_name) const
{
	int num_att = m_outputNcFilePtr->num_atts();
	for(int i=0;i<num_att;i++)
	{
		NcAtt* a_att = m_outputNcFilePtr->get_att(i);
		std::string att_name =a_att->name();
		if (att_name==a_att_name)
		{
			return a_att->as_string(0);
		}
	}
	
	return "";
}



void   NetcdfSchismOutput10::fill_node_bottom()
{
	std::string node_bottom_name = MeshConstants10::NODE_BOTTOM;
	if (has_var("bottom_index_node"))
	{
		node_bottom_name = "bottom_index_node";
	}
	if (has_var(node_bottom_name))
	{
		NcVar * ncvar = m_outputNcFilePtr->get_var(node_bottom_name.c_str());
		if (ncvar->is_valid())
		{
			NcDim * dimNodePtr = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_NODES.c_str());
			long numMeshNodes = 0;
			numMeshNodes = dimNodePtr->size();
			NcDim * dimLayerPtr = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_LAYERS.c_str());
			long nLayers = 0;
			nLayers = dimLayerPtr->size();
			if (!(m_node_bottom))
			{
				m_node_bottom = new int[numMeshNodes];
			}

			for (long i = 0; i < numMeshNodes; i++)
			{
				m_node_bottom[i] = nLayers + 1;
			}


			long count[1];
			count[0] = numMeshNodes;
			ncvar->get(m_node_bottom, count[0]);
            
			//fill should be called for first step
			//m_node_bottom_time_id = 0;
		}
	}
}
void   NetcdfSchismOutput10::fill_edge_bottom()
{
	if (has_var(MeshConstants10::EDGE_BOTTOM))
	{
		NcVar * ncvar = m_outputNcFilePtr->get_var(MeshConstants10::EDGE_BOTTOM.c_str());
		if (ncvar->is_valid())
		{
			NcDim * dimFacePtr = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_FACES.c_str());
			long numMeshFaces = 0;
			numMeshFaces = dimFacePtr->size();
			NcDim * dimNodePtr = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_NODES.c_str());
			long numMeshNodes = 0;
			numMeshNodes = dimNodePtr->size();

			NcDim * dimEdgePtr = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_EDGES.c_str());
			long numMeshEdges = 0;
			numMeshEdges = dimEdgePtr->size();

			if (!(m_edge_bottom))
			{
				m_edge_bottom = new int[numMeshEdges];
			}

			long count[1];
			count[0] = numMeshEdges;
			ncvar->get(m_edge_bottom, count[0]);
		}
	}
	else if(m_node_bottom) // infer from node bottom if node bottom filled
	{
		if (has_var(MeshConstants10::MESH_FACE_NODES))
		{
			SCHISMVar10 * ncFacePtr = this->get_var(MeshConstants10::MESH_FACE_NODES);
			NcDim * dimFacePtr = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_FACES.c_str());
			long numMeshFaces = 0;
			numMeshFaces = dimFacePtr->size();
			long *  faceNodesPtr = new long[numMeshFaces*(MeshConstants10::MAX_NUM_NODE_PER_CELL + 1)];
			if (!ncFacePtr->get(faceNodesPtr))
			{
				throw SCHISMFileException10("fail to read mesh element in " + this->file());
			}
			NcDim * dimNodePtr = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_NODES.c_str());
			long numMeshNodes = 0;
			numMeshNodes = dimNodePtr->size();
			long numSide = meshSideNum(faceNodesPtr, numMeshFaces, numMeshNodes);

			long* sideNodes = new long[numSide * 2];
			meshSideNode(sideNodes, faceNodesPtr, numSide, numMeshFaces, numMeshNodes);

			if (!(m_edge_bottom))
			{
				m_edge_bottom = new int[numSide];
			}

			for (long iedge = 0; iedge < numSide; iedge++)
			{
				long node1 = sideNodes[iedge * 2];
				long node2 = sideNodes[iedge * 2 + 1]; //node id is 1 based
				m_edge_bottom[iedge] = m_node_bottom[node1 - 1];
				if (m_node_bottom[node2 - 1] < m_node_bottom[node1 - 1])
				{
					m_edge_bottom[iedge] = m_node_bottom[node2 - 1];
				}

			}

			delete faceNodesPtr;
			delete sideNodes;
		}
	}
}
void   NetcdfSchismOutput10::fill_ele_bottom()
{
	if (has_var(MeshConstants10::FACE_BOTTOM))
	{
		NcVar * ncvar = m_outputNcFilePtr->get_var(MeshConstants10::FACE_BOTTOM.c_str());
		if (ncvar->is_valid())
		{
			NcDim * dimFacePtr = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_FACES.c_str());
			long numMeshFaces = 0;
			numMeshFaces = dimFacePtr->size();

			if (!(m_face_bottom))
			{
				m_face_bottom = new int[numMeshFaces];
			}


			long count[1];
			count[0] = numMeshFaces;
			ncvar->get(m_face_bottom, count[0]);
		}
	}
	else if (m_node_bottom) // infer from node bottom if node bottom filled
	{
		if (has_var(MeshConstants10::MESH_FACE_NODES))
		{
			SCHISMVar10 * ncFacePtr = this->get_var(MeshConstants10::MESH_FACE_NODES);
			NcDim * dimFacePtr = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_FACES.c_str());
			long numMeshFaces = 0;
			numMeshFaces = dimFacePtr->size();
			long *  faceNodesPtr = new long[numMeshFaces*(MeshConstants10::MAX_NUM_NODE_PER_CELL + 1)];
			if (!ncFacePtr->get(faceNodesPtr))
			{
				throw SCHISMFileException10("fail to read mesh element in " + this->file());
			}
			NcDim * dimNodePtr = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_NODES.c_str());
			long numMeshNodes = 0;
			numMeshNodes = dimNodePtr->size();
			

			if (!(m_face_bottom))
			{
				m_face_bottom = new int[numMeshFaces];
			}

			for (long iface = 0; iface < numMeshFaces; iface++)
			{
				int num_point_in_face = faceNodesPtr[iface*(MeshConstants10::MAX_NUM_NODE_PER_CELL + 1)];
				int min_node_bottom = MeshConstants10::BIG_LEVEL;
				for (int inode = 0; inode < num_point_in_face; inode++)
				{
					long node_id = faceNodesPtr[iface*(MeshConstants10::MAX_NUM_NODE_PER_CELL + 1) + inode + 1]; //node id is 1 based
					int node_bottom = m_node_bottom[node_id - 1];
					if (node_bottom < min_node_bottom)
					{
						min_node_bottom = node_bottom;
					}
				}
				m_face_bottom[iface] = min_node_bottom;

			}

			delete faceNodesPtr;
			
		}
	}
}
void   NetcdfSchismOutput10::get_node_bottom(int* a_node_bottom,const int& a_time)
{
	if ((m_node_bottom)&&(a_node_bottom))
	{
		NcDim * dimNodePtr = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_NODES.c_str());
		long numMeshNodes = 0;
		numMeshNodes = dimNodePtr->size();

		for (long i = 0; i < numMeshNodes; i++)
		{
			a_node_bottom[i] = m_node_bottom[i];
		}
		
	}


}
void   NetcdfSchismOutput10::get_face_bottom(int* a_face_bottom,const int& a_time)
{
	if ((m_node_bottom)&&(a_face_bottom))
	{
		NcDim * dimFacePtr = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_FACES.c_str());
		long numMeshFaces = 0;
		numMeshFaces = dimFacePtr->size();

		for (long i = 0; i < numMeshFaces; i++)
		{
			a_face_bottom[i] = m_face_bottom[i];
		}
	}

}
void  NetcdfSchismOutput10::get_edge_bottom(int* a_edge_bottom,const int& a_time)
{
	if ((m_node_bottom)&&(a_edge_bottom))
	{
		NcDim * dimEdgePtr = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_EDGES.c_str());
		long numMeshEdges = 0;
		numMeshEdges = dimEdgePtr->size();

		for (long i = 0; i < numMeshEdges; i++)
		{
			a_edge_bottom[i] = m_edge_bottom[i];
		}
	}

}



void  NetcdfSchismOutput10::set_mesh_bottom(SCHISMFile10 * a_ptr, const int& a_time)
{
	if (!m_node_bottom)
	{
		NcDim * dimNodePtr = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_NODES.c_str());
		long numMeshNodes = 0;
		numMeshNodes = dimNodePtr->size();
		m_node_bottom = new int[numMeshNodes];
		a_ptr->get_node_bottom(m_node_bottom, a_time);
	}
	

	if (!m_face_bottom)
	{
		NcDim * dimFacePtr = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_FACES.c_str());
		long numMeshFaces = 0;
		numMeshFaces = dimFacePtr->size();
		m_face_bottom = new int[numMeshFaces];
		a_ptr->get_face_bottom(m_face_bottom, a_time);
	}
	
	if (!m_edge_bottom)
	{
		NcDim * dimEdgePtr = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_EDGES.c_str());
		long numMeshEdges = 0;
		numMeshEdges = dimEdgePtr->size();
		m_edge_bottom = new int[numMeshEdges];
		a_ptr->get_edge_bottom(m_edge_bottom, a_time);
	}
	
}

 bool    NetcdfSchismOutput10::update_bottom_index(const int& a_time)
 {
	 bool bottom_changed =false;
	 if (m_node_bottom)
	 {
		 bottom_changed = update_node_bottom(a_time);
		 if (bottom_changed)
		 {
			 update_ele_bottom(a_time, m_node_bottom);
			 update_edge_bottom(a_time, m_node_bottom);
		 }
	 }
	 return bottom_changed;
 }
bool   NetcdfSchismOutput10::update_node_bottom(const int& a_time)
{
	if (m_node_bottom)
	{
		NcDim * dimNodePtr = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_NODES.c_str());
		long numMeshNodes = 0;
		numMeshNodes = dimNodePtr->size();
		NcDim * dimLayerPtr = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_LAYERS.c_str());
		long nLayers = 0;
		nLayers = dimLayerPtr->size();

		NcVar * ncvar = m_outputNcFilePtr->get_var(MeshConstants10::ZCOORD.c_str());

		float missing_val = (ncvar->get_att("missing_value"))->as_float(0);

		float * zcor = new float[numMeshNodes*nLayers];

		long current[3];
		current[0] = a_time;
		current[1] = 0;
		current[2] = 0;
		long count[3];
		count[0] = 1;
		count[1] = numMeshNodes;
		count[2] = nLayers;

		ncvar->set_cur(current);
		ncvar->get(zcor, count[0], count[1], count[2]);
		m_node_bottom_time_id = a_time;

		int  bottom_id = nLayers + 1;
		bool bottom_changed = false;


		for (long i = 0; i < numMeshNodes; i++)
		{
			bottom_id = nLayers + 1;
			for (int ilevel = 0; ilevel < nLayers; ilevel++)
			{
				if (zcor[i*nLayers + ilevel] != missing_val)
				{
					bottom_id = ilevel + 1;
					break;
				}
			}

			if (bottom_id != m_node_bottom[i])
			{
				bottom_changed = true;
				m_node_bottom[i] = bottom_id;
			}

		}
		delete zcor;
		return bottom_changed;
	}
	else
	{
		return false;
	}
}
void   NetcdfSchismOutput10::update_edge_bottom(const int& a_time, int* a_node_bottom)
{
	
		NcDim * dimFacePtr = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_FACES.c_str());
		long numMeshFaces = 0;
		numMeshFaces = dimFacePtr->size();
		NcDim * dimNodePtr = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_NODES.c_str());
		long numMeshNodes = 0;
		numMeshNodes = dimNodePtr->size();

		NcDim * dimEdgePtr = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_EDGES.c_str());
		long numMeshEdges = 0;
		numMeshEdges = dimEdgePtr->size();


		SCHISMVar10 * edge_var_ptr = this->get_var(MeshConstants10::MESH_EDGE_NODES);
		long *  nodes = new long[numMeshEdges * 2];

		edge_var_ptr->get(nodes);

		if (!(m_edge_bottom))
		{
			m_edge_bottom = new int[numMeshEdges];
		}

		for (long iedge = 0; iedge < numMeshEdges; iedge++)
		{
			long node1 = nodes[iedge * 2];
			long node2 = nodes[iedge * 2 + 1]; //node id is 1 based
			m_edge_bottom[iedge] = a_node_bottom[node1 - 1];
			if (a_node_bottom[node2 - 1] < a_node_bottom[node1 - 1])
			{
				m_edge_bottom[iedge] = a_node_bottom[node2 - 1];
			}

		}

		m_edge_bottom_time_id = a_time;
		delete nodes;
	
}
void   NetcdfSchismOutput10::update_ele_bottom(const int& a_time,int* a_node_bottom)
{
  
  NcDim * dimFacePtr      = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_FACES.c_str());
  long numMeshFaces=0;
  numMeshFaces = dimFacePtr->size(); 
  NcDim * dimNodePtr      = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_NODES.c_str());
  long numMeshNodes=0;
  numMeshNodes = dimNodePtr->size(); 
  SCHISMVar10 * mesh_var_ptr=this->get_var( MeshConstants10::MESH_FACE_NODES);
  long *  nodes = new long [numMeshFaces*(MeshConstants10::MAX_NUM_NODE_PER_CELL+1)];

  mesh_var_ptr->get(nodes);

  if(!(m_face_bottom))
  {
	  m_face_bottom=new int[numMeshFaces];
  }

  for(long iface=0;iface<numMeshFaces;iface++)
  {
	  int num_point_in_face = nodes[iface*(MeshConstants10::MAX_NUM_NODE_PER_CELL+1)];
	  int min_node_bottom= MeshConstants10::BIG_LEVEL;
	  for(int inode=0;inode<num_point_in_face;inode++)
	  {
		  long node_id = nodes[iface*(MeshConstants10::MAX_NUM_NODE_PER_CELL+1)+inode+1]; //node id is 1 based
		  int node_bottom=a_node_bottom[node_id-1];
		  if (node_bottom<min_node_bottom)
		  {
			  min_node_bottom=node_bottom;
		  }
	  }
	  m_face_bottom[iface]=min_node_bottom;

  }
  m_face_bottom_time_id=a_time;
  delete nodes;
}

bool  NetcdfSchismOutput10::has_var(const std::string& a_var_name) const
{
	bool found = false;
	for (int i = 0; i < m_total_num_vars; i++)
	{
		if ((m_variables[i]->name())== a_var_name)
		{
			found = true;
			break;
		}
	}
	return found;
}
bool NetcdfSchismOutput10::inquire_var(const std::string& a_var_name) const
{
	return has_var(a_var_name);
}
bool  NetcdfSchismOutput10::load_dim_var()
{
	int kz = 1;
	
   
	int num_dim = m_outputNcFilePtr->num_dims();
	m_dimensions = new SCHISMDim10 * [num_dim];
	m_total_num_dims = num_dim;
	for(int idim=0;idim<num_dim;idim++)
	{
		NcDim * a_dim = m_outputNcFilePtr->get_dim(idim);
		
		SCHISMDim10 * wrapped_dim;

		if (a_dim->name() == MeshConstants10::DIM_MESH_FACE_NODES)
		{
		  wrapped_dim = newSCHISMDim(a_dim->name(),
			                                   idim,
											   a_dim->size()+1);
		}
		else
		{
		  wrapped_dim = newSCHISMDim(a_dim->name(),
			                                   idim,
											   a_dim->size());
		}
		m_dimensions[idim]= wrapped_dim;
	}
	

	std::map<std::string, int>  var_name_added;

	int num_var = m_outputNcFilePtr->num_vars();
	 m_variables  = new SCHISMVar10 * [num_var];
	for(int ivar=0;ivar<num_var;ivar++)
	{
		NcVar * a_var =m_outputNcFilePtr->get_var(ivar);
		std::string var_name = a_var->name();
		NetcdfSchismOutputVar10 * schism_var        = new NetcdfSchismOutputVar10(var_name);
		schism_var->fill_ncVar(a_var);
		schism_var->m_schismfilePtr = this;
		int num_dim_var = a_var->num_dims();
		
		// filled global dim id in ncfile
		for(int i_dim_var=0;i_dim_var<num_dim_var;i_dim_var++)
		{
			NcDim * dim_var = a_var->get_dim(i_dim_var);
			std::string dim_name = dim_var->name();
			for(int i_dim_file=0;i_dim_file<num_dim;i_dim_file++)
			{
				if(m_dimensions[i_dim_file]->name() == dim_name)
				{
					schism_var->add_dimension(i_dim_file);
					break;
				}
			}
		}

		// fill att
		int num_att_var = a_var->num_atts();
		
		for(int i_att_var=0;i_att_var<num_att_var;i_att_var++)
		{
			NcAtt * att_var = a_var->get_att(i_att_var);
		    SCHISMAtt10 * schism_att = new SCHISMAtt10(att_var->name());
			NcType type = att_var->type();
			if (type == ncChar) 
			{
				schism_att->add_string_value(att_var->as_string(0));

			}
			else if ((type == ncShort) || (type ==ncInt))
			{
				schism_att->add_int_value(att_var->as_int(0));
			}
			else if ((type==ncLong))
			{
				
				throw std::invalid_argument( "current version of nectd c++ api doesn't support long int attribute!");
			}
			else if (type==ncFloat)
			{
				schism_att->add_float_value(att_var->as_float(0));
			}
			else if (type==ncDouble)
			{
				schism_att->add_double_value(att_var->as_double(0));
			}
			schism_var->add_att(schism_att);

			std::string nc_att_name(att_var->name());

			//if ((!(nc_att_name.compare(MeshConstants10::CENTER)))||(!(nc_att_name.compare(MeshConstants10::LOCATION))))
			//{
			//	schism_var->set_horizontal_center((att_var->as_string(0)));
			//}
			//else if (!(nc_att_name.compare(MeshConstants10::LAYER_CENTER)))
			//{
			//	schism_var->set_vertical_center(att_var->as_string(0));
			//}

			if (!(nc_att_name.compare(MeshConstants10::i23d)))
			{
	
				int i23d = att_var->as_int(0);
				schism_var->set_horizontal_center(i23d_horizontal_center[i23d]);
				schism_var->set_vertical_center(i23d_vertical_center[i23d]);
			}
		
		}

		if ( (schism_var->name()) == MeshConstants10::MESH_FACE_NODES)
		{
			cache_face_nodes(schism_var);
		}

		int var_id            = m_total_num_vars;
        m_variables[var_id]   = schism_var;
		var_name_added[var_name]=var_id;
        m_total_num_vars++; 
	}
	return true;
}

void   NetcdfSchismOutput10::set_mesh_data_ptr(SCHISMFile10* a_ptr)
{
	m_meshFilePtr = a_ptr;
}
 bool  NetcdfSchismOutput10::cache_face_nodes(SCHISMVar10 * mesh_var)
 {
  
  
  long numMeshFaces = MeshConstants10::INVALID_NUM;
  
  NcDim * dimFacePtr      = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_FACES.c_str());
  
  if (dimFacePtr->is_valid())
    {
      numMeshFaces = dimFacePtr->size(); 
    }
  else
    {
      throw SCHISMFileException10("no element dim in data file "+ m_SCHISMOutputFile);
    }
	 
  long *  faceNodesPtr           = new long  [numMeshFaces*MeshConstants10::MAX_NUM_NODE_PER_CELL];
  NcVar * ncFacePtr = m_outputNcFilePtr->get_var(MeshConstants10::MESH_FACE_NODES.c_str());
  if (!(ncFacePtr->is_valid()))
    {
      
      throw SCHISMFileException10("No face node in data file "+ m_SCHISMOutputFile);
    }
 

  if (!ncFacePtr->get(faceNodesPtr,numMeshFaces, MeshConstants10::MAX_NUM_NODE_PER_CELL))
    {

      throw SCHISMFileException10("fail to load face node in data file "+ m_SCHISMOutputFile);;
    }

  long *  nodes = new long [numMeshFaces*(MeshConstants10::MAX_NUM_NODE_PER_CELL+1)];

  long index1 = 0;
  long index2 = 0;
  for(long iCell=0;iCell<numMeshFaces;iCell++)
  {
	  index1 = iCell*MeshConstants10::MAX_NUM_NODE_PER_CELL;
	  index2 = iCell*(MeshConstants10::MAX_NUM_NODE_PER_CELL+1);
	  int num_node_in_cell = 0;

	  for(int i=0;i<MeshConstants10::MAX_NUM_NODE_PER_CELL;i++)
	  {
		  if (faceNodesPtr[i+index1] >=0)
		  {
			  num_node_in_cell++;
			  nodes[index2+i+1]=faceNodesPtr[i+index1];
		  }
		  else
		  {
			  break;
		  }
	  }

	  
	  if ((num_node_in_cell<3)||(num_node_in_cell>4))
	  {
		  ostringstream temp;
		  temp<<"cell "<<(iCell+1)<<"is not quad or triangle shape";
		  throw SCHISMFileException10(temp.str());
	  }

	  nodes[index2]=num_node_in_cell;
  }

  mesh_var->cache_data(nodes);
  delete nodes;
  return true;
}

 

NetcdfSchismOutputVar10::~NetcdfSchismOutputVar10()
{

}

bool  NetcdfSchismOutputVar10::put_a_float(const float&    a_value,
                                int  *    a_dim1Loc)
{
	return false;
}

void  NetcdfSchismOutputVar10::fill_ncVar(NcVar * a_nc_var)
{
	m_ncVar[0] = a_nc_var;
	m_num_component=1;
	
	for(int iatt=0;iatt<a_nc_var->num_atts();iatt++)
	{
		NcAtt* att_ptr=0;

		att_ptr =a_nc_var->get_att(iatt);

		string att_name = att_ptr->name(); 
		
		if (!att_name.compare("ivs")) //this is the label for number of vector components
		{
			m_num_component=att_ptr->as_int(0);
			break;
		}

		delete att_ptr;
	}


}

 void      NetcdfSchismOutputVar10::fill_current_bottom(int * a_kbp00) 
 {
   if(m_horizontal_center==MeshConstants10::NODE)
   {
	    m_schismfilePtr->get_node_bottom(a_kbp00,m_current_record);
   }
   else if(m_horizontal_center==MeshConstants10::EDGE)
   {
	    m_schismfilePtr->get_edge_bottom(a_kbp00,m_current_record);
   }
   else 
   {
	    m_schismfilePtr->get_face_bottom(a_kbp00,m_current_record);
   }
 }

 bool  NetcdfSchismOutputVar10::get(float *     a_buffer,int* a_bottom)
 {

	 int dataSize = computeDataNumPerTIMEStep();

	 if (m_data_cached)
	 {
;
		 return get_float_cache(a_buffer);
	 }
	 return load_from_file<float>(a_buffer, a_bottom);
 }

 bool  NetcdfSchismOutputVar10::get(double *     a_buffer,int* a_bottom)
 {

	 int dataSize = computeDataNumPerTIMEStep();
	 return load_from_file<double>(a_buffer, a_bottom);
 }


 bool  NetcdfSchismOutputVar10::get(int *     a_buffer,int *a_bottom)
 {

	 int dataSize = computeDataNumPerTIMEStep();

	 if (m_data_cached)
	 {

		 return get_int_cache(a_buffer);
	 }

	 return load_from_file<int>(a_buffer, a_bottom);
 }

bool  NetcdfSchismOutputVar10::get(float *     a_buffer) 
{

 int dataSize = computeDataNumPerTIMEStep();

 if(m_data_cached)
 {
   //for(int idata=0;idata<dataSize;idata++)
   //{
    //  a_buffer[idata] = m_float_cache[idata];
   //}
   //return true;
   return get_float_cache(a_buffer);
 }
  return load_from_file<float>(a_buffer,NULL);
}

bool  NetcdfSchismOutputVar10::get(double *     a_buffer) 
{

 int dataSize = computeDataNumPerTIMEStep();
  return load_from_file<double>(a_buffer,NULL);
}


bool  NetcdfSchismOutputVar10::get(int *     a_buffer)
{

 int dataSize = computeDataNumPerTIMEStep();

 if(m_data_cached)
 {
   //for(int idata=0;idata<dataSize;idata++)
   //{
   //   a_buffer[idata] = m_int_cache[idata];
   //}
   //return true;
   return get_int_cache(a_buffer);
 }
  
 return load_from_file<int>(a_buffer,NULL);
}


  //  get data from cache
bool   NetcdfSchismOutputVar10::get_int_cache(int *       a_buffer)  
{
   long dataSize = computeDataNumPerTIMEStep();
   for(long idata=0;idata<dataSize;idata++)
   {
      a_buffer[idata] = m_int_cache[idata];
   }
   return true;
}

bool   NetcdfSchismOutputVar10::get_long_cache(long *       a_buffer) 
{
   long dataSize = computeDataNumPerTIMEStep();
   for(long idata=0;idata<dataSize;idata++)
   {
      a_buffer[idata] = m_long_cache[idata];
   }
   return true;
}

bool   NetcdfSchismOutputVar10::get_float_cache(float *     a_buffer) 
{
   long dataSize = computeDataNumPerTIMEStep();
   for(long idata=0;idata<dataSize;idata++)
   {
      a_buffer[idata] = m_float_cache[idata];
   }
   return true;
}



bool  NetcdfSchismOutputVar10::get(long *     a_buffer) 
{

 long dataSize = computeDataNumPerTIMEStep();

 if(m_data_cached)
 {
  // for(long idata=0;idata<dataSize;idata++)
  // {
  //    a_buffer[idata] = m_long_cache[idata];
  // }
  // return true;
	return get_long_cache(a_buffer);
 }
  
  return load_from_file<long>(a_buffer,NULL);
}

bool  NetcdfSchismOutputVar10::get(long *     a_buffer,int* a_bottom)
{

	long dataSize = computeDataNumPerTIMEStep();

	if (m_data_cached)
	{
		// for(long idata=0;idata<dataSize;idata++)
		// {
		//    a_buffer[idata] = m_long_cache[idata];
		// }
		// return true;
		return get_long_cache(a_buffer);
	}

	return load_from_file<long>(a_buffer, a_bottom);
}


void  NetcdfSchismOutputVar10::set_cur(const int& a_time_record)
{
	m_current_record = a_time_record;
}




NetcdfSchismOutputVar10::NetcdfSchismOutputVar10(const std::string& a_varName):SCHISMVar10(a_varName)
{
	  m_num_component = 0;
}
NetcdfSchismOutputVar10::NetcdfSchismOutputVar10():SCHISMVar10()
{
	  m_num_component=0;
}

int  NetcdfSchismOutputVar10::computeDataNumPerTIMEStep() const
{
	int dataSize =1;

  for(int iDim=0;iDim<m_num_dim;iDim++)
  { 
     SCHISMDim10 * aDim = m_schismfilePtr->get_dim(m_dimensions[iDim]);
  
     if(!(aDim->name()=="TIME"))
      {
        dataSize *= aDim->size();
      }
      else if (m_name.compare("TIME")==0)
     {
        dataSize *= aDim->size();
     }
    else
     {
     }
  }
  return dataSize;
}





 
   



  
