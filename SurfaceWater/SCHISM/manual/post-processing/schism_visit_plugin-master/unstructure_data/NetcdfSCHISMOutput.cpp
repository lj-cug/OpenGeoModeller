
#include "NetcdfSCHISMOutput.h"
#include "MeshConstants.h"
#include "ncFloat.h"
#include "ncDouble.h"
#include "ncDouble.h"
#include "ncInt.h"
#include "ncShort.h"
#include "ncChar.h"

#include <sstream>
#include <vector>
#include <algorithm> 


NetcdfSchismOutput::NetcdfSchismOutput(const std::string a_outputFile):SCHISMFile(a_outputFile)
{
	m_outputNcFilePtr=new NcFile(a_outputFile.c_str(),NcFile::ReadOnly);
	if (m_outputNcFilePtr->is_valid()==0)
	{
		m_is_valid=false;
	}
	else
	{
		m_is_valid=true;
	}
	m_vector_component_map["hvel_u"] = "hvel";
	m_vector_component_map["hvel_v"] = "hvel";
	m_vector_component_map["wind_u"] = "wind";
	m_vector_component_map["wind_v"] = "wind";
	m_vector_component_map["wist_x"] = "wist";
	m_vector_component_map["wist_y"] = "wist";
	m_vector_component_map["dahv_x"] = "dahv";
	m_vector_component_map["dahv_y"] = "dahv";
	m_vector_component_map["bpgr_x"] = "bpgr";
	m_vector_component_map["bpgr_y"] = "bpgr";
	m_vector_component_map["tfu1_u"] = "tfu1";
	m_vector_component_map["tfu1_v"] = "tfu1";
	m_vector_component_map["tfu2_u"] = "tfu2";
	m_vector_component_map["tfu2_v"] = "tfu2";
	
	if(m_is_valid)
	{
	   this->load_dim_var();
	}
	
}

NetcdfSchismOutput::~NetcdfSchismOutput()
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
	close();
	
}
void  NetcdfSchismOutput::close()
{
  
 if(m_outputNcFilePtr)
 {
   m_outputNcFilePtr->close();
 }
}

bool  NetcdfSchismOutput::load_dim_var()
{
	int kz = 1;

	int num_dim = m_outputNcFilePtr->num_dims();
	m_total_num_dims = num_dim;
	for(int idim=0;idim<num_dim;idim++)
	{
		NcDim * a_dim = m_outputNcFilePtr->get_dim(idim);
		
		SCHISMDim * wrapped_dim;

		if (a_dim->name() == MeshConstants::DIM_MESH_FACE_NODES)
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

	for(int ivar=0;ivar<num_var;ivar++)
	{
		NcVar * a_var =m_outputNcFilePtr->get_var(ivar);
		std::string var_name = a_var->name();
		if (m_vector_component_map.find(var_name)!=m_vector_component_map.end())
        {
           var_name = m_vector_component_map[var_name];
        }
		 
		
	    if (var_name_added.find(var_name) != var_name_added.end())
		{
			int loc = var_name_added[var_name];
			static_cast<NetcdfSchismOutputVar*>(m_variables[loc])->fill_ncVar(a_var);
			continue;
		}

		NetcdfSchismOutputVar * schism_var        = new NetcdfSchismOutputVar(var_name);
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
		    SCHISMAtt * schism_att = new SCHISMAtt(att_var->name());
			NcType type = att_var->type();
			if (type == ncChar) 
			{
				schism_att->add_string_value(att_var->as_string(0));
			}
			else if ((type == ncShort) || (type ==ncInt))
			{
				schism_att->add_int_value(att_var->as_int(0));
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
		}

		if ( (schism_var->name()) == MeshConstants::MESH_FACE_NODES)
		{
			cache_face_nodes(schism_var);
		}

		int var_id            = m_total_num_vars;
        m_variables[var_id]   = schism_var;
		var_name_added[var_name]=var_id;
        m_total_num_vars++; 
	}

	// for the vector output var add component dim
	for(int ivar=0;ivar<m_total_num_vars;ivar++)
	{
		std::string name = m_variables[ivar]->name();
		int num_com  = m_variables[ivar]->num_component();

		if(num_com>1)
		{
		   int idim = m_total_num_dims;
		   SCHISMDim * wrapped_dim = newSCHISMDim(name+"_component",
			                                      idim,
											      num_com);
		   m_dimensions[idim]= wrapped_dim;
		   m_variables[ivar]->add_dimension(idim);
		   m_total_num_dims++;
		}
	}

	//get global data horizontal and vertical centering att

	int num_global_att = m_outputNcFilePtr->num_atts();
	for(int i_att=0;i_att<num_global_att;i_att++)
	{
		NcAtt * att_global = m_outputNcFilePtr->get_att(i_att);
		if(att_global->name() == MeshConstants::CENTER)
		{
		   this->m_data_center = att_global->as_string(0);
		}
		if(att_global->name() == MeshConstants::LAYER_CENTER)
		{
			this->m_layer_center = att_global->as_string(0);
		}
	}


	return true;
}
 bool  NetcdfSchismOutput::cache_face_nodes(SCHISMVar * mesh_var)
 {
  
  
  int numMeshFaces = MeshConstants::INVALID_NUM;
  
  NcDim * dimFacePtr      = m_outputNcFilePtr->get_dim(MeshConstants::DIM_MESH_FACES.c_str());
  
  if (dimFacePtr->is_valid())
    {
      numMeshFaces = dimFacePtr->size(); 
    }
  else
    {
      throw SCHISMFileException("no element dim in data file "+ m_SCHISMOutputFile);
    }
	 
  int *  faceNodesPtr           = new int  [numMeshFaces*MeshConstants::MAX_NUM_NODE_PER_CELL];
  NcVar * ncFacePtr = m_outputNcFilePtr->get_var(MeshConstants::MESH_FACE_NODES.c_str());
  if (!(ncFacePtr->is_valid()))
    {
      
      throw SCHISMFileException("No face node in data file "+ m_SCHISMOutputFile);
    }
 

  if (!ncFacePtr->get(faceNodesPtr,numMeshFaces, MeshConstants::MAX_NUM_NODE_PER_CELL))
    {

      throw SCHISMFileException("fail to load face node in data file "+ m_SCHISMOutputFile);;
    }

  int *  nodes = new int [numMeshFaces*(MeshConstants::MAX_NUM_NODE_PER_CELL+1)];

  int index1 = 0;
  int index2 = 0;
  for(int iCell=0;iCell<numMeshFaces;iCell++)
  {
	  index1 = iCell*MeshConstants::MAX_NUM_NODE_PER_CELL;
	  index2 = iCell*(MeshConstants::MAX_NUM_NODE_PER_CELL+1);
	  int num_node_in_cell = 0;

	  for(int i=0;i<MeshConstants::MAX_NUM_NODE_PER_CELL;i++)
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
		  throw SCHISMFileException(temp.str());
	  }

	  nodes[index2]=num_node_in_cell;
  }

  mesh_var->cache_data(nodes);
  return true;
}

 

NetcdfSchismOutputVar::~NetcdfSchismOutputVar()
{

}

bool  NetcdfSchismOutputVar::put_a_float(const float&    a_value,
                                int  *    a_dim1Loc)
{
	return false;
}

void  NetcdfSchismOutputVar::fill_ncVar(NcVar * a_nc_var)
{
	if (m_num_component==MAX_VAR_COMPONENT)
	{
		 throw SCHISMFileException(m_name+" having component more than allowd");
	}

	m_ncVar[m_num_component] = a_nc_var;
	m_num_component++;
}
  
bool  NetcdfSchismOutputVar::get(int *       a_buffer) const 
{
   int dataSize = computeDataNumPerTIMEStep();
   if(m_data_cached)
   {
     for(int idata=0;idata<dataSize;idata++)
      {
        a_buffer[idata] = m_int_cache[idata];
      }
	 return true;
   }

  int num_component = m_num_component;
  long buffer_size =1;
  long node_num = 0;
  int num_layer = 1;
  NcVar * ncvar = m_ncVar[0];
  int num_dim = ncvar->num_dims();
  long * current = new long [num_dim];
  long * count   = new long [num_dim];

  for(int idim =0; idim<num_dim;idim++)
  {
	 // int dim_id = m_dimensions[idim];
	  NcDim * dim = ncvar->get_dim(idim);
	  std::string dim_name=dim->name();
	  if ( (dim_name==MeshConstants::DIM_MESH_NODES) || 
		   (dim_name==MeshConstants::DIM_MESH_EDGES) ||
		   (dim_name==MeshConstants::DIM_MESH_FACES))
	  {
		  node_num = dim->size(); 
	  }
	  if (dim_name==MeshConstants::DIM_LAYERS)
	  {
		  num_layer = dim->size();
	  }
	  if ((dim_name== MeshConstants::DIM_TIME) && (m_name != MeshConstants::TIME))
	  {
		  current[idim] = m_current_record;
		  count[idim] = 1;
	  }
	  else
	  {
		  current[idim]=0;
		  count[idim]=dim->size();
		  buffer_size*=(long)dim->size();
	  }
  }


  if (num_layer ==1) // not 3d data
  {
	for(int icom=0;icom<num_component;icom++)
	{
		int * buffer = new int [buffer_size];
		NcVar * ncvar = m_ncVar[icom];
		ncvar->set_cur(current);
		if (num_dim ==1)
		{
		  ncvar->get(buffer,count[0]);
		}
		else if (num_dim==2)
		{
		  ncvar->get(buffer,count[0],count[1]);
		}
		else if (num_dim==3)
		{
		  ncvar->get(buffer,count[0],count[1],count[2]);
		}
		else if (num_dim==4)
		{
		  ncvar->get(buffer,count[0],count[1],count[2],count[3]);
		}
		else
		{
			ncvar->get(buffer,count);
		}
		for(long id=0;id<buffer_size;id++)
		{
			a_buffer[icom+id*num_component]=buffer[id];
		}
		delete buffer;
	}
	return true;
  }


  int * bottom_layer = new int [node_num];
  fill_bottom(bottom_layer);

  long * start_loc = new long [node_num];

  for(long inode=0;inode<node_num;inode++)
  {
	  start_loc[inode]=0;
  }
  
  for(long inode=1;inode<node_num;inode++)
  {
	  int bottom = bottom_layer[inode-1];
	  long pre_len = num_component*(num_layer-bottom+1);
	  start_loc[inode]=start_loc[inode-1]+pre_len;
  }

  for(int icom=0;icom<num_component;icom++)
  {
	  int * buffer = new int [node_num*num_layer];
	  NcVar * ncvar = m_ncVar[icom];
	  ncvar->set_cur(current);
	    if (num_dim ==1)
		{
		  ncvar->get(buffer,count[0]);
		}
		else if (num_dim==2)
		{
		  ncvar->get(buffer,count[0],count[1]);
		}
		else if (num_dim==3)
		{
		  ncvar->get(buffer,count[0],count[1],count[2]);
		}
		else if (num_dim==4)
		{
		  ncvar->get(buffer,count[0],count[1],count[2],count[3]);
		}
		else
		{
			ncvar->get(buffer,count);
		}
	  long index =0;
	  for(int ilayer=0;ilayer<num_layer;ilayer++)
	  {
		  for(long inode=0;inode<node_num;inode++)
		  {
			  int bottom = bottom_layer[inode]-1;
			  if(ilayer>=bottom)
			  {
				  long loc = start_loc[inode];
				  loc = loc + (ilayer-bottom)*num_component+icom;
				  a_buffer[loc] = buffer[index];
			  }
			  index++;
		  }
	  }
	  delete buffer;
  }

  delete bottom_layer;
  delete start_loc;
  return true;
}

bool  NetcdfSchismOutputVar::get(float *     a_buffer) const 
{

 int dataSize = computeDataNumPerTIMEStep();

 if(m_data_cached)
 {
   for(int idata=0;idata<dataSize;idata++)
   {
      a_buffer[idata] = m_float_cache[idata];
   }
   return true;
 }
  
  int num_component = m_num_component;

  long node_num = 0;
  long buffer_size =1;
  int num_layer = 1;
  NcVar * ncvar = m_ncVar[0];
  int num_dim = ncvar->num_dims();
  long * current = new long [num_dim];
  long * count   = new long [num_dim];

  for(int idim =0; idim<num_dim;idim++)
  {
	 // int dim_id = m_dimensions[idim];
	  NcDim * dim = ncvar->get_dim(idim);
	  std::string dim_name=dim->name();
	  if ( (dim_name==MeshConstants::DIM_MESH_NODES) || 
		   (dim_name==MeshConstants::DIM_MESH_EDGES) ||
		   (dim_name==MeshConstants::DIM_MESH_FACES))
	  {
		  node_num = dim->size(); 
	  }
	  if (dim_name==MeshConstants::DIM_LAYERS)
	  {
		  num_layer = dim->size();
	  }
	  if ((dim_name== MeshConstants::DIM_TIME) && (m_name != MeshConstants::TIME))
	  {
		  current[idim] = m_current_record;
		  count[idim] = 1;
	  }
	  else
	  {
		  current[idim]=0;
		  count[idim]=dim->size();
		  buffer_size*=(long)dim->size();
	  }
  }


  if (num_layer ==1) // not 3d data
  {
	for(int icom=0;icom<num_component;icom++)
	{
		float * buffer = new float [buffer_size];
		NcVar * ncvar = m_ncVar[icom];
		ncvar->set_cur(current);
		if (num_dim ==1)
		{
		  ncvar->get(buffer,count[0]);
		}
		else if (num_dim==2)
		{
		  ncvar->get(buffer,count[0],count[1]);
		}
		else if (num_dim==3)
		{
		  ncvar->get(buffer,count[0],count[1],count[2]);
		}
		else if (num_dim==4)
		{
		  ncvar->get(buffer,count[0],count[1],count[2],count[3]);
		}
		else
		{
			ncvar->get(buffer,count);
		}
		for(long id=0;id<buffer_size;id++)
		{
			a_buffer[icom+id*num_component]=buffer[id];
		}
		delete buffer;
	}
	return true;
   }


  int * bottom_layer = new int [node_num];
  fill_bottom(bottom_layer);

  long * start_loc = new long [node_num];

  for(long inode=0;inode<node_num;inode++)
  {
	  start_loc[inode]=0;
  }
  
  std::string level_center = (this->m_schismfilePtr)->level_center();
  for(long inode=1;inode<node_num;inode++)
  {
	  int bottom = max(1,bottom_layer[inode-1]);
	  long pre_len = num_component*(num_layer-bottom+1);
	  if(level_center==MeshConstants::HALF_LAYER)
	  {
		  pre_len=num_component*(num_layer-bottom);
	  }
	  start_loc[inode]=start_loc[inode-1]+pre_len;
  }

  int num_bottom_duplicate =0; // for half level data, bottom data are extra data need to get rid of
  num_bottom_duplicate = node_num;

  for(int icom=0;icom<num_component;icom++)
  {
	  float * buffer = new float [node_num*num_layer];
	  NcVar * ncvar = m_ncVar[icom];
	  ncvar->set_cur(current);
	  if (num_dim ==1)
		{
		  ncvar->get(buffer,count[0]);
		}
		else if (num_dim==2)
		{
		  ncvar->get(buffer,count[0],count[1]);
		}
		else if (num_dim==3)
		{
		  ncvar->get(buffer,count[0],count[1],count[2]);
		}
		else if (num_dim==4)
		{
		  ncvar->get(buffer,count[0],count[1],count[2],count[3]);
		}
		else
		{
			ncvar->get(buffer,count);
		}
	  int index =0;
	  int last_layer =num_layer;
	  if(level_center==MeshConstants::HALF_LAYER)
	  {
		  index=num_bottom_duplicate;
		  last_layer--;
	  }
	  for(int ilayer=0;ilayer<last_layer;ilayer++)
	  {
		  for(long inode=0;inode<node_num;inode++)
		  {
			  int bottom = max(1,bottom_layer[inode])-1;
			  if(ilayer>=bottom)
			  {
				  long loc = start_loc[inode];
				  loc = loc + (ilayer-bottom)*num_component+icom;
				  a_buffer[loc] = buffer[index];
			  }
			  index++;
		  }
	  }
	  delete buffer;
  }

  delete bottom_layer;
  delete start_loc;
  return true;
}

void  NetcdfSchismOutputVar::set_cur(const int& a_time_record)
{
	m_current_record = a_time_record;
}




 

NetcdfSchismOutputVar::NetcdfSchismOutputVar(const std::string& a_varName):SCHISMVar(a_varName)
{
	  m_num_component = 0;
}
NetcdfSchismOutputVar::NetcdfSchismOutputVar():SCHISMVar()
{
	  m_num_component=0;
}

int  NetcdfSchismOutputVar::computeDataNumPerTIMEStep() const
{
	int dataSize =1;

  for(int iDim=0;iDim<m_num_dim;iDim++)
  { 
     SCHISMDim * aDim = m_schismfilePtr->get_dim(m_dimensions[iDim]);
  
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





 
   



  
