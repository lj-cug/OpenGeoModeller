#include "SCHISMFile.h"
#include "MeshConstants.h"
#include <algorithm> 
#include <exception>
#include <sstream>


//const int DATA_NUM_BYTE = 4;
//const int MAX_CELL_NODE = 3;


const int         DEFAULT_INT_VALUE         = -9999;
const float       DEFAULT_FLOAT_VALUE       = -9999.0;
const std::string DEFAULT_STR_VALUE         = "error";



SCHISMAtt::SCHISMAtt(const std::string& a_attName):m_name(a_attName),
                                                 m_num_str_value(0),
                                                 m_num_float_value(0),
                                                 m_num_int_value(0),
                                                 m_is_valid(true)
{
}

SCHISMAtt::SCHISMAtt():m_is_valid(false)
{
}

SCHISMAtt::~SCHISMAtt()
{
}

std::string SCHISMAtt::name() const
{
  return m_name;
}

void   SCHISMAtt::add_string_value(const std::string& a_value)
{
  if(m_num_str_value >= MAX_ATT_NUM)
    {
       std::cerr<<"attribute capacity limit is reached, new attribute value is not added";
    }
  else
   {
       m_str_value[m_num_str_value] = a_value;
       m_num_str_value++;
   }
}

void   SCHISMAtt::add_float_value(const float& a_value)
{
  if(m_num_float_value >= MAX_ATT_NUM)
    {
       std::cerr<<"attribute capacity limit is reached, new attribute value is not added";
    }
  else
   {
       m_float_value[m_num_float_value] = a_value;
       m_num_float_value++;
   }
}

void   SCHISMAtt::add_double_value(const double& a_value)
{
  if(m_num_double_value >= MAX_ATT_NUM)
    {
       std::cerr<<"attribute capacity limit is reached, new attribute value is not added";
    }
  else
   {
       m_double_value[m_num_double_value] = a_value;
       m_num_double_value++;
   }
}

void   SCHISMAtt::add_int_value(const int& a_value)
{
  if(m_num_int_value >= MAX_ATT_NUM)
    {
       std::cerr<<"attribute capacity limit is reached, new attribute value is not added";
    }
  else
   {
       m_int_value[m_num_int_value] = a_value;
       m_num_int_value++;
   }
}


std::string SCHISMAtt::string_value(const int& a_index) const
{
  if ((a_index>=  m_num_str_value) || (a_index<0))
  {
    std::cerr<<"attribute index out of range\n";
    return DEFAULT_STR_VALUE;
  }
  else
  {
    return m_str_value[a_index];
  }
}

float SCHISMAtt::float_value(const int& a_index) const
{
  if ((a_index>=  m_num_float_value) || (a_index<0))
  {
    std::cerr<<"attribute index out of range\n";
    return DEFAULT_FLOAT_VALUE;
  }
  else
  {
    return m_float_value[a_index];
  }
}

double SCHISMAtt::double_value(const int& a_index) const
{
  if ((a_index>=  m_num_double_value) || (a_index<0))
  {
    std::cerr<<"attribute index out of range\n";
    return DEFAULT_FLOAT_VALUE;
  }
  else
  {
    return m_double_value[a_index];
  }
}

int SCHISMAtt::int_value(const int& a_index) const
{
  if ((a_index>=  m_num_int_value) || (a_index<0))
  {
    std::cerr<<"attribute index out of range\n";
    return DEFAULT_INT_VALUE;
  }
  else
  {
    return m_int_value[a_index];
  }
}


bool SCHISMAtt::is_valid() const
{
  return m_is_valid;
}

SCHISMDim::SCHISMDim(const std::string& a_name,
                   const int        & a_id,
                   const int        & a_size):m_is_valid(true),
                                              m_id(a_id),
                                              m_name(a_name),
                                              m_size(a_size)
{
  
}

SCHISMDim::SCHISMDim():m_is_valid(false),
	                   m_name("junkdim"),
					   m_size(0)
{
}

SCHISMDim::~SCHISMDim()
{
}

bool SCHISMDim::is_valid() const
{
  return m_is_valid;
}


int SCHISMDim::size() const
{
  return m_size;
}

int SCHISMDim::id() const
{
  return m_id;
}

std::string SCHISMDim::name() const
{
  return m_name;
}

SCHISMVar::SCHISMVar(const std::string& a_varName):m_is_valid(true),
                                                 m_data_cached(false),
                                                 m_name(a_varName),
												 m_horizontal_center(MeshConstants::NODE),
					                             m_vertical_center(MeshConstants::FULL_LAYER),
                                                 m_num_dim(0),
                                                 m_num_attributes(0),
                                                 m_float_cache(NULL),
                                                 m_int_cache(NULL),
												 m_dimensions(new int [MAX_DIM_NUM]),
												 m_attributes(new SCHISMAtt* [MAX_ATT_NUM])
{ 
}


SCHISMVar::SCHISMVar():m_is_valid(false),
	                 m_data_cached(false),
                     m_name("Iamjunk"),
					 m_horizontal_center(""),
					 m_vertical_center(""),
                     m_num_dim(0),
                     m_num_attributes(0),
                     m_float_cache(NULL),
                     m_int_cache(NULL),
					 m_dimensions(NULL),
					 m_attributes(NULL)
{ 
}

int SCHISMVar::num_dims() const
{
  return m_num_dim;
}

SCHISMAtt*    SCHISMVar::get_att(const std::string& a_attName) const
{
  for(int iAtt=0;iAtt<m_num_attributes;iAtt++)
    {
       if(((m_attributes[iAtt])->name()) == a_attName)
         {
            return m_attributes[iAtt];
         }
    }
   return 0;
}

bool        SCHISMVar::add_att(SCHISMAtt*  a_att)
{
   if (m_num_attributes>=MAX_ATT_NUM)
   {
     std::cerr<<"variable "<<m_name<<":attributes capacity is reached, can't add more\n";
     return false;
   }
  else
  {
    m_attributes[m_num_attributes] = a_att;
    m_num_attributes++;
    return true;
  }
  
}

bool  SCHISMVar::add_dimension(const int& a_dimID)
{
   if (m_num_dim>=MAX_DIM_NUM)
   {
     std::cerr<<"variable "<<m_name<<":dimension capacity is reached, can't add more\n";
     return false;
   }
  else
  {
    m_dimensions[m_num_dim] = a_dimID;
    m_num_dim++;
    return true;
  }
}



std::string SCHISMVar::name() const
{
  return m_name;
}


SCHISMDim*   SCHISMVar::get_dim(const int& a_dimNumber) const
{
  
    if (a_dimNumber>=m_num_dim)
    {
      return m_schismfilePtr->get_invalidDim();
    }
    else
    {
      return m_schismfilePtr->get_dim(m_dimensions[a_dimNumber]);
    }

}



bool SCHISMVar::is_valid() const
{
  return m_is_valid;
}

bool  SCHISMVar::put_a_float(const float&    a_value,
                            int  *    a_dim1Loc)
{
	return false;
}

  
bool   SCHISMVar::get(int *       a_buffer) const 
{
	return false;
}

bool   SCHISMVar::get(float *     a_buffer) const 
{
	return false;
}

void    SCHISMVar::set_cur(const int& a_TIMERecord)
{
	return;
}



bool SCHISMVar::hasVerticalLayerDimension() const
{
  bool hasLayer = false;

  for(int iDim=0;iDim<m_num_dim;iDim++)
  { 
     SCHISMDim * aDim = m_schismfilePtr->get_dim(m_dimensions[iDim]);
  
     if((aDim->name()==MeshConstants::DIM_LAYERS))
      {
        return true;
      }
  }
  return hasLayer;

}




int  SCHISMVar::num_component() const
{
   return m_num_component;
}

std::string  SCHISMVar::get_horizontal_center() const
{
   return m_horizontal_center;
}

std::string  SCHISMVar::get_vertical_center() const
{
   return m_vertical_center;
}

void SCHISMVar::set_horizontal_center(std::string a_horizontal_center)
{
	m_horizontal_center= a_horizontal_center;
}

void SCHISMVar::set_vertical_center(std::string a_vertical_center)
{
	m_vertical_center=a_vertical_center;
}


int  SCHISMVar::computeDataNumPerTIMEStepWitHStaggeredLayers() const
{

   // nodeDim is dim 1
   SCHISMDim * nodeDim = m_schismfilePtr->get_dim(m_dimensions[1]);
   
   int totalNodeNum = nodeDim->size();

   SCHISMDim * layerDim = m_schismfilePtr->get_dim(m_dimensions[2]);
   int  numLayer = layerDim->size();
   
   int comSize =1;
   if (m_num_dim>3)
   {
      SCHISMDim * comDim = m_schismfilePtr->get_dim(m_dimensions[3]);
      comSize = comDim->size();
    }
  
  
   int totalDataNum = 0;
   int * kbp00 = new int [totalNodeNum];
   fill_bottom(kbp00);
   for(int iNode=0;iNode<totalNodeNum;iNode++)
   {
       totalDataNum += comSize*(numLayer-max(1,kbp00[iNode])+1);
    }

   delete kbp00;
  return totalDataNum;
}

int  SCHISMVar::computeDataNumPerTIMEStep() const
{
  int dataSize =1;

  for(int iDim=0;iDim<m_num_dim;iDim++)
  { 
     SCHISMDim * aDim = m_schismfilePtr->get_dim(m_dimensions[iDim]);
  
     if(!(aDim->name()==MeshConstants::DIM_TIME))
      {
        dataSize *= aDim->size();
      }
      else if (m_name.compare(MeshConstants::TIME)==0)
     {
        dataSize *= aDim->size();
     }
    else
     {
     }
  }
  return dataSize;

}

bool SCHISMVar::cache_data(int *  a_data)
{
  int dataSize = computeDataNumPerTIMEStep();

  if(m_int_cache)
  {
    delete m_int_cache;
  }

  m_int_cache = new int [dataSize];

  for(int idata=0;idata<dataSize;idata++)
  {
    m_int_cache[idata] = a_data[idata];
  }
   m_data_cached = true;
   return true;
}

bool SCHISMVar::cache_data(float * a_data)
{
  int dataSize = computeDataNumPerTIMEStep();

  if(m_float_cache)
  {
    delete m_float_cache;
  }

  m_float_cache = new float [dataSize];

  for(int idata=0;idata<dataSize;idata++)
  {
    m_float_cache[idata] = a_data[idata];
  }
  m_data_cached = true;
  return true;
}

void  SCHISMVar::fill_bottom(int * a_kbp00) const
{
  std::string dim_name;
 

  for(int idim =0; idim<m_num_dim;idim++)
  {
	  int dim_id = m_dimensions[idim];
	  SCHISMDim * dim = m_schismfilePtr->get_dim(dim_id);
	  if ( (dim->name()==MeshConstants::DIM_MESH_NODES) || 
		   (dim->name()==MeshConstants::DIM_MESH_EDGES) ||
		   (dim->name()==MeshConstants::DIM_MESH_FACES))
	  {
		  dim_name = dim->name();  
	  }
  }

  std::string bottom = MeshConstants::NODE_BOTTOM;
  if (dim_name == MeshConstants::DIM_MESH_EDGES)
  {
	  bottom = MeshConstants::EDGE_BOTTOM;
  }
  else if (dim_name == MeshConstants::DIM_MESH_FACES)
  {
	  bottom = MeshConstants::FACE_BOTTOM;
  }

  SCHISMVar * bottom_ptr = m_schismfilePtr->get_var(bottom);
  bottom_ptr->get(a_kbp00);

}


SCHISMVar::~SCHISMVar() 
{
  
  if (m_int_cache)
   {
     delete m_int_cache;
   }  

   if (m_float_cache)
   {
     delete m_float_cache;
   }  


   if (m_dimensions)
   {
     delete m_dimensions;
   }

  if (m_attributes)
  {
	  for(int iAtt=0;iAtt<m_num_attributes;iAtt++)
	   {
		 delete m_attributes[iAtt];
	   } 

	  delete m_attributes;
  }

}


SCHISMFile::SCHISMFile(const std::string a_SCHISMOutputFile):m_total_num_vars(0),
                                                          m_total_num_dims(0),
                                                          m_SCHISMOutputFile(a_SCHISMOutputFile),
														  m_var_location_att(MeshConstants::CENTER),
														  m_dim_mesh_nodes(MeshConstants::DIM_MESH_NODES),
                                                          m_dim_mesh_faces(MeshConstants::DIM_MESH_FACES),
                                                          m_mesh_face_nodes(MeshConstants::MESH_FACE_NODES),
                                                          m_dim_time(MeshConstants::TIME),
                                                          m_time(MeshConstants::TIME),
                                                          m_node_x(MeshConstants::NODE_X), 
                                                          m_node_y(MeshConstants::NODE_Y), 
                                                          m_node_depth(MeshConstants::NODE_DEPTH),
                                                          m_node_surface(MeshConstants::NODE_SURFACE),
                                                          m_layer_scoord(MeshConstants::LAYER_SCOORD),
                                                          m_dim_layers(MeshConstants::DIM_LAYERS),
                                                          m_hs_att(MeshConstants::HS),
                                                          m_hc_att(MeshConstants::HC),
                                                          m_thetab_att(MeshConstants::THETAB),
                                                          m_thetaf_att(MeshConstants::THETAF),
														  m_data_center(MeshConstants::NODE),
														  m_layer_center(MeshConstants::FULL_LAYER),
														  m_is_valid(true)
												
{
  m_dimensions = new SCHISMDim * [MAX_DIM_NUM];
  m_variables  = new SCHISMVar * [MAX_VAR_NUM];

}

SCHISMFile::~SCHISMFile()
{
}

 std::string      SCHISMFile::level_center() const
{
	return m_layer_center;
}
 
 std::string       SCHISMFile::data_center() const
{
	return m_data_center;
}



bool  SCHISMFile::none_data_var(const std::string a_varName) const
{
	if ((a_varName==m_mesh_face_nodes) || 
	    (a_varName==m_node_x)         ||
	    (a_varName==m_node_y)         ||
	    (a_varName==m_node_depth)     ||
	    (a_varName==m_node_surface)   ||
	    (a_varName==m_layer_scoord))
	{
	   return true;
	}
	else
	{
		return false;
	}

}
std::string  SCHISMFile::file() const
{
	return m_SCHISMOutputFile;
}
bool SCHISMFile::is_valid()const
{
  return m_is_valid;
}

int SCHISMFile::num_vars()const
{
  return m_total_num_vars;
}

int SCHISMFile::num_dims()const
{
  return m_total_num_dims;
}

SCHISMDim*   SCHISMFile::get_dim(const int& a_dimID) const
{
  if(a_dimID<m_total_num_dims)
   {
     return m_dimensions[a_dimID];
   }
  else
  {
    throw SCHISMFileException("input dimension id is larger than total  number of dim\n");
  }
}

SCHISMDim*   SCHISMFile::newSCHISMDim(const std::string& a_name,
                                    const int        & a_id,
                                    const int        & a_size)
{

	return new SCHISMDim(a_name,a_id,a_size);
}

SCHISMDim*   SCHISMFile::get_dim(const std::string& a_dimName) const
{
  for(int dimID = 0; dimID < m_total_num_dims; dimID++)
   {
     SCHISMDim * dim = m_dimensions[dimID];
     if (dim->name() == a_dimName)
      {
        return dim;
      }
   }
  
   ostringstream temp;
   temp<<a_dimName<<"is invalid\n";
   throw SCHISMFileException(temp.str());
  
}

SCHISMDim*   SCHISMFile::get_invalidDim() const
{
   return new SCHISMDim();
}


SCHISMVar*   SCHISMFile::get_var(const int& a_varID) const
{
  if(a_varID<m_total_num_vars)
   {
     return m_variables[a_varID];
   }
  else
  {
     throw SCHISMFileException("input var id is larger than total  number of var\n");
  }
}

SCHISMVar*     SCHISMFile::get_var(const std::string& a_varName) const
{
  for(int iVar=0;iVar<m_total_num_vars;iVar++)
    {
       SCHISMVar * var;
       var = m_variables[iVar];
       if((var->name()) == a_varName)
         {
            return var;
         }
    }
   ostringstream temp;
   temp<<a_varName<<"is invalid\n";
   throw SCHISMFileException(temp.str());

}

//empty api 
bool    SCHISMFile::read(char * a_buffer, const int& a_numByte)
{
	return false;
}


//empty api
void    SCHISMFile::set_cur(const int& a_TIMEStep,
                           const int& a_extraOffset)
{
	return;
}



