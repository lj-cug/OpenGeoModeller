#include "SCHISMFile10.h"
#include "MeshConstants10.h"
#include <algorithm> 
#include <exception>
#include <sstream>



//const int DATA_NUM_BYTE = 4;
//const int MAX_CELL_NODE = 3;


const int         DEFAULT_INT_VALUE         = -9999;
const float       DEFAULT_FLOAT_VALUE       = -9999.0;
const std::string DEFAULT_STR_VALUE         = "error";



SCHISMAtt10::SCHISMAtt10(const std::string& a_attName):m_name(a_attName),
                                                 m_num_str_value(0),
                                                 m_num_float_value(0),
                                                 m_num_int_value(0),
												 m_num_double_value(0),
                                                 m_is_valid(true)
{
}

SCHISMAtt10::SCHISMAtt10():m_is_valid(false)
{
}

SCHISMAtt10::~SCHISMAtt10()
{
}

std::string SCHISMAtt10::name() const
{
  return m_name;
}

void   SCHISMAtt10::add_string_value(const std::string& a_value)
{
  if(m_num_str_value >= MAX_ATT_NUM_10)
    {
       std::cerr<<"attribute capacity limit is reached, new attribute value is not added";
    }
  else
   {
       m_str_value[m_num_str_value] = a_value;
       m_num_str_value++;
   }
}

void   SCHISMAtt10::add_float_value(const float& a_value)
{
  if(m_num_float_value >= MAX_ATT_NUM_10)
    {
       std::cerr<<"attribute capacity limit is reached, new attribute value is not added";
    }
  else
   {
       m_float_value[m_num_float_value] = a_value;
       m_num_float_value++;
   }
}

void   SCHISMAtt10::add_double_value(const double& a_value)
{
  if(m_num_double_value >= MAX_ATT_NUM_10)
    {
       std::cerr<<"attribute capacity limit is reached, new attribute value is not added";
    }
  else
   {
       m_double_value[m_num_double_value] = a_value;
       m_num_double_value++;
   }
}

void   SCHISMAtt10::add_int_value(const int& a_value)
{
  if(m_num_int_value >= MAX_ATT_NUM_10)
    {
       std::cerr<<"attribute capacity limit is reached, new attribute value is not added";
    }
  else
   {
       m_int_value[m_num_int_value] = a_value;
       m_num_int_value++;
   }
}


std::string SCHISMAtt10::string_value(const int& a_index) const
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

float SCHISMAtt10::float_value(const int& a_index) const
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

double SCHISMAtt10::double_value(const int& a_index) const
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

int SCHISMAtt10::int_value(const int& a_index) const
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


bool SCHISMAtt10::is_valid() const
{
  return m_is_valid;
}

SCHISMDim10::SCHISMDim10(const std::string& a_name,
                   const int        & a_id,
                   const int        & a_size):m_is_valid(true),
                                              m_id(a_id),
                                              m_name(a_name),
                                              m_size(a_size)
{
  
}

SCHISMDim10::SCHISMDim10():m_is_valid(false),
	                   m_name("junkdim"),
					   m_size(0)
{
}

SCHISMDim10::~SCHISMDim10()
{
}

bool SCHISMDim10::is_valid() const
{
  return m_is_valid;
}


int SCHISMDim10::size() const
{
  return m_size;
}

int SCHISMDim10::id() const
{
  return m_id;
}

std::string SCHISMDim10::name() const
{
  return m_name;
}

SCHISMVar10::SCHISMVar10(const std::string& a_varName):m_is_valid(true),
                                                 m_data_cached(false),
                                                 m_name(a_varName),
												 m_horizontal_center(MeshConstants10::NODE),
					                             m_vertical_center(MeshConstants10::FULL_LAYER),
                                                 m_num_dim(0),
                                                 m_num_attributes(0),
                                                 m_float_cache(NULL),
                                                 m_int_cache(NULL),
												 m_long_cache(NULL),
												 m_dimensions(new int [MAX_DIM_NUM_10]),
												 m_attributes(new SCHISMAtt10* [MAX_ATT_NUM_10])
{ 
}


SCHISMVar10::SCHISMVar10():m_is_valid(false),
	                 m_data_cached(false),
                     m_name("Iamjunk"),
					 m_horizontal_center(""),
					 m_vertical_center(""),
                     m_num_dim(0),
                     m_num_attributes(0),
                     m_float_cache(NULL),
                     m_int_cache(NULL),
					 m_long_cache(NULL),
					 m_dimensions(NULL),
					 m_attributes(NULL)
{ 
}

int SCHISMVar10::num_dims() const
{
  return m_num_dim;
}

SCHISMAtt10*    SCHISMVar10::get_att(const std::string& a_attName) const
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

bool        SCHISMVar10::add_att(SCHISMAtt10*  a_att)
{
   if (m_num_attributes>=MAX_ATT_NUM_10)
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

bool  SCHISMVar10::add_dimension(const int& a_dimID)
{
   if (m_num_dim>=MAX_DIM_NUM_10)
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



std::string SCHISMVar10::name() const
{
  return m_name;
}


SCHISMDim10*   SCHISMVar10::get_dim(const int& a_dimNumber) const
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



bool SCHISMVar10::is_valid() const
{
  return m_is_valid;
}

bool  SCHISMVar10::put_a_float(const float&    a_value,
                            int  *    a_dim1Loc)
{
	return false;
}

  
bool   SCHISMVar10::get(int *       a_buffer) 
{
	return false;
}



bool   SCHISMVar10::get(float *     a_buffer) 
{
	return false;
}

bool   SCHISMVar10::get(double *     a_buffer) 
{
	return false;
}

bool   SCHISMVar10::get(int *       a_buffer,int* a_bottom)
{
	return false;
}



bool   SCHISMVar10::get(float *     a_buffer,int* a_bottom)
{
	return false;
}

bool   SCHISMVar10::get(double *     a_buffer,int* a_bottom)
{
	return false;
}

void    SCHISMVar10::set_cur(const int& a_timeRecord)
{
	return;
}

bool   SCHISMVar10::get(long *       a_buffer,int* a_bottom)
{
	return false;
}

bool   SCHISMVar10::get(long *       a_buffer) 
{
	return false;
}

 
bool    SCHISMVar10::get_int_cache(int *       a_buffer)  
{
	return false;
}

bool    SCHISMVar10::get_long_cache(long *       a_buffer) 
{
	return false;
}

bool     SCHISMVar10::get_float_cache(float *     a_buffer) 
{
	return false;
}



bool SCHISMVar10::hasVerticalLayerDimension() const
{
  bool hasLayer = false;

  for(int iDim=0;iDim<m_num_dim;iDim++)
  { 
     SCHISMDim10 * aDim = m_schismfilePtr->get_dim(m_dimensions[iDim]);
  
     if((aDim->name()==MeshConstants10::DIM_LAYERS))
      {
        return true;
      }
  }
  return hasLayer;

}




int  SCHISMVar10::num_component() const
{
   return m_num_component;
}

std::string  SCHISMVar10::get_horizontal_center() const
{
   return m_horizontal_center;
}

std::string  SCHISMVar10::get_vertical_center() const
{
   return m_vertical_center;
}

void SCHISMVar10::set_horizontal_center(std::string a_horizontal_center)
{
	m_horizontal_center= a_horizontal_center;
}

void SCHISMVar10::set_vertical_center(std::string a_vertical_center)
{
	m_vertical_center=a_vertical_center;
}


int  SCHISMVar10::computeDataNumPerTIMEStepWitHStaggeredLayers() const
{

   // nodeDim is dim 1
   SCHISMDim10 * nodeDim = m_schismfilePtr->get_dim(m_dimensions[1]);
   
   int totalNodeNum = nodeDim->size();

   SCHISMDim10 * layerDim = m_schismfilePtr->get_dim(m_dimensions[2]);
   int  numLayer = layerDim->size();
   
   int comSize =1;
   if (m_num_dim>3)
   {
      SCHISMDim10 * comDim = m_schismfilePtr->get_dim(m_dimensions[3]);
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

int  SCHISMVar10::computeDataNumPerTIMEStep() const
{
  int dataSize =1;

  for(int iDim=0;iDim<m_num_dim;iDim++)
  { 
     SCHISMDim10 * aDim = m_schismfilePtr->get_dim(m_dimensions[iDim]);
  
     if(!(aDim->name()==MeshConstants10::DIM_TIME))
      {
        dataSize *= aDim->size();
      }
      else if (m_name.compare(MeshConstants10::TIME)==0)
     {
        dataSize *= aDim->size();
     }
    else
     {
     }
  }
  return dataSize;
}

bool SCHISMVar10::cache_data(int *  a_data)
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

bool SCHISMVar10::cache_data(float * a_data)
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

bool SCHISMVar10::cache_data(long * a_data)
{
  int dataSize = computeDataNumPerTIMEStep();

  if(m_long_cache)
  {
    delete m_long_cache;
  }

  m_long_cache = new long [dataSize];

  for(long idata=0;idata<dataSize;idata++)
  {
    m_long_cache[idata] = a_data[idata];
  }
  m_data_cached = true;
  return true;
}

void  SCHISMVar10::fill_bottom(int * a_kbp00) const
{
  std::string dim_name;
 

  for(int idim =0; idim<m_num_dim;idim++)
  {
	  int dim_id = m_dimensions[idim];
	  SCHISMDim10 * dim = m_schismfilePtr->get_dim(dim_id);
	  if ( (dim->name()==MeshConstants10::DIM_MESH_NODES) || 
		   (dim->name()==MeshConstants10::DIM_MESH_EDGES) ||
		   (dim->name()==MeshConstants10::DIM_MESH_FACES))
	  {
		  dim_name = dim->name();  
	  }
  }

  std::string bottom = MeshConstants10::NODE_BOTTOM;
  if (dim_name == MeshConstants10::DIM_MESH_EDGES)
  {
	  bottom = MeshConstants10::EDGE_BOTTOM;
  }
  else if (dim_name == MeshConstants10::DIM_MESH_FACES)
  {
	  bottom = MeshConstants10::FACE_BOTTOM;
  }

  SCHISMVar10 * bottom_ptr = m_schismfilePtr->get_var(bottom);
  bottom_ptr->get(a_kbp00);

}


SCHISMVar10::~SCHISMVar10() 
{
  

  if (m_int_cache)
   {
     delete m_int_cache;
   }  

   if (m_float_cache)
   {
     delete m_float_cache;
   }  

   if (m_long_cache)
   {
     delete m_long_cache;
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

bool   SCHISMVar10::is_defined_over_grid()
{
  std::string dim_name;
 

  for(int idim =0; idim<m_num_dim;idim++)
  {
	  int dim_id = m_dimensions[idim];
	  SCHISMDim10 * dim = m_schismfilePtr->get_dim(dim_id);
	  if ( (dim->name()==MeshConstants10::DIM_MESH_NODES) || 
		   (dim->name()==MeshConstants10::DIM_MESH_EDGES) ||
		   (dim->name()==MeshConstants10::DIM_MESH_FACES))
	  {
		  return true;  
	  }
  }
	return false;
}

bool  SCHISMVar10::is_SCHISM_mesh_parameter()
{
	if (m_name.find(MeshConstants10::MESH_PAR_HEAD)!=string::npos)
	{
		return true;
	}
	return false;
}

SCHISMVar10::SCHISMVar10(const SCHISMVar10& a_other_var)
 {
	 throw SCHISMVarException10("SCHISMVar copy is not allowed ");
 }

SCHISMVar10& SCHISMVar10::operator=(const SCHISMVar10& a_other_var)
{
	throw SCHISMVarException10("SCHISMVar assignment is not allowed ");
}

SCHISMFile10::SCHISMFile10(const std::string a_SCHISMOutputFile):m_total_num_vars(0),
                                                          m_total_num_dims(0),
                                                          m_SCHISMOutputFile(a_SCHISMOutputFile),
														  m_var_location_att(MeshConstants10::CENTER),
														  m_dim_mesh_nodes(MeshConstants10::DIM_MESH_NODES),
                                                          m_dim_mesh_faces(MeshConstants10::DIM_MESH_FACES),
                                                          m_mesh_face_nodes(MeshConstants10::MESH_FACE_NODES),
                                                          m_dim_time(MeshConstants10::TIME),
                                                          m_time(MeshConstants10::TIME),
                                                          m_node_x(MeshConstants10::NODE_X), 
                                                          m_node_y(MeshConstants10::NODE_Y), 
                                                          m_node_depth(MeshConstants10::NODE_DEPTH),
                                                          m_node_surface(MeshConstants10::NODE_SURFACE),
                                                          m_layer_scoord(MeshConstants10::LAYER_SCOORD),
                                                          m_dim_layers(MeshConstants10::DIM_LAYERS),
                                                          m_hs_att(MeshConstants10::HS),
                                                          m_hc_att(MeshConstants10::HC),
                                                          m_thetab_att(MeshConstants10::THETAB),
                                                          m_thetaf_att(MeshConstants10::THETAF),
														  m_data_center(MeshConstants10::NODE),
														  m_layer_center(MeshConstants10::FULL_LAYER),
														  m_is_valid(true)
												
{
  //m_dimensions = new SCHISMDim10 * [MAX_DIM_NUM_10];
  //m_variables  = new SCHISMVar10 * [MAX_VAR_NUM_10];

}

SCHISMFile10::~SCHISMFile10()
{
}
int  SCHISMFile10::get_dry_wet_val_flag()
{
	return 1;
}
SCHISMFile10::SCHISMFile10(const SCHISMFile10& a_other_file)
{
	throw SCHISMFileException10("SCHISMFile copy is not allowed ");
}
SCHISMFile10& SCHISMFile10::operator=(const SCHISMFile10& a_other_file)
{
	throw SCHISMFileException10("SCHISMFile assignment is not allowed ");
}


 std::string      SCHISMFile10::level_center() const
{
	return m_layer_center;
}
 
 std::string       SCHISMFile10::data_center() const
{
	return m_data_center;
}

 void   SCHISMFile10::set_mesh_data_ptr(SCHISMFile10* a_ptr)
 {
	 return;
 }

 void   SCHISMFile10::set_mesh_bottom(SCHISMFile10* a_ptr, const int& a_time)
 {
	 return;
 }


bool  SCHISMFile10::none_data_var(const std::string a_varName) const
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
std::string  SCHISMFile10::file() const
{
	return m_SCHISMOutputFile;
}
bool SCHISMFile10::is_valid()const
{
  return m_is_valid;
}

int SCHISMFile10::num_vars()const
{
  return m_total_num_vars;
}

int SCHISMFile10::num_dims()const
{
  return m_total_num_dims;
}

SCHISMDim10*   SCHISMFile10::get_dim(const int& a_dimID) const
{
  if(a_dimID<m_total_num_dims)
   {
     return m_dimensions[a_dimID];
   }
  else
  {
    throw SCHISMFileException10("input dimension id is larger than total  number of dim\n");
  }
}

SCHISMDim10*   SCHISMFile10::newSCHISMDim(const std::string& a_name,
                                    const int        & a_id,
                                    const int        & a_size)
{

	return new SCHISMDim10(a_name,a_id,a_size);
}

SCHISMDim10*   SCHISMFile10::get_dim(const std::string& a_dimName) const
{
  for(int dimID = 0; dimID < m_total_num_dims; dimID++)
   {
     SCHISMDim10 * dim = m_dimensions[dimID];
     if (dim->name() == a_dimName)
      {
        return dim;
      }
   }
  
   ostringstream temp;
   temp<<a_dimName<<"is invalid\n";
   throw SCHISMFileException10(temp.str());
  
}

SCHISMDim10*   SCHISMFile10::get_invalidDim() const
{
   return new SCHISMDim10();
}


SCHISMVar10*   SCHISMFile10::get_var(const int& a_varID) const
{
  if(a_varID<m_total_num_vars)
   {
     return m_variables[a_varID];
   }
  else
  {
     throw SCHISMFileException10("input var id is larger than total  number of var\n");
  }
}

SCHISMVar10*     SCHISMFile10::get_var(const std::string& a_varName) const
{

  for(int iVar=0;iVar<m_total_num_vars;iVar++)
    {
       SCHISMVar10 * var;
       var = m_variables[iVar];

       if((var->name()) == a_varName)
         {
            return var;
         }
    }
   ostringstream temp;
   temp<<a_varName<<" is invalid\n";
   throw SCHISMFileException10(temp.str());

}

//empty api 
bool    SCHISMFile10::read(char * a_buffer, const int& a_numByte)
{
	return false;
}

bool SCHISMFile10::inquire_var(const std::string& a_var_name) const
{
	return false;
}

//empty api
void    SCHISMFile10::set_cur(const int& a_TIMEStep,
                           const int& a_extraOffset)
{
	return;
}


void  SCHISMFile10::get_node_bottom(int* a_node_bottom,const int& a_time)
{
	return;
}
void   SCHISMFile10::get_face_bottom(int* a_face_bottom,const int& a_time)
{
	return;
}

void   SCHISMFile10::get_edge_bottom(int* a_ele_bottom,const int& a_time)
{
	return;
}

bool   SCHISMFile10::update_bottom_index(const int& a_time)
{
	return false;
}


int SCHISMFile10::global_att_as_int(const std::string& a_att_name) const
{
	 throw SCHISMFileException10("global att is not implemented\n");
	 return 0;
}

std::string SCHISMFile10::global_att_as_string(const std::string& a_att_name) const
{
	 throw SCHISMFileException10("global att is not implemented\n");
	 return "";
}
