#include "SCHISMFile10.h"
#include "MeshConstants10.h"
#include "netcdfcpp.h"


#ifndef _NETCDFSCHISMOUTPUT10_H_
#define _NETCDFSCHISMOUTPUT10_H_




class NetcdfSchismOutput10: public SCHISMFile10
{
public:

                     NetcdfSchismOutput10(const std::string a_outputFile);
  virtual            ~NetcdfSchismOutput10();
  void               close();
  int               get_dry_wet_val_flag();// 0: filled with last wetting val 1: junk
  // if this file has bottom data, it will return cahced data, if not
  // (like most of scriber format files), it will delegate to mesh file ptr.
  void              get_node_bottom(int* a_node_bottom,const int& a_time);
  void              get_face_bottom(int* a_face_bottom,const int& a_time);
  void              get_edge_bottom(int* a_ele_bottom,const int& a_time);
  bool              update_bottom_index(const int& a_time);
  int               global_att_as_int(const std::string& a_att_name) const;
  std::string       global_att_as_string(const std::string& a_att_name) const;
  bool              inquire_var(const std::string& a_var_name) const;
  void              set_mesh_data_ptr(SCHISMFile10* a_ptr);
  void              set_mesh_bottom(SCHISMFile10* a_ptr, const int& a_time);
protected:
   
private:

  bool              load_dim_var();
  bool              has_var(const std::string& a_var_name) const;
  bool              cache_face_nodes(SCHISMVar10 * mesh_node_var);
  void              fill_node_bottom();
  void              fill_edge_bottom();
  void              fill_ele_bottom();

  // those function will update mesh node/edge/element bottom according
  // to zcor filled value.
  bool              update_node_bottom(const int& a_time);
  void              update_edge_bottom(const int& a_time, int* a_node_bottom);
  void              update_ele_bottom(const int& a_time,int* a_node_bottom);



  NcFile*           m_outputNcFilePtr;
  //only set for latest scriber format file to access bottom index info.

  SCHISMFile10*           m_meshFilePtr; 
  std::map<std::string, std::string>  m_varLongNameMap;
  std::map<std::string, std::string>  m_vector_component_map;

  int*             m_face_bottom;
  int*             m_node_bottom;
  int*             m_edge_bottom;
  int              m_face_bottom_time_id;
  int              m_node_bottom_time_id;
  int              m_edge_bottom_time_id;
};


class  NetcdfSchismOutputVar10: public SCHISMVar10
{
public:
  virtual           ~NetcdfSchismOutputVar10();  
  bool              put_a_float(const float&    a_value,
                                   int  *    a_dim1Loc);
  
  //  get data for current TIME step.if no TIME dimension, get all

  bool              get(int *       a_buffer)  ;
  bool              get(float *     a_buffer)  ;
  bool              get(double *     a_buffer)  ;
  bool              get(long *     a_buffer) ;

  bool              get(int *       a_buffer,int* a_bottom);
  bool              get(float *     a_buffer, int* a_bottom);
  bool              get(double *     a_buffer, int* a_bottom);
  bool              get(long *     a_buffer, int* a_bottom);

  void              set_cur(const int& a_record);
  
  // add reference to thoe ncvar it wrapped
  void              fill_ncVar(NcVar * a_nc_var);
protected:

  int               computeDataNumPerTIMEStep() const;
  void              fill_current_bottom(int * a_kbp00);

  //  get data from cache
  bool              get_int_cache(int *       a_buffer)  ;
  bool              get_long_cache(long *       a_buffer) ;
  bool              get_float_cache(float *     a_buffer) ;

  template<class T>
  bool              load_from_file(T * a_buffer,int* a_bottom);

private:

                   NetcdfSchismOutputVar10(const std::string& a_var_name);
                   NetcdfSchismOutputVar10();


 //bool              hasVerticalLayerDimension() const;
 //int               computeDataNumPerTIMEStepWitHStaggeredLayers() const;
 //void              fill_bottom(int * a_kbp00) const;
 
 

private:

  //NcFile*       m_schismfilePtr;
  NcVar*        m_ncVar[MAX_VAR_COMPONENT10];
  int           m_current_record;
  
  //only file class can create var
  friend class NetcdfSchismOutput10;
  
};


template<class T>
bool  NetcdfSchismOutputVar10::load_from_file(T * a_buffer,int * a_bottom)
{
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
	  if ( (dim_name==MeshConstants10::DIM_MESH_NODES) || 
		   (dim_name==MeshConstants10::DIM_MESH_EDGES) ||
		   (dim_name==MeshConstants10::DIM_MESH_FACES))
	  {
		  node_num = dim->size(); 
	  }
	  if (dim_name==MeshConstants10::DIM_LAYERS)
	  {
		  num_layer = dim->size();
	  }
	  if ((dim_name== MeshConstants10::DIM_TIME) && (m_name != MeshConstants10::TIME))
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

	T * buffer = new T [buffer_size];
	NcVar * ncvar = m_ncVar[0];
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
		a_buffer[id]=buffer[id];
	}
	delete buffer;
	
	return true;
  }

 
  int * bottom_layer = NULL;
  if(a_bottom)
  { 
	  bottom_layer = a_bottom;
  }
  else
  {
	 bottom_layer = new int[node_num];
	 fill_current_bottom(bottom_layer);
  }

  
 
  long * start_loc = new long [node_num];

  for(long inode=0;inode<node_num;inode++)
  {
	  start_loc[inode]=0;
  }
  
  std::string level_center = m_vertical_center;

  int last_node_valid_record_len=0;

   last_node_valid_record_len =(num_layer-max(1,bottom_layer[node_num-1])+1)*num_component;
   if(level_center==MeshConstants10::HALF_LAYER)
   {
		last_node_valid_record_len=last_node_valid_record_len-num_component;
   }
 
  for(long inode=1;inode<node_num;inode++)
  {
	  int bottom = max(1,bottom_layer[inode-1]);
	  int pre_len = (num_layer-bottom+1)*num_component;
	  if(level_center==MeshConstants10::HALF_LAYER)
	  {
		  pre_len=pre_len-num_component;
		  if(pre_len<0) pre_len=0; //dry element 
	  }
	  start_loc[inode]=start_loc[inode-1]+pre_len;

	  
  }

 
	T * buffer = new T [buffer_size];
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
	
	
	int a_node_record_length_in_nc_buffer = buffer_size/node_num;

	for(long inode=0;inode<node_num;inode++)
	{
	  long a_node_valid_data_start= start_loc[inode];
	  long a_node_valid_data_end = 0;
	  if(inode<(node_num-1))
	  {
	      a_node_valid_data_end = start_loc[inode+1]-1;
	  }
	  else
	  {
		  a_node_valid_data_end = a_node_valid_data_start+last_node_valid_record_len-1;
	  }

	  long a_node_valid_record_len= a_node_valid_data_end-a_node_valid_data_start+1;

	  long valid_record_start_in_nc_buffer = (inode+1)* a_node_record_length_in_nc_buffer-a_node_valid_record_len;
	  long nc_buffer_id = valid_record_start_in_nc_buffer;
	  for(long valid_data_index=a_node_valid_data_start;valid_data_index<(a_node_valid_data_end+1);valid_data_index++)
	  {
		  a_buffer[valid_data_index] = buffer[nc_buffer_id];
		  nc_buffer_id++;
	  }
	}
	
	delete buffer;
  
	if (!a_bottom)
	{
		delete bottom_layer;
	}
  delete start_loc;
  return true;
}


#endif
