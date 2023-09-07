#include "SCHISMFile10.h"
#include "MeshConstants10.h"
#include "netcdfcpp.h"
//#include <DebugStream.h>
#ifndef _MDNETCDFSCHISMOUTPUT_H_
#define _MDNETCDFSCHISMOUTPUT_H_




class MDSchismOutput: public SCHISMFile10
{
public:

                     MDSchismOutput(const std::string a_mdoutputFile,const std::string a_local_mesh_file);
  virtual            ~MDSchismOutput();
  void               close();

  void              get_node_bottom(int* a_node_bottom,const int& a_time);
  void              get_face_bottom(int* a_face_bottom,const int& a_time);
  void              get_edge_bottom(int* a_ele_bottom,const int& a_time);
  void              get_prism_bottom(int* a_prism_bottom, const int& a_time);
  void              set_prism_bottom(const int& a_time, int* a_prism_bottom);
  bool              update_bottom_index(const int& a_time);
  int              global_att_as_int(const std::string& a_att_name) const;
  std::string       global_att_as_string(const std::string& a_att_name) const;
 
protected:
   
private:

  bool              load_dim_var();
  //bool              cache_face_nodes(SCHISMVar10 * mesh_node_var);
  void              fill_bottom(); //fill ele node side bottoms
  

  // those function will update mesh node/edge/element bottom according
  // to zcor filled value.
  bool              update_node_bottom(const int& a_time);
  void              update_edge_bottom(const int& a_time, int* a_node_bottom);
  void              update_ele_bottom(const int& a_time,int* a_node_bottom);
  void              update_prism_bottom(const int& a_time, int* a_prism_bottom);


  NcFile*           m_outputNcFilePtr;
  std::string       m_local_mesh_file;
  
  std::map<std::string, std::string>  m_varLongNameMap;
  std::map<std::string, std::string>  m_vector_component_map;

  int*             m_face_bottom;
  int*             m_node_bottom;
  int*             m_edge_bottom;
  int*             m_prism_bottom;
  int              m_face_bottom_time_id;
  int              m_node_bottom_time_id;
  int              m_edge_bottom_time_id;
  int              m_prism_bottom_time_id;
  long *           m_face_nodes;
  long *           m_side_nodes;
};


class  MDSchismOutputVar: public SCHISMVar10
{
public:
  virtual           ~MDSchismOutputVar();  
  bool              put_a_float(const float&    a_value,
                                   int  *    a_dim1Loc);
  
  //  get data for current TIME step.if no TIME dimension, get all

  bool              get(int *       a_buffer)  ;
  bool              get(float *     a_buffer)  ;
  bool              get(double *     a_buffer)  ;
  bool              get(long *     a_buffer) ;
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
  bool              load_from_file(T * a_buffer);

private:

                   MDSchismOutputVar(const std::string& a_var_name);
                   MDSchismOutputVar();


 //bool              hasVerticalLayerDimension() const;
 //int               computeDataNumPerTIMEStepWitHStaggeredLayers() const;
 //void              fill_bottom(int * a_kbp00) const;
 
 

private:

  //MDSchismOutput*       m_schismfilePtr;
  NcVar*        m_ncVar[MAX_VAR_COMPONENT10];
  int           m_current_record;
 
  
  //only file class can create var
  friend class MDSchismOutput;
  
};


template<class T>
bool  MDSchismOutputVar::load_from_file(T * a_buffer)
{
  int num_component = m_num_component;

  long node_num = 0;
  long buffer_size =1;
  int num_layer = 1;
  NcVar * ncvar = m_ncVar[0];
  int num_dim = ncvar->num_dims();
  long * current = new long [num_dim];
  long * count   = new long [num_dim];
  //debug1 << " begin load dims in load from file\n";
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

  //debug1 << " done load dims in load from file "<<buffer_size<<" "<<node_num<<" "<<num_layer<<"\n";
  if (num_layer ==1) // not 3d data
  {

	T * buffer = new T [buffer_size];
	//debug1 << " buffer_size is "<<buffer_size<<"\n";
	NcVar * ncvar = m_ncVar[0];
	//debug1 << " ncvar is  " << ncvar << "\n";
	ncvar->set_cur(current);
	//debug1 << "var current is set "<<current[0]<<" "<<current[1]<<"\n";
	//debug1 << "var count is set " << count[0] << " " << count[1] << "\n";
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
	//debug1 << " done read from file 2d \n";
	return true;
  }


  int * bottom_layer = new int [node_num];

  fill_current_bottom(bottom_layer);
  
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
	
	//debug1 << "done read ncvar data\n";
	int a_node_record_length_in_nc_buffer = buffer_size/node_num;
	long total_valid_data_len = 0;
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
	  //debug1 << "done read valid ncvar data "<<inode<<" " << valid_record_start_in_nc_buffer<<"\n";
	  total_valid_data_len += a_node_valid_record_len;
	}
	//debug1 << "done read valid ncvar data " <<total_valid_data_len<<"\n";
	delete buffer;
  

  delete bottom_layer;
  delete start_loc;
  return true;
}


#endif
