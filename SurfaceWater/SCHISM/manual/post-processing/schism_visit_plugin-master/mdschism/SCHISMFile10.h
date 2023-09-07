#ifndef _SCHISMFILE10_H_
#define _SCHISMFILE10_H_

#include <iostream>
#include <fstream>    
#include <map>     
#include <string>   
#include <exception>
using std::ios;
using std::ifstream;
using namespace std;


const int         MAX_ATT_VALUES_10            = 5;
const int MAX_DIM_NUM_10   = 20;
const int MAX_VAR_NUM_10   = 100;
const int MAX_ATT_NUM_10   = 10;
const int MAX_VAR_COMPONENT10 = 3;

class SCHISMAtt10;
class SCHISMDim10;
class SCHISMVar10;
class SCHISMFile10;
class SCHISMFileException10;


class SCHISMFileException10: public exception
{
public:

	SCHISMFileException10(std::string a_message):m_errorMessage(a_message){};

	virtual const char* what() const throw()
    {
      return m_errorMessage.c_str();
    }
	virtual  ~ SCHISMFileException10() throw(){};
private: 

	std::string m_errorMessage;

};

class SCHISMVarException10: public exception
{
public:

	SCHISMVarException10(std::string a_message):m_errorMessage(a_message){};

	virtual const char* what() const throw()
    {
      return m_errorMessage.c_str();
    }
	virtual  ~ SCHISMVarException10() throw(){};
private: 

	std::string m_errorMessage;

};

class SCHISMAtt10
{
public:
                   ~SCHISMAtt10();
                   SCHISMAtt10(const std::string& a_attName);
                   SCHISMAtt10();
     bool          is_valid() const;

     std::string   string_value(const int& a_attIndex) const;      
     float         float_value (const int& a_attIndex) const;
     int           int_value   (const int& a_attIndex) const;
	 double        double_value(const int& a_attIndex) const;

     void          add_string_value(const std::string& a_value);      
     void          add_float_value (const  float     & a_value);
	 void          add_double_value (const double     & a_value);
     void          add_int_value   (const  int       & a_value);

     std::string   name() const;
private:
     std::string   m_name;
     std::string   m_str_value[MAX_ATT_VALUES_10];
     float         m_float_value[MAX_ATT_VALUES_10];
     int           m_int_value[MAX_ATT_VALUES_10];
	 double        m_double_value[MAX_ATT_VALUES_10];

     int           m_num_str_value;
     int           m_num_float_value;
     int           m_num_int_value;
	 int           m_num_double_value;
     bool          m_is_valid;
};

class SCHISMDim10
{
public:
                   ~SCHISMDim10();
  bool             is_valid() const;
  int              size() const;
  int              id() const;
  std::string      name() const;

private:
  bool             m_is_valid;
  int              m_id;
  std::string      m_name;
  int              m_size;

private:
                   SCHISMDim10(const std::string& a_name,
                            const int        & a_id,
                            const int        & a_size);

                   SCHISMDim10();

//only file class can create dim
friend class SCHISMFile10;

};




class SCHISMVar10
{
public:
          virtual           ~SCHISMVar10();
          int               num_dims() const;
          std::string       name() const;
		  std::string       get_horizontal_center() const;
		  std::string       get_vertical_center() const;
		  void              set_horizontal_center(std::string a_horizontal_center);
		  void              set_vertical_center(std::string a_vertical_center);
          SCHISMDim10*        get_dim(const int& a_dimNumber) const;
          bool              is_valid()const;
          bool              add_dimension(const int&   a_dimID);
          int               num_component() const;
          SCHISMAtt10*         get_att(const std::string& a_attName) const;
  
          bool              add_att(SCHISMAtt10*          a_att);

          bool              is_defined_over_grid();
		  bool              is_SCHISM_mesh_parameter();

  virtual bool              put_a_float(const float&    a_value,
                                        int  *    a_dim1Loc);
  
  //  get data for current TIME step.if no TIME dimension, get all
  virtual bool              get(int *       a_buffer)  ;
  virtual bool              get(long *       a_buffer) ;
  virtual bool              get(float *     a_buffer) ;
  virtual bool              get(double *     a_buffer) ;
  virtual void              set_cur(const int& a_record);
  //virtual void              set_current(const int& a_record);

  // cache data only works for data not changing with TIME
   bool              cache_data(int  * a_data);
   bool              cache_data(float* a_data);
   bool              cache_data(long* a_data);
   //void              set_kbp(int * a_kbp00);

protected:
                   SCHISMVar10(const std::string& a_varName);
                   SCHISMVar10();
				   SCHISMVar10(const SCHISMVar10& a_other_var);
				   SCHISMVar10& operator=(const SCHISMVar10& a_other_var);

    //  get data from cache
  virtual bool              get_int_cache(int *       a_buffer)  ;
  virtual bool              get_long_cache(long *       a_buffer) ;
  virtual bool              get_float_cache(float *     a_buffer) ;

  //void              fill_bottom(int * a_kbp00) const;

  bool              m_data_cached;
  std::string       m_name;
  std::string       m_horizontal_center;
  std::string       m_vertical_center;
  int               m_num_dim;
  int               m_num_component; // number of data component (scalar:1, vector:>1) 
  
  float*            m_float_cache;        //cache float 
  int *             m_int_cache;          //cache int
  long*             m_long_cache;
  SCHISMFile10 *      m_schismfilePtr;
  
  //int*              m_kbp00;   // staggered vertical layer size
  int*              m_dimensions;

  // compute norminal number of data with uniform layer size
  int               computeDataNumPerTIMEStep() const;
  bool              hasVerticalLayerDimension() const;
  //int               computeDataNumPerTIMEStepWitHStaggeredLayers() const;
  bool              m_is_valid;
  int               m_num_attributes;
  SCHISMAtt10**       m_attributes;

  //only file class can create var
  friend class SCHISMFile10;
};

class SCHISMFile10
{
public:

                     SCHISMFile10(const std::string a_SCHISMOutputFile);
  virtual            ~SCHISMFile10();
  int                num_vars() const;
  int                num_dims() const;
  bool               is_valid() const;
  bool               none_data_var(const std::string a_varName) const;
 
  SCHISMVar10*          get_var(const int& a_varID) const;
  SCHISMVar10*          get_var(const std::string& a_varName) const;
  SCHISMDim10*          get_dim(const int& a_dimID) const;
  SCHISMDim10*          get_dim(const std::string& a_dimName) const;
  SCHISMDim10*          get_invalidDim() const;
  
  virtual bool          read(char * a_buffer, const int& a_numByte);

  // move stream reader to a specified TIME step
  virtual void               set_cur(const int& a_TIMEStep,
                             const int& a_extraOffset);
  std::string        file() const;
  SCHISMDim10*          newSCHISMDim(const std::string& a_name,
                                    const int        & a_id,
                                    const int        & a_size);
  std::string       level_center() const;
  std::string       data_center() const;


  virtual bool              update_bottom_index(const int& a_time); // this function will update mesh node/edge/element bottom according
                                                                    // to zcor filled value, it should be called when mesh bottom changes.
                                                                    // if bottom changes it will return true, or false.
                                                                    // for mesh with fixed bottom it is not used.
 
  
  virtual void              get_node_bottom(int* a_node_bottom,const int& a_time);
  virtual void              get_face_bottom(int* a_face_bottom,const int& a_time);
  virtual void              get_edge_bottom(int* a_ele_bottom,const int& a_time);
  virtual void              get_prism_bottom(int* a_ele_bottom, const int& a_time);
  virtual int       global_att_as_int(const std::string& a_att_name) const;
  virtual std::string global_att_as_string(const std::string& a_att_name) const;
private:

	 SCHISMFile10(const SCHISMFile10& a_other_file);
	 SCHISMFile10& operator=(const SCHISMFile10& a_other_file);

protected:

  SCHISMDim10**                m_dimensions;
  SCHISMVar10**                m_variables;
  int                        m_total_num_vars;
  int                        m_total_num_dims;
  bool                       m_is_valid;
  std::string                m_SCHISMOutputFile;
  

  //some parameter in self output file header
  char                       m_data_format[48];
  char                       m_data_description[48];
  char                       m_start_time[48];
  // those two variable is not filled in self header
  char                       m_var_nm[48];
  char                       m_var_dim[48];

  //data horizontal and vertical center
  std::string                m_layer_center;
  std::string                m_data_center;
 

  // some mesh  related variable and dimension names
  std::string                m_dim_mesh_nodes;
  std::string                m_dim_mesh_faces;
  std::string                m_mesh_face_nodes;

  //data var should have this att
  std::string                m_var_location_att;
 
  std::string                m_dim_time;
  std::string                m_time;

  int                        m_numLayers;
  //coordinates and depth
  std::string                m_node_x;
  std::string                m_node_y;
  std::string                m_node_depth;
  std::string                m_node_surface;
  std::string                m_layer_scoord;
  std::string                m_dim_layers;
 
  // those att should belong to sigma var 
  std::string                m_hs_att;
  std::string                m_hc_att;
  std::string                m_thetab_att;
  std::string                m_thetaf_att;

  //std::map<std::string, std::string>  m_varLongNameMap;
};

#endif