#include "SCHISMFile.h"
#include "netcdfcpp.h"

#ifndef _NETCDFSCHISMOUTPUT_H_
#define _NETCDFSCHISMOUTPUT_H_



class NetcdfSchismOutput: public SCHISMFile
{
public:

                     NetcdfSchismOutput(const std::string a_outputFile);
                     ~NetcdfSchismOutput();
  void               close();
 
protected:


private:

  bool              load_dim_var();
  bool              cache_face_nodes(SCHISMVar * mesh_node_var);
  NcFile*           m_outputNcFilePtr;

  std::map<std::string, std::string>  m_varLongNameMap;
  std::map<std::string, std::string>  m_vector_component_map;
};


class  NetcdfSchismOutputVar: public SCHISMVar
{
public:
                    ~NetcdfSchismOutputVar();  
  bool              put_a_float(const float&    a_value,
                                int  *    a_dim1Loc);
  
  //  get data for current TIME step.if no TIME dimension, get all
  bool              get(int *       a_buffer) const ;
  bool              get(float *     a_buffer) const ;
  void              set_cur(const int& a_record);
  
  // add reference to thoe ncvar it wrapped
  void              fill_ncVar(NcVar * a_nc_var);
protected:

  int               computeDataNumPerTIMEStep() const;

private:

                   NetcdfSchismOutputVar(const std::string& a_var_name);
                   NetcdfSchismOutputVar();


 //bool              hasVerticalLayerDimension() const;
 //int               computeDataNumPerTIMEStepWitHStaggeredLayers() const;
 //void              fill_bottom(int * a_kbp00) const;
 
 

private:

  //NcFile*       m_schismfilePtr;
  NcVar*        m_ncVar[MAX_VAR_COMPONENT];
  int           m_current_record;
  
  //only file class can create var
  friend class NetcdfSchismOutput;
  
};



#endif
