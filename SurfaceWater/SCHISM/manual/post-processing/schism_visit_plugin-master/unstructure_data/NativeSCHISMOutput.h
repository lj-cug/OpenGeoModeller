#include "SCHISMFile.h"

#ifndef _NATIVESCHISMOUTPUT_H_
#define _NATIVESCHISMOUTPUT_H_

class NativeSchismOutput: public SCHISMFile
{
public:

                     NativeSchismOutput(const std::string a_outputFile);
                     ~NativeSchismOutput();
  void               close();
  bool               read(char * a_buffer, const int& a_numByte);

  // move stream reader to a specified record
  void               set_cur(const int& a_record,
                             const int& a_extraOffset);


  
protected:



private:
  bool               load_dims_Vars();
  ifstream*          m_schismOutputFileStream;
  long long          m_fileLength;
  int                m_numByte;     
  
   

  //some parameter in self output file header
  char               m_data_format[48];
  char               m_data_description[48];
  char               m_start_time[48];
  // those two variable is not filled in self header
  char               m_var_nm[48];
  char               m_var_dim[48];
 
  //data record begin offset and bytelength
  long long          m_dataBlockBeginOffset;
  long long          m_dataBlockLength;

  std::map<std::string, std::string>  m_varLongNameMap;
};


class  NativeSchismOutputVar: public SCHISMVar
{
public:
                    ~NativeSchismOutputVar();
  bool              put_a_float(const float&    a_value,
                                int  *    a_dim1Loc);
  
  //  get data for current TIME step.if no TIME dimension, get all
  bool              get(int *       a_buffer) const ;
  bool              get(float *     a_buffer) const ;
  void              set_cur(const int& a_TIMERecord);

  //int               buffer_size_per_step() const;
  //void              setStaggeredLayerSize(int * a_kbp00);

protected:

 // int               computeDataNumPerTIMEStep() const;
private:

                   NativeSchismOutputVar(const std::string& a_varName);
                   NativeSchismOutputVar();

 
 //bool              hasVerticalLayerDimension() const;
 //int               computeDataNumPerTIMEStepWitHStaggeredLayers() const;
 
 void              setOffsetInDataBlock(const int& a_offset);

private:

  long               m_offsetInDataBlock;
  
  //only file class can create var
  friend class NativeSchismOutput;
  
};







#endif