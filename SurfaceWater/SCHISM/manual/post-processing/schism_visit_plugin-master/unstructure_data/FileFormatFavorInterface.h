#ifndef _FILEFORMATFAVORINTERFACE_H_
#define _FILEFORMATFAVORINTERFACE_H_  
#include <string>
#include <avtMTSDFileFormat.h>

class avtSCHISMFileFormat;



class FileFormatFavorInterface
{

public:
	virtual void           GetTimes(std::vector<double> & a_times){return;};
  virtual int            GetNTimesteps(const std::string& a_filename){return -1;};
  virtual void           ActivateTimestep(const std::string& a_filename){return;};
  virtual void           FreeUpResources(void){return;}; 
  
  virtual vtkDataSet    *GetMesh(int          a_timeState, 
	                             avtSCHISMFileFormat * a_avtFile,
								 const char * a_meshName){return NULL;};

  virtual vtkDataArray  *GetVar(int           a_timeState,
                                const char *  a_varName){return NULL;};

  virtual vtkDataArray  *GetVectorVar(int          a_timeState, 
                                      const char * a_varName){return NULL;};

  virtual void   PopulateDatabaseMetaData(avtDatabaseMetaData * a_metaData, 
	                                      avtSCHISMFileFormat * a_avtFile,
										  int                   a_timeState){return;};
 
};


#endif