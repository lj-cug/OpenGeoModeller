#include "SCHISMFileUtil.h"
#include <string.h>
#include <stdio.h>

void  retrieve1DVar(float            * a_valBuff,
	                SCHISMFile        * a_SCHISMFilePtr,
                    const std::string & a_varName,
                    const int         & a_varNum) 
{
  SCHISMVar * SCHISMPtr = a_SCHISMFilePtr->get_var(a_varName);
  
  if (!(SCHISMPtr->is_valid()))
    {
      
      throw SCHISMFileException("invlaid var "+a_varName+" for data file "+a_SCHISMFilePtr->file());
    }
  else
    {
      
      if (!SCHISMPtr->get(a_valBuff))
        {
          throw SCHISMFileException("fail to retrieve var "+a_varName+" from data file "+a_SCHISMFilePtr->file());
        }
      
    }
}

void  retrieve1DVar(int           * a_valBuff,
	                SCHISMFile        * a_SCHISMFilePtr,
                    const std::string & a_varName,
                    const int         & a_varNum) 
{
  SCHISMVar * SCHISMPtr = a_SCHISMFilePtr->get_var(a_varName);
  
  if (!(SCHISMPtr->is_valid()))
    {
      
      throw SCHISMFileException("invlaid var "+a_varName+" for data file "+a_SCHISMFilePtr->file());
    }
  else
    {
      
      if (!SCHISMPtr->get(a_valBuff))
        {
          throw SCHISMFileException("fail to retrieve var "+a_varName+" from data file "+a_SCHISMFilePtr->file());
        }
      
    }
}


void retrieve2DVar(float    *         a_valBuff,
                    SCHISMFile*         a_SCHISMOutPtr,
                    const int &        a_TIMEState, 
                    const std::string& a_varName) 
{
  
  SCHISMVar * SCHISMVarPtr = a_SCHISMOutPtr->get_var(a_varName);
  if (!(SCHISMVarPtr->is_valid()))
    {
      
       throw SCHISMFileException("invlaid var "+a_varName+" for data file "+a_SCHISMOutPtr->file());
    }
  

  int nodeIndex0  = 0;
  int TIMERecord  = a_TIMEState;
   
  SCHISMVarPtr->set_cur(TIMERecord);
    
  
  int numOfRecord  = 1;
   
  if (!SCHISMVarPtr->get(a_valBuff))
  {
       throw SCHISMFileException("fail to retrieve var "+a_varName+" from data file "+ a_SCHISMOutPtr->file()); 
   }
   
  
}

// function extract file name,dir, and ext from a path,source:internet
void decomposePath(const char *filePath, char *fileDir, char *fileName, char *fileExt)
{
    #if defined _WIN32
        const char *lastSeparator = strrchr(filePath, '\\');
    #else
        const char *lastSeparator = strrchr(filePath, '/');
    #endif

    const char *lastDot = strrchr(filePath, '.');
    const char *endOfPath = filePath + strlen(filePath);
    const char *startOfName = lastSeparator ? lastSeparator + 1 : filePath;
    const char *startOfExt = lastDot > startOfName ? lastDot : endOfPath;

	#if defined _WIN32
	
	  if(fileDir)
          _snprintf(fileDir, MAX_PATH, "%.*s", startOfName - filePath, filePath);

      if(fileName)
          _snprintf(fileName, MAX_PATH, "%.*s", startOfExt - startOfName, startOfName);

      if(fileExt)
          _snprintf(fileExt, MAX_PATH, "%s", startOfExt);
    #else
	   if(fileDir)
          snprintf(fileDir, MAX_PATH, "%.*s", startOfName - filePath, filePath);

       if(fileName)
          snprintf(fileName, MAX_PATH, "%.*s", startOfExt - startOfName, startOfName);

       if(fileExt)
          snprintf(fileExt, MAX_PATH, "%s", startOfExt);
    #endif
}