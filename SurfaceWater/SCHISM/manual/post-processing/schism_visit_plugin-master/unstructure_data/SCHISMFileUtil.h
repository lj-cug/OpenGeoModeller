
#ifndef _SCHISMFILEUTIL_H_
#define _SCHISMFILEUTIL_H_
#include "SCHISMFile.h"


void  retrieve1DVar(float            * a_valBuff,
	                SCHISMFile        * a_SCHISMFilePtr,
                    const std::string & a_varName,
                    const int         & a_varNum);


void  retrieve1DVar(int            * a_valBuff,
	                  SCHISMFile        * a_SCHISMFilePtr,
                      const std::string & a_varName,
                      const int         & a_varNum);



void retrieve2DVar(float    *         a_valBuff,
                    SCHISMFile*         a_SCHISMOutPtr,
                    const int &        a_TIMEState, 
                    const std::string& a_varName) ;


void decomposePath(const char *filePath, char *fileDir, char *fileName, char *fileExt);

const size_t MAX_PATH = 300;
const int MAX_FILE_NAME_LEN =100;
const int MAX_PATH_LEN =300;
#endif