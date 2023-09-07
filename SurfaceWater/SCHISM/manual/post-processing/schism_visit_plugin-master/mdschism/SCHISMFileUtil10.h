
#ifndef _SCHISMFILEUTIL_H_
#define _SCHISMFILEUTIL_H_
#include "SCHISMFile10.h"


void  retrieve1DVar(float            * a_valBuff,
	                SCHISMFile10        * a_SCHISMFilePtr,
                    const std::string & a_varName,
                    const int         & a_varNum);


void  retrieve1DVar(double            * a_valBuff,
	                SCHISMFile10        * a_SCHISMFilePtr,
                    const std::string & a_varName,
                    const int         & a_varNum);


void  retrieve1DVar(int            * a_valBuff,
	                  SCHISMFile10        * a_SCHISMFilePtr,
                      const std::string & a_varName,
                      const int         & a_varNum);



void retrieve2DVar(float    *         a_valBuff,
                    SCHISMFile10*         a_SCHISMOutPtr,
                    const int &        a_TIMEState, 
                    const std::string& a_varName) ;



#endif