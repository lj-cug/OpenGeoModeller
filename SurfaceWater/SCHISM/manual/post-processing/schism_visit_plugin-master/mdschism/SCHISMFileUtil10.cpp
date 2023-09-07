#include "SCHISMFileUtil10.h"
#include <string.h>
#include <stdio.h>

void  retrieve1DVar(float            * a_valBuff,
	                SCHISMFile10        * a_SCHISMFilePtr,
                    const std::string & a_varName,
                    const int         & a_varNum) 
{
  SCHISMVar10 * SCHISMPtr = a_SCHISMFilePtr->get_var(a_varName);
  
  if (!(SCHISMPtr->is_valid()))
    {
      
      throw SCHISMFileException10("invlaid var "+a_varName+" for data file "+a_SCHISMFilePtr->file());
    }
  else
    {
      
      if (!SCHISMPtr->get(a_valBuff))
        {
          throw SCHISMFileException10("fail to retrieve var "+a_varName+" from data file "+a_SCHISMFilePtr->file());
        }
      
    }
}


void  retrieve1DVar(double            * a_valBuff,
	                SCHISMFile10        * a_SCHISMFilePtr,
                    const std::string & a_varName,
                    const int         & a_varNum) 
{
  SCHISMVar10 * SCHISMPtr = a_SCHISMFilePtr->get_var(a_varName);
  
  if (!(SCHISMPtr->is_valid()))
    {
      
      throw SCHISMFileException10("invlaid var "+a_varName+" for data file "+a_SCHISMFilePtr->file());
    }
  else
    {
      
      if (!SCHISMPtr->get(a_valBuff))
        {
          throw SCHISMFileException10("fail to retrieve var "+a_varName+" from data file "+a_SCHISMFilePtr->file());
        }
      
    }
}


void  retrieve1DVar(int           * a_valBuff,
	                SCHISMFile10        * a_SCHISMFilePtr,
                    const std::string & a_varName,
                    const int         & a_varNum) 
{
  SCHISMVar10 * SCHISMPtr = a_SCHISMFilePtr->get_var(a_varName);
  
  if (!(SCHISMPtr->is_valid()))
    {
      
      throw SCHISMFileException10("invlaid var "+a_varName+" for data file "+a_SCHISMFilePtr->file());
    }
  else
    {
      
      if (!SCHISMPtr->get(a_valBuff))
        {
          throw SCHISMFileException10("fail to retrieve var "+a_varName+" from data file "+a_SCHISMFilePtr->file());
        }
      
    }
}


void retrieve2DVar(float    *         a_valBuff,
                    SCHISMFile10*         a_SCHISMOutPtr,
                    const int &        a_TIMEState, 
                    const std::string& a_varName) 
{
  
  SCHISMVar10 * SCHISMVarPtr = a_SCHISMOutPtr->get_var(a_varName);
  if (!(SCHISMVarPtr->is_valid()))
    {
      
       throw SCHISMFileException10("invlaid var "+a_varName+" for data file "+a_SCHISMOutPtr->file());
    }
  

  int nodeIndex0  = 0;
  int TIMERecord  = a_TIMEState;
   
  SCHISMVarPtr->set_cur(TIMERecord);
    
  
  int numOfRecord  = 1;
   
  if (!SCHISMVarPtr->get(a_valBuff))
  {
       throw SCHISMFileException10("fail to retrieve var "+a_varName+" from data file "+ a_SCHISMOutPtr->file()); 
   }
   
  
}

