#include "Average.h"

// ****************************************************************************
//  Method:  trapezoidAverage
//
//  Purpose:
//      Return  averaged node value over layers by trapezpoid method
//      
//      
//
//  Arguments:
//      a_state     layered node variable with size of mesh layers
//      a_z         elevation of every layer, size of mesh layers
//      a_layerNum  number  of mesh layers
//
// ****************************************************************************
float trapezoidAverage(float    *  a_state,
                       float    *  a_z,
					   const int &  a_layerNum) 
{
  float  sum = 0.0;
  float  half =0.5;
  for (int iLayer=0;iLayer<a_layerNum-1;iLayer++)
    {
      float state1 = a_state[iLayer];
      float state2 = a_state[iLayer+1];
      float     z1 = a_z[iLayer];
      float     z2 = a_z[iLayer+1];
      sum          += (state1+state2)*(z2-z1)*half;
    }
  
  float average    = sum / (-a_z[0]+a_z[a_layerNum-1]);
  return average;
}


// ****************************************************************************
//  Method: rectAverage
//
//  Purpose:
//      Return  averaged node value over layers by rect method
//      
//      
//
//  Arguments:
//      a_state     prsime center variable with one less than num of mesh layers
//      a_z         elevation of every layer, size of mesh layers
//      a_layerNum  number  of mesh layers
//
//
// ****************************************************************************

float rectAverage(float    *  a_state,
                  float    *  a_z,
                  const int &  a_layerNum)
{
  float  sum = 0.0;
  for (int iLayer=0;iLayer<a_layerNum-1;iLayer++)
    {
      float state  = a_state[iLayer];
      float     z1 = a_z[iLayer];
      float     z2 = a_z[iLayer+1];
      sum          += state*(z2-z1);
    }
  
  float average    = sum / (-a_z[0]+a_z[a_layerNum-1]);
  return average;
}
