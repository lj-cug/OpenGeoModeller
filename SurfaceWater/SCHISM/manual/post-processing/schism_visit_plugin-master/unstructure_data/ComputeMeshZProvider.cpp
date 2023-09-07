#include <math.h>
#include "ComputeMeshZProvider.h"
#include "MeshConstants.h"
#include "SCHISMFileUtil.h"
#include <cmath>
#include <algorithm>



ComputeMeshZProvider::ComputeMeshZProvider(const std::string& a_selfeOutputFile):SCHISMMeshProvider(a_selfeOutputFile)
{

}

float  ComputeMeshZProvider::convertStoZ(const float    & a_sigma,
                                     const float    & a_surface,
                                     const float    & a_depth,
                                     const float    & a_hs,
                                     const float    & a_hc,
                                     const float    & a_thetab,
                                     const float    & a_thetaf) const
{

  float surface = a_surface;
  if (fabs(a_surface-MeshConstants::DRY_SURFACE)<1.0e-6)
    {
      surface = 0.0;
    }

  float one =1.0;
  float half=0.5;
  float two =2.0;

  float csigma = (one-a_thetab)*sinh(a_thetaf*a_sigma)/sinh(a_thetaf)
  +a_thetab*(tanh(a_thetaf*(a_sigma+half))-tanh(a_thetaf/two))/(two*tanh(a_thetaf/two));
  

  float hat    = a_depth;
  if (a_hs < a_depth)
    {
      hat       = a_hs;
    }
   float z        = MeshConstants::DRY_SURFACE;
   if (hat>a_hc)
    {
            z      = surface*(one+a_sigma)+a_hc*a_sigma+(hat-a_hc)*csigma;
    }
   else
    {
            z      = (hat+surface)*a_sigma+surface;   
    }
   
  return z;

}

bool ComputeMeshZProvider::zcoords3D(float * a_zCachePtr,const int & a_timeStep) const
{
 float*           surfacePtr;
   surfacePtr       = new float [m_number_node];

   retrieve2DVar (surfacePtr,
		          m_dataFilePtr,
                  a_timeStep,
                  MeshConstants::NODE_SURFACE);  
 
    float * nodeDepthPtr  = new float [m_number_node];
    
 
    retrieve1DVar(nodeDepthPtr,
	            m_dataFilePtr,
                MeshConstants::NODE_DEPTH,
                m_number_node);

	for (int iLayer= 0; iLayer<m_number_layer;iLayer++)
    {
		 
        for(int iNode=0;iNode <m_number_node; iNode++)
        {
			 
			if (iLayer>=(std::max(1,m_kbp00[iNode])-1))
			{
				float sigma        = m_layerSCoords[iLayer]; 
			  
				float surface      = surfacePtr[iNode];
             
				float depth        = nodeDepthPtr[iNode];    
			  
				float z            = convertStoZ(sigma,
												surface,
												depth,
												m_hs,
												m_hc,
												m_thetab,
												m_thetaf);
				*a_zCachePtr++         = z;
			}
        }

    }


	//delete layerSCoords;
	delete nodeDepthPtr;
	delete surfacePtr;
	return true;
}


bool ComputeMeshZProvider::zcoords3D2(float * a_zCachePtr,const int & a_timeStep) const
{
   float*           surfacePtr;
   surfacePtr       = new float [m_number_node];

    retrieve2DVar (surfacePtr,
		            m_dataFilePtr,
                    a_timeStep,
                    MeshConstants::NODE_SURFACE);


    float * nodeDepthPtr  = new float [m_number_node];
    
 
    retrieve1DVar(nodeDepthPtr,
	            m_dataFilePtr,
                MeshConstants::NODE_DEPTH,
                m_number_node);

	
		 
    for(int iNode=0;iNode <m_number_node; iNode++)
    {
	    for (int iLayer= 0; iLayer<m_number_layer;iLayer++)
        {
			if (iLayer>=(std::max(1,m_kbp00[iNode])-1))
			{
				float sigma        = m_layerSCoords[iLayer]; 
			  
				float surface      = surfacePtr[iNode];
             
				float depth        = nodeDepthPtr[iNode];    
			  
				float z            = convertStoZ(sigma,
												surface,
												depth,
												m_hs,
												m_hc,
												m_thetab,
												m_thetaf);
				*a_zCachePtr++         = z;
			}
        }

    }


	//delete layerSCoords;
	delete nodeDepthPtr;
	delete surfacePtr;
	return true;
}

