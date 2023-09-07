#include <avtSCHISMFileFormatImpl11.h>

#include <string>
#include <iostream>
#include <sstream>
#include <time.h>
#include <math.h>

#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkRectilinearGrid.h>
#include <vtkRectilinearGrid.h>
#include <vtkUnstructuredGrid.h>
#include <vtkSmartPointer.h>
#include <vtkCellArray.h>
#include <vtkPolyhedron.h>
#include <vtkPlaneSource.h>
#include <vtkMath.h>
#include <vtkPoints.h>
#include <vtkCellType.h> 

#include <avtDatabaseMetaData.h>
#include <avtVariableCache.h>

#include <DBOptionsAttributes.h>
#include <Expression.h>

#include <InvalidVariableException.h>
#include <InvalidDBTypeException.h>
#include <InvalidTimeStepException.h>
#include <InvalidFilesException.h>
#include <DBYieldedNoDataException.h>
#include <DebugStream.h>
//L3 #include <malloc.h>
#if defined(__MACH__)
#include <stdlib.h>
#else
#include <malloc.h>
#endif


#include "ZCoordFileMeshProvider10.h"
#include "SCHISMFileUtil10.h"
#include "Average.h"
#include "MeshConstants10.h"
#include "NetcdfSCHISMOutput10.h"
#include "avtSCHISMFileFormat.h"
#include "Registar.h"

using     std::string;
using     std::stringstream;

const std::string NODE      = MeshConstants10::NODE;
const std::string FACE      = MeshConstants10::ELEM;
const std::string SIDE      = MeshConstants10::EDGE;
const std::string UNKOWN    ="unkown";
const int NODESPERELE       = MeshConstants10::MAX_NUM_NODE_PER_CELL;
const int NODESPERWEDGE     = NODESPERELE*2;


avtSCHISMFileFormatImpl11::avtSCHISMFileFormatImpl11():avtSCHISMFileFormatImpl10()
{
  debug1<<"nc xy double created\n";
}

FileFormatFavorInterface * avtSCHISMFileFormatImpl11::create()
{
	return new avtSCHISMFileFormatImpl11();
}



void   avtSCHISMFileFormatImpl11::create2DUnstructuredMesh( vtkUnstructuredGrid *a_uGrid,
	                                                  long                 *a_meshEle,
												      const  int          &a_timeState) 
{
	long   numNodes           = m_num_mesh_nodes;
	vtkPoints *points      = vtkPoints::New();
	points->SetDataTypeToDouble();
	points->GetData()->SetNumberOfComponents(3);
	points->SetNumberOfPoints(numNodes);
	
    
    double * pointPtr       = (double *) points->GetVoidPointer(0);
    debug1<<"begin  filling mesh double xy \n";    
	
    if (!m_external_mesh_provider->fillPointCoord2D(pointPtr,a_timeState))
    {
        stringstream msgStream(stringstream::out);
        msgStream <<"Fail to retrieve faces nodes coord at step " <<a_timeState;
        EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
    }
	
    debug1<<"finish filling mesh xy \n";
    a_uGrid ->SetPoints(points);
    points->Delete();
    a_uGrid ->Allocate( m_num_mesh_faces);
       
    long *  nodePtrTemp = a_meshEle;
	load_ele_dry_wet(a_timeState);
    for(long iCell = 0; iCell < m_num_mesh_faces; ++iCell)
        {
			int numberOfNodeInCell = *nodePtrTemp;
			
			//if (!(m_ele_dry_wet[iCell]))
			//{
				if (numberOfNodeInCell ==3)
				{
				vtkIdType verts[3];
				for(int iNode=0;iNode<3;++iNode)
				{
					verts[iNode] = nodePtrTemp[iNode+1]-1;
				    
				} 
				nodePtrTemp += (MeshConstants10::MAX_NUM_NODE_PER_CELL+1) ;
				 
				a_uGrid->InsertNextCell(VTK_TRIANGLE, 3, verts);
				}
				else if (numberOfNodeInCell ==4)
				{
				vtkIdType verts[4];
				for(int iNode=0;iNode<4;++iNode)
				{
					verts[iNode] = nodePtrTemp[iNode+1]-1;
				  
				} 
				nodePtrTemp += (MeshConstants10::MAX_NUM_NODE_PER_CELL+1);
				 
				a_uGrid->InsertNextCell(VTK_QUAD, 4, verts);
				}
				else
				{
				  stringstream msgStream(stringstream::out);
				  msgStream <<"invalid cell type with number of nodes: " <<numberOfNodeInCell;
				  EXCEPTION1(InvalidVariableException,msgStream.str());
				}
			//}
             
        }
      
}


void   avtSCHISMFileFormatImpl11::create2DUnstructuredMeshNoDryWet( vtkUnstructuredGrid *a_uGrid,
	                                                  long                 *a_meshEle,
												      const  int          &a_timeState) 
{
	long   numNodes           = m_num_mesh_nodes;
	vtkPoints *points      = vtkPoints::New();
	points->SetDataTypeToDouble();
	points->GetData()->SetNumberOfComponents(3);
    points->SetNumberOfPoints(numNodes);
    double * pointPtr       = (double *) points->GetVoidPointer(0);
        
    if (!m_external_mesh_provider->fillPointCoord2D(pointPtr,a_timeState))
    {
        stringstream msgStream(stringstream::out);
        msgStream <<"Fail to retrieve faces nodes coord at step " <<a_timeState;
        EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
    }


    a_uGrid ->SetPoints(points);
    points->Delete();
    a_uGrid ->Allocate( m_num_mesh_faces);
       
    long *  nodePtrTemp = a_meshEle;
	 
    for(long iCell = 0; iCell < m_num_mesh_faces; ++iCell)
        {
			int numberOfNodeInCell = *nodePtrTemp;
			
			
			if (numberOfNodeInCell ==3)
			{
			vtkIdType verts[3];
			for(int iNode=0;iNode<3;++iNode)
			{
				verts[iNode] = nodePtrTemp[iNode+1]-1;
				    
			} 
			nodePtrTemp += (MeshConstants10::MAX_NUM_NODE_PER_CELL+1) ;
				 
			a_uGrid->InsertNextCell(VTK_TRIANGLE, 3, verts);
			}
			else if (numberOfNodeInCell ==4)
			{
			vtkIdType verts[4];
			for(int iNode=0;iNode<4;++iNode)
			{
				verts[iNode] = nodePtrTemp[iNode+1]-1;
				  
			} 
			nodePtrTemp += (MeshConstants10::MAX_NUM_NODE_PER_CELL+1);
				 
			a_uGrid->InsertNextCell(VTK_QUAD, 4, verts);
			}
			else
			{
				stringstream msgStream(stringstream::out);
				msgStream <<"invalid cell type with number of nodes: " <<numberOfNodeInCell;
				EXCEPTION1(InvalidVariableException,msgStream.str());
			}
			
             
        }
      
}


void   avtSCHISMFileFormatImpl11::createLayerMesh(vtkUnstructuredGrid *a_uGrid,
	                                        long                 *a_meshEle,
										    long                *a_2DPointto3DPoints,
										    const  int          &a_timeState) 
{
	  vtkPoints *points      = vtkPoints::New();
	  points->SetDataTypeToDouble();
	  points->GetData()->SetNumberOfComponents(3);
      points->SetNumberOfPoints(m_total_valid_3D_point);
      double * pointPtr       = (double *) points->GetVoidPointer(0);
	  //debug only
	 
	  if (!m_external_mesh_provider->fillPointCoord3D(pointPtr,a_timeState))
        {
          stringstream msgStream(stringstream::out);
          msgStream <<"Fail to retrieve faces nodes coord at step " <<a_timeState;
          EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
        }
	
      a_uGrid->SetPoints(points);
      points->Delete();
      a_uGrid->Allocate( m_num_mesh_faces*m_num_layers);
      
     
	  int * kbe = m_kbp_ele;
	  debug1<<"test layer mesh ele\n";
      for (int iLayer= 0; iLayer<m_num_layers;iLayer++)
        {
		 
          for(int iCell = 0; iCell < m_num_mesh_faces; ++iCell)
            {
			
			  if (iLayer>=(std::max(1,kbe[iCell])-1))
			  {
				
				  int numberOfNodeInCell = a_meshEle[iCell*(NODESPERELE+1)];

				  if (numberOfNodeInCell ==3)
				  {
					 vtkIdType verts[3]; 

					 for(int i=0;i<3;i++)
					 {
						int p = a_meshEle[iCell*(NODESPERELE+1)+i+1]-1;
						int p3d =  a_2DPointto3DPoints[p*m_num_layers+iLayer];
						int valid_bottom = std::max(1,m_kbp_node[p])-1;
						if (iLayer<valid_bottom)
						{
							p3d = a_2DPointto3DPoints[p*m_num_layers+valid_bottom];
						}

						verts[i]=p3d;
					 }

					 a_uGrid->InsertNextCell(VTK_TRIANGLE, 3, verts);
				  }
				  else if  (numberOfNodeInCell ==4)
				  {
					 vtkIdType verts[4]; 
					 for(int i=0;i<4;i++)
					 {
						int p = a_meshEle[iCell*(NODESPERELE+1)+i+1]-1;
						int p3d =  a_2DPointto3DPoints[p*m_num_layers+iLayer];
						int valid_bottom = std::max(1,m_kbp_node[p])-1;
						if (iLayer<valid_bottom)
						{
							p3d = a_2DPointto3DPoints[p*m_num_layers+valid_bottom];
						}

						verts[i]=p3d;
					 }
					 a_uGrid->InsertNextCell(VTK_QUAD, 4, verts);
				  }
				  else
		          {
			        stringstream msgStream(stringstream::out);
                    msgStream <<"invalid cell type with number of nodes: " <<numberOfNodeInCell;
			        EXCEPTION1(InvalidVariableException,msgStream.str());
		          }
			  }
			  
            }
		 
        }
	 
	 
}


void   avtSCHISMFileFormatImpl11::create3DUnstructuredMesh(vtkUnstructuredGrid *a_uGrid,
	                                                 long                 *a_meshEle,
												     long                 *a_2DPointto3DPoints,
												     const  int          &a_timeState) 
{
	 vtkPoints *points      = vtkPoints::New();
	 points->SetDataTypeToDouble();
	 points->GetData()->SetNumberOfComponents(3);
     points->SetNumberOfPoints(m_total_valid_3D_point);
	  debug1<<"total valid 3d pts: "<<m_total_valid_3D_point<<"\n";
      double * pointPtr       = (double *) points->GetVoidPointer(0);
	  debug1 << "pointer of mesh double xy " << m_external_mesh_provider << "\n";
	  if (!m_external_mesh_provider->fillPointCoord3D(pointPtr,a_timeState))
        {
          stringstream msgStream(stringstream::out);
          msgStream <<"Fail to retrieve faces nodes coord at step " <<a_timeState;
          EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
        }
	 

      debug1<<"finish compute and cahce z\n";
      a_uGrid->SetPoints(points);
      points->Delete();
      a_uGrid->Allocate( m_num_mesh_faces*(m_num_layers-1));
      debug1<<"total 2d cell num "<<m_num_mesh_faces<<"\n";
      long *  nodePtrTemp = a_meshEle;
	  
	
	  m_tri_wedge=0;
      m_tri_pyramid=0;
      m_tri_tetra=0;

      m_quad_hexhedron=0;
      m_quad_wedge=0;
      m_quad_pyramid=0;
	  load_ele_dry_wet(a_timeState);

       for (int iLayer= 0; iLayer<m_num_layers-1;iLayer++)
        {
          nodePtrTemp    = a_meshEle;
           for(int iCell = 0; iCell < m_num_mesh_faces; ++iCell)
            {
              {
				  nodePtrTemp = a_meshEle+(MeshConstants10::MAX_NUM_NODE_PER_CELL+1)*iCell;
				  int numberOfNodeInCell = *nodePtrTemp;

				  long validTopNode[MeshConstants10::MAX_NUM_NODE_PER_CELL];
				  long validBottomNode[MeshConstants10::MAX_NUM_NODE_PER_CELL];
				  int validTopNodeNum    =0;
				  int validBottomNodeNum =0;

				  validTopBottomNode(validTopNodeNum,
									 validBottomNodeNum,
									 validTopNode,
									 validBottomNode,
									 iLayer,
									 nodePtrTemp);
			 

				  if (numberOfNodeInCell ==3)   
				  {
					 insertTriangle3DCell(a_uGrid,
										  validTopNodeNum,
										  validBottomNodeNum,
										  validTopNode,
										  validBottomNode,
										  nodePtrTemp,
										  a_2DPointto3DPoints,
										  iCell,
										  iLayer);
					 //move pointer to next element
					 //nodePtrTemp += (MeshConstants10::MAX_NUM_NODE_PER_CELL+1);
				 }
				 else if (numberOfNodeInCell ==4)
				 {
					  insertQuad3DCell(a_uGrid,
									   validTopNodeNum,
									   validBottomNodeNum,
										  validTopNode,
										  validBottomNode,
										  nodePtrTemp,
										  a_2DPointto3DPoints,
										  iCell,
										  iLayer);
					// nodePtrTemp += (MeshConstants10::MAX_NUM_NODE_PER_CELL+1);
				  }
                
				 else
				 {
				   //omit
				 }
                
			}
		  }
           
        }

	  debug1<<" tri_wedge "<<m_tri_wedge<<" tri_pyramid "<<m_tri_pyramid<<" tri_tetra "<<m_tri_tetra;

	  debug1<<" quad_hexhedron "<<m_quad_hexhedron<<" quad_wedge "<<m_quad_wedge<<" quad_pyramid "<<m_quad_pyramid<<"\n";
}

void    avtSCHISMFileFormatImpl11::create2DPointMesh( vtkUnstructuredGrid *a_uGrid,
	                                            long                 *a_meshEle,
										        const  int          &a_timeState) 
{
	long   numNodes           = m_num_mesh_edges;
	vtkPoints *points      = vtkPoints::New();
	points->SetDataTypeToDouble();
	points->GetData()->SetNumberOfComponents(3);
    points->SetNumberOfPoints(numNodes);
    double * pointPtr       = (double *) points->GetVoidPointer(0);
	debug1 << "before load side center 2d xy\n";
    if (!m_external_mesh_provider->fillSideCenterCoord2D(pointPtr,a_timeState))
    {
        stringstream msgStream(stringstream::out);
        msgStream <<"Fail to retrieve edge center coord at step " <<a_timeState;
        EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
    }

	debug1 << "done load side center 2d xy\n";
    a_uGrid ->SetPoints(points);
    points->Delete();
	a_uGrid->Allocate(numNodes);
    vtkIdType onevertex;
    for(long i = 0; i < numNodes; ++i)
    {
       onevertex = i;
       a_uGrid->InsertNextCell(VTK_VERTEX, 1, &onevertex);
    }
 
}

void   avtSCHISMFileFormatImpl11::create3DPointMesh( vtkUnstructuredGrid *a_uGrid,
	                                                long            *a_meshEle,
										            const  int     &a_timeState) 
{
	long   numNodes           = m_total_valid_3D_side;
	debug1 << "total num of side 3d " << m_total_valid_3D_side << "\n";
	vtkPoints *points      = vtkPoints::New();
	points->SetDataTypeToDouble();
	points->GetData()->SetNumberOfComponents(3);
    points->SetNumberOfPoints(numNodes);
    double * pointPtr       = (double *) points->GetVoidPointer(0);

    if (!m_external_mesh_provider->fillSideCenterCoord3D(pointPtr,a_timeState))
    {
        stringstream msgStream(stringstream::out);
        msgStream <<"Fail to retrieve edge center coord at step " <<a_timeState;
        EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
    }

    a_uGrid ->SetPoints(points);
    points->Delete();
	a_uGrid->Allocate(numNodes);
    vtkIdType onevertex;
    for(long i = 0; i < numNodes; ++i)
     { 
	   onevertex = i;
       a_uGrid->InsertNextCell(VTK_VERTEX, 1, &onevertex);
    }
 
}

// this is the mesh consisits of center at 3d prism side face, used for var like flux 
void   avtSCHISMFileFormatImpl11::create3DPointFaceMesh( vtkUnstructuredGrid *a_uGrid,
	                                                long               *a_meshEle,
										            const  int         &a_timeState) 
{
	int   numNodes           = m_total_valid_3D_side-(m_external_mesh_provider->numberOfSide());
	vtkPoints *points      = vtkPoints::New();
	points->SetDataTypeToDouble();
	points->GetData()->SetNumberOfComponents(3);
    points->SetNumberOfPoints(numNodes);
    double * pointPtr       = (double *) points->GetVoidPointer(0);
        
    if (!m_external_mesh_provider->fillSideFaceCenterCoord3D(pointPtr,a_timeState))
    {
        stringstream msgStream(stringstream::out);
        msgStream <<"Fail to retrieve edge center coord at step " <<a_timeState;
        EXCEPTION3(DBYieldedNoDataException,m_data_file,m_plugin_name,msgStream.str());
    }

    a_uGrid ->SetPoints(points);
    points->Delete();
	a_uGrid->Allocate(numNodes);
    vtkIdType onevertex;
    for(int i = 0; i < numNodes; ++i)
    {
       onevertex = i;
       a_uGrid->InsertNextCell(VTK_VERTEX, 1, &onevertex);
    }
 
}



static Registrar registrar("combine10_nc4_double_xy", &avtSCHISMFileFormatImpl11::create);
