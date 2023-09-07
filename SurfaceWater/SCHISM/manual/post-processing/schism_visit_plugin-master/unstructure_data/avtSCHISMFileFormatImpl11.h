#ifndef AVTSCHISMFILEFORMATIMPL11_H
#define AVTSCHISMFILEFORMATIMPL11_H
#include <vector>
#include <map>
#include <vtkUnstructuredGrid.h>
#include "SCHISMFile10.h"
#include "MeshProvider10.h"
#include "FileFormatFavorInterface.h"
#include "avtSCHISMFileFormatImpl10.h"

// this Implementation change x,y coordates to double precision to match new double xy schsim output
class avtSCHISMFileFormatImpl11 : public avtSCHISMFileFormatImpl10
{
  public:
                    avtSCHISMFileFormatImpl11();
  virtual           ~avtSCHISMFileFormatImpl11() {;};

  static           FileFormatFavorInterface * create();


  private:
	  void           create2DUnstructuredMesh( vtkUnstructuredGrid *a_uGrid,
	                                       long                 *a_meshEle,
										   const  int          &a_timeState);

      void           create2DUnstructuredMeshNoDryWet( vtkUnstructuredGrid *a_uGrid,
	                                                  long                 *a_meshEle,
												      const  int          &a_timeState);

      void           create3DUnstructuredMesh(vtkUnstructuredGrid *a_uGrid,
	                                      long                 *a_meshEle,
										  long                 *a_2DPointto3DPoints,
										  const  int          &a_timeState);

      void           createLayerMesh(vtkUnstructuredGrid *a_uGrid,
	                             long                 *a_meshEle,
								 long                 *a_2DPointto3DPoints,
							     const  int          &a_timeState);

      void           create2DPointMesh( vtkUnstructuredGrid *a_uGrid,
	                                long                 *a_meshEle,
								    const  int          &a_timeState);

      void           create3DPointMesh( vtkUnstructuredGrid *a_uGrid,
	                                long                 *a_meshEle,
								    const  int          &a_timeState);

      void           create3DPointFaceMesh( vtkUnstructuredGrid *a_uGrid,
	                                    long                *a_meshEle,
								        const  int          &a_timeState);
};

#endif