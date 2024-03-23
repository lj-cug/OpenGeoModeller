# Hacking VTK for Parallelisation (MPI and C++)

https://gerstrong.github.io/blog/2016/08/20/hacking-vtk-for-parallelisation

Have you ever wondered how to fully use VTK (Visualisation Toolkit) for stuff you really want to accomplish?

In my case we have a code that uses VTK and we want to run it across several machines in parallel (Using OpenMPI). Kitware’s ParaView is a nice piece of software using VTK for the visualization of data and as the name says enforces parallelisation. For every VTK file extension (vtu, vtp, vts, etc.) there exist a parallel version (pvtu, pvtp, pvts, etc.) which actually has more or less meta-information about your data, like your point and cell data tags but it also defines a set of multiples files which can be assigned, the so called piece files.

So if you use MPI and let run everything in parallel even writing the data into separate files (.vtu) and reference all of them using the parallel file (.pvtu) you have the advantage that you can open your case in paraview through that one file.

Unfortunately there is not that much of documentation about how to use this class vtkXMLPUnstructuredGridWriter.

## 在各rank上创建.vtu文件

Let’s define some vtu files. The following code is run in parallel with MPI:

```
std::string SPHEREDATAFNAME = "sphereData_";

// Append process number
SPHEREDATAFNAME += std::to_string(gParallelMngr.getProcID());

SPHEREDATAFNAME += ".vtu";

// ... Your VTK Code here ...

 
// Write file
auto writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
writer->SetFileName(SPHEREDATAFNAME.c_str());
writer->SetInputData(unstructuredGrid);
writer->Write();
```

## 在master rank上创建.pvtu文件

Let’s say you have four parallel processes running. You are going to generate sphereData_0.vtu, sphereData_1.vtu, sphereData_2.vtu and sphereData_3.vtu. Now only one process is allowed to create the parallel file, let it be the master process (usually number 0):

```
if( gParallelMngr.isMaster() )
{
    const std::string PIECESDATAFNAME = "sphereData.pvtu";

    auto pwriterObj = vtkSmartPointer<vtkXMLPUnstructuredGridWriter>::New();

    pwriterObj->EncodeAppendedDataOff();
    pwriterObj->SetFileName(PIECESDATAFNAME.c_str());
    pwriterObj->SetNumberOfPieces( gParallelMngr.getNumProcs() );
    pwriterObj->SetInputData(unstructuredGrid);
    pwriterObj->Update();
}

```

There ya go! With pwriterObj->SetInputData(unstructuredGrid); the parallel file gets the metainformation and with SetFileName(...) and SetNumberOfPieces(...) it creates the reference names by that scheme.

So far this is very useful. I still have one problem, especially considering that the serial files are scattered across different machines. With this approach you can only use parallelisation on one local machine having the generated files in the same directory.

I asked myself, is it possible to set the parallel file paths manually? There is no VTK documentation in how to do that so far to my knowledge, but I found a way. Let me show you how. 
通常, 并行化文件.pvtu与.vtu必须在一个路径下，是否有方法可手动设置.pvtu的路径呢? 答案是：有!

Through the VTK classes there exist one virtual method where the attribute of theses references is set void WritePPieceAttributes(int index). Because it is virtual you can create your own class and overload with your code. Let’s do this by creating a new class:

```
#include "vtkXMLPUnstructuredGridWriter.h"

#include "vtkObjectFactory.h"
#include "vtkSetGet.h"

class LDEMVTKXMLPUnstructuredDataWriter : public vtkXMLPUnstructuredGridWriter
{
public:

    static LDEMVTKXMLPUnstructuredDataWriter* New();

    void WritePPieceAttributes(int index)
    {
        std::string DATAFNAME = "ProcData";
        DATAFNAME += std::to_string(index);
        DATAFNAME += "/spheredata_";
        DATAFNAME += std::to_string(index);
        DATAFNAME += ".vtu";

        this->WriteStringAttribute("Source", DATAFNAME);
    }

};
vtkStandardNewMacro(LDEMVTKXMLPUnstructuredDataWriter);
```

By overloading the WritePPieceAttributes() method you now can define your own rules.

Instead of creating a vtkXMLPUnstructuredGridWriter you create a LDEMVTKXMLPUnstructuredDataWriter instance or however you called the class, but you call the same VTK Methods as before:

```
if( gParallelMngr.isMaster() )
{
    const std::string PIECESDATAFNAME = "sphereData.pvtu";

    auto pwriterObj = vtkSmartPointer<LDEMVTKXMLPUnstructuredDataWriter>::New();

    pwriterObj->EncodeAppendedDataOff();
    pwriterObj->SetFileName(PIECESDATAFNAME.c_str());
    pwriterObj->SetNumberOfPieces( gParallelMngr.getNumProcs() );
    pwriterObj->SetInputData(unstructuredGrid);
    pwriterObj->Update();
}
```

That’s it! In my case I put another directory for every file like ProcData/spheredata_1.vtu, so I can mount them through SMB, NFS or similar and visuallize everything through the pvtu file.

## 为什么不集合数据形成一个大的文件呢?

Why write in parallel instead of collecting data and then create a big file? Because you can use the whole cpu nodes for everything and that the concept of MPI. If you visuallise the data with paraview you can collect the data you only require you will render what you only want to see and save important CPU ressources.

