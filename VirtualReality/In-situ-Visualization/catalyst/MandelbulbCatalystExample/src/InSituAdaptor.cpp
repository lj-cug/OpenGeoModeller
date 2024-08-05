#include "InSituAdaptor.hpp"
#include "Mandelbulb.hpp"
#include <iostream>

#include <vtkCPDataDescription.h>
#include <vtkCPInputDataDescription.h>
#include <vtkCPProcessor.h>
#include <vtkCPPythonScriptPipeline.h>
#include <vtkCellData.h>
#include <vtkCellType.h>
#include <vtkCommunicator.h>
#include <vtkIntArray.h>
#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkMultiPieceDataSet.h>
#include <vtkMultiProcessController.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPoints.h>

#include <mpi.h>

namespace
{
    vtkCPProcessor* Processor = NULL;
    vtkMultiBlockDataSet* VTKGrid;

    void BuildVTKGrid(Mandelbulb& grid, int nprocs, int rank)
    {
        int* extents = grid.GetExtents();
        vtkNew<vtkImageData> imageData;
        imageData->SetSpacing(1.0/nprocs, 1, 1);
        imageData->SetExtent(extents);
        imageData->SetOrigin(grid.GetOrigin()); // Not necessary for (0,0,0)
        vtkNew<vtkMultiPieceDataSet> multiPiece;
        multiPiece->SetNumberOfPieces(nprocs);
        multiPiece->SetPiece(rank, imageData.GetPointer());
        VTKGrid->SetNumberOfBlocks(1);
        VTKGrid->SetBlock(0, multiPiece.GetPointer());
    }

    void UpdateVTKAttributes(Mandelbulb& mandelbulb,
            int rank,
            vtkCPInputDataDescription* idd)
    {
        vtkMultiPieceDataSet* multiPiece = vtkMultiPieceDataSet::SafeDownCast(VTKGrid->GetBlock(0));
        if (idd->IsFieldNeeded("mandelbulb", vtkDataObject::POINT))
        {
            vtkDataSet* dataSet = vtkDataSet::SafeDownCast(multiPiece->GetPiece(rank));
            if (dataSet->GetPointData()->GetNumberOfArrays() == 0)
            {
                // pressure array
                vtkNew<vtkIntArray> data;
                data->SetName("mandelbulb");
                data->SetNumberOfComponents(1);
                dataSet->GetPointData()->AddArray(data.GetPointer());
            }
            vtkIntArray* data =
                vtkIntArray::SafeDownCast(dataSet->GetPointData()->GetArray("mandelbulb"));
            // The pressure array is a scalar array so we can reuse
            // memory as long as we ordered the points properly.
            int* theData = mandelbulb.GetData();
            data->SetArray(theData, static_cast<vtkIdType>(mandelbulb.GetNumberOfLocalCells()), 1);
        }
    }

    void BuildVTKDataStructures(Mandelbulb& mandelbulb,
            int nprocs, int rank,
            vtkCPInputDataDescription* idd)
    {
        if (VTKGrid == NULL)
        {
            // The grid structure isn't changing so we only build it
            // the first time it's needed. If we needed the memory
            // we could delete it and rebuild as necessary.
            VTKGrid = vtkMultiBlockDataSet::New();
            BuildVTKGrid(mandelbulb, nprocs, rank);
        }
        UpdateVTKAttributes(mandelbulb, rank, idd);
    }
}

namespace InSitu
{

    void Initialize(const std::string& script)
    {
        if (Processor == NULL)
        {
            Processor = vtkCPProcessor::New();
            Processor->Initialize();
        }
        else
        {
            Processor->RemoveAllPipelines();
        }
        vtkNew<vtkCPPythonScriptPipeline> pipeline;
        pipeline->Initialize(script.c_str());
        Processor->AddPipeline(pipeline.GetPointer());
    }

    void Finalize()
    {
        if (Processor)
        {
            Processor->Delete();
            Processor = NULL;
        }
        if (VTKGrid)
        {
            VTKGrid->Delete();
            VTKGrid = NULL;
        }
    }

    void CoProcess(Mandelbulb& mandelbulb,
            int nprocs, int rank,
            double time, unsigned int timeStep)
    {
        vtkNew<vtkCPDataDescription> dataDescription;
        dataDescription->AddInput("input");
        dataDescription->SetTimeData(time, timeStep);
        if (Processor->RequestDataDescription(dataDescription.GetPointer()) != 0)
        {
            vtkCPInputDataDescription* idd = dataDescription->GetInputDescriptionByName("input");
            BuildVTKDataStructures(mandelbulb, nprocs, rank, idd);
            idd->SetGrid(VTKGrid);

            Processor->CoProcess(dataDescription.GetPointer());
        }
    }
}
