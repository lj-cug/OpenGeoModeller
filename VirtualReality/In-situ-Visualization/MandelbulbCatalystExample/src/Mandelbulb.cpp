#include <iostream>
#include <mpi.h>
#include "Mandelbulb.hpp"
#include "InSituAdaptor.hpp"

int main(int argc, char** argv)
{
    MPI_Init(&argc,&argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if(argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <script.py>" << std::endl;
        exit(0);
    }

    std::string scriptname = argv[1];    

    InSitu::Initialize(scriptname);  // Catalyst初始化

    unsigned WIDTH  = 30;
    unsigned HEIGHT = 30;
    unsigned DEPTH  = 30;

    int offset_z = rank*DEPTH;

    Mandelbulb mandelbulb(WIDTH, HEIGHT, DEPTH, offset_z, 1.2, nprocs);
    
	// 时间层迭代
    for(int i=0; i < 1; i++) {
        double t_start, t_end;
        {
            double order = 4.0 + ((double)i)*8.0/100.0;
            MPI_Barrier(MPI_COMM_WORLD);
            t_start = MPI_Wtime();
            mandelbulb.compute(order);
            MPI_Barrier(MPI_COMM_WORLD);
            t_end = MPI_Wtime();
        }
        if(rank == 0) std::cout << "Computation " << i << " completed in " << (t_end-t_start) << " seconds." << std::endl;
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t_start = MPI_Wtime();
            InSitu::CoProcess(mandelbulb, nprocs, rank, i, i);  // Co-Process执行
            MPI_Barrier(MPI_COMM_WORLD);
            t_end = MPI_Wtime();
        }
        if(rank == 0) std::cout << "InSitu " << i << " completed in " << (t_end-t_start) << " seconds." << std::endl;
    }

    InSitu::Finalize();  // Catalyst结束

    MPI_Finalize();
    return 0;
}
