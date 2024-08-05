#ifndef __IN_SITU_ADAPTOR_HEADER
#define __IN_SITU_ADAPTOR_HEADER

#include <string>

class Mandelbulb;

namespace InSitu
{
    void Initialize(const std::string& script);

    void Finalize();

    void CoProcess(Mandelbulb& mandelbulb, int nprocs, int rank, double time, unsigned int timeStep);
}

#endif
