#include "Mandelbrot.h"

#include <iostream>
#include <iterator>

#define CHECKED_CUDA_INVOKE(call)                                              \
  do {                                                                         \
    call;                                                                      \
    auto err = cudaGetLastError();                                             \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA ERROR: " << cudaGetErrorString(err) << " in "         \
                << #call << " at " << __FILE__ << ':' << __LINE__              \
                << std::endl;                                                  \
    }                                                                          \
  } while (0)

Mandelbrot::Mandelbrot(std::array<double, 4> origin, std::array<double, 4> size,
            std::array<std::size_t, 3> dimensions,
            std::array<std::size_t, 3> projectionAxes)
  : Origin(origin)
  , Size(size)
  , Dimensions(dimensions)
  , ProjectionAxes(projectionAxes)
{
  std::size_t numPoints = dimensions[0] * dimensions[1] * dimensions[2];
  CHECKED_CUDA_INVOKE(cudaStreamCreate(&this->IterationStream));
  CHECKED_CUDA_INVOKE(cudaMallocAsync(&this->Iterations, numPoints * sizeof(double), this->IterationStream));
  // Default 0.01 spacing in each dimension.
  std::fill(this->Spacings.begin(), this->Spacings.end(), 0.01);
  // For different projection axes, recompute spacings.
  if ((this->ProjectionAxes[0] != 0) || 
      (this->ProjectionAxes[1] != 1) ||
      (this->ProjectionAxes[2] != 2))
  {
    for (int idx = 0; idx < 3; ++idx)
    {
      const auto& length = this->Dimensions[idx];
      const auto& axis = this->ProjectionAxes[idx];
      this->Spacings[axis] = this->Size[axis] / length;
    }
  }
}

Mandelbrot::~Mandelbrot()
{
  if (this->Iterations != nullptr)
  {
    CHECKED_CUDA_INVOKE(cudaFree(this->Iterations));
    this->Iterations=  nullptr;
  }
  CHECKED_CUDA_INVOKE(cudaStreamDestroy(this->IterationStream));
  this->IterationStream = nullptr;
}

double *Mandelbrot::GetIterationsArray() const
{
  CHECKED_CUDA_INVOKE(cudaStreamSynchronize(this->IterationStream));
  return this->Iterations;
}

namespace
{
  __global__ void EvaluateMandelbrotSet(std::size_t nx, std::size_t ny, std::size_t nz,
                  std::size_t a0, std::size_t a1, std::size_t a2,
                  double o0, double o1, double o2, double o3,
                  double d0, double d1, double d2, double d3, std::size_t maxIters, double* iters)
  {
    const auto threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= nx * ny * nz)
    {
      return;
    }
    // z increases slowest, x increases fastest
    const std::size_t z = threadId / (ny * nx);
    const std::size_t temp = threadId - (z * ny * nx);
    const std::size_t y = temp / nx;
    const std::size_t x = temp % nx;

    double p[4] = { o0, o1, o2, o3 };
    const double o[4] = { o0, o1, o2, o3 };
    const double d[4] = { d0, d1, d2, d3 };
    const std::size_t axes[3] = { a0, a1, a2};
    const std::size_t ax_map[3] = { x, y, z };
    for (int dim = 0; dim < 3; ++dim)
    {
      const std::size_t& axis = axes[dim];
      p[axis] = o[axis] + ax_map[dim] * d[axis];
    }
    std::size_t count = 0;
    double v0, v1;
    double cReal, cImag, zReal, zImag;
    double zReal2, zImag2;

    cReal = p[0];
    cImag = p[1];
    zReal = p[2];
    zImag = p[3];

    zReal2 = zReal * zReal;
    zImag2 = zImag * zImag;
    v0 = 0.0;
    v1 = (zReal2 + zImag2);
    while (v1 < 4.0 && count < maxIters)
    {
      zImag = 2.0 * zReal * zImag + cImag;
      zReal = zReal2 - zImag2 + cReal;
      zReal2 = zReal * zReal;
      zImag2 = zImag * zImag;
      ++count;
      v0 = v1;
      v1 = (zReal2 + zImag2);
    }

    if (count == maxIters)
    {
      iters[threadId] = static_cast<double>(count);
    }
    else
    {
      iters[threadId] = static_cast<double>(count) + (4.0 - v0) / (v1 - v0);
    }
  }
}

void Mandelbrot::Compute(std::size_t maxIters)
{
  int blockSize = 512;
  int numThreads = this->Dimensions[0] * this->Dimensions[1] * this->Dimensions[2];
  int numBlocks = (numThreads + blockSize - 1) / blockSize;
  ::EvaluateMandelbrotSet<<<numBlocks, blockSize, 0, this->IterationStream>>>(
    this->Dimensions[0], this->Dimensions[1], this->Dimensions[2],
    this->ProjectionAxes[0], this->ProjectionAxes[1], this->ProjectionAxes[2],
    this->Origin[0], this->Origin[1], this->Origin[2], this->Origin[3],
    this->Spacings[0], this->Spacings[1], this->Spacings[2], this->Spacings[3],
    maxIters, this->Iterations);
  CHECKED_CUDA_INVOKE(void("After kernel 'EvaluateMandelbrotSet' was launched"));
}
