#ifndef Mandelbrot_h
#define Mandelbrot_h

#include <array>

typedef struct CUstream_st *cudaStream_t;

class Mandelbrot {
public:
  /**
   * @param origin Initial values of Creal, Cimag, Zreal and Zimag,
   * @param size   Size of the 4D space for Creal, Cimag, Zreal and Zimag
   * @param dimensions No. of dims in 3D space. The 4D space will be projected
   *                   on to a 3D space for visualization.
   * @param projectionAxes Map the 4D space indices into 3D x-y-z space used for visualization.
   *                       0: Creal, 1: Cimag, 2: Zreal, 3: Zimag
   */
  Mandelbrot(std::array<double, 4> origin, std::array<double, 4> size,
             std::array<std::size_t, 3> dimensions,
             std::array<std::size_t, 3> projectionAxes);

  ~Mandelbrot();
  
  std::array<double, 4> GetOrigin() const { return this->Origin; }
  
  std::array<double, 4> GetSpacings() const { return this->Spacings; }
  
  std::array<std::size_t, 3> GetDimensions() const { return this->Dimensions; }
  
  std::array<std::size_t, 3> GetProjectionAxes() const { return this->ProjectionAxes; }

  double *GetIterationsArray() const;

  void Compute(std::size_t maxIters);

private:
  cudaStream_t IterationStream = nullptr;
  double* Iterations = nullptr;
  std::array<double, 4> Origin;
  std::array<double, 4> Size;
  std::array<double, 4> Spacings;
  std::array<std::size_t, 3> Dimensions;
  std::array<std::size_t, 3> ProjectionAxes;
};

#endif
