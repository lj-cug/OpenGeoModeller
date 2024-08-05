#include "CatalystAdaptor.h"
#include "Mandelbrot.h"

int main(int argc, char *argv[]) {
  Mandelbrot brot(/*origin=*/{-1.75, -1.25, 0., 0.},
                  /*size=*/{2.5, 2.5, 2.0, 1.5},
                  /*dimensions=*/{250, 250, 250},
                  /*projectionAxes=*/{0, 1, 3});
  CatalystAdaptor::Initialize(argc, argv);
  std::size_t nt = 100;
  // brot.Compute(/*maxIters=*/nt);

  for (std::size_t t = 0; t < nt; ++t) {
    std::size_t timestep = t + 1;
    brot.Compute(/*maxIters=*/timestep);
    CatalystAdaptor::Execute(t, timestep, brot);
  }
  CatalystAdaptor::Finalize();
}
