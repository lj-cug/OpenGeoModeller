## Visualize the iterations for determining complex numbers in a mandelbrot set

Evaluate a mandelbrot set using CUDA and visualize each iteration with ParaView Catalyst2.
Consider the equation `Z = Z^2 + C`, in the 4D space made up of `Creal`, `Cimag`, `Zreal` and `Zimag`,
this visualization maps the x-axis to `Creal`, y-axis to `Cimag` and z-axis to `Zimag`.

![Mandelbrot visualization](screenshot.png)

## Build

```sh
git clone https://github.com/jspanchu/MandelbrotCatalyst2
cmake -S . -B out -GNinja -Dcatalyst_DIR=/path/to/catalyst/lib/cmake/catalyst-2.0
cmake --build out
```

## Run

```sh
export CATAYLST_IMPLEMENTATION_PATHS='/path/to/paraview/install/lib/catalyst'
export CATALYST_IMPLEMENTATION_NAME='paraview'
./out/bin/Mandelbrot ./pipeline.py

# PNG images will be written out to `out/datasets/screenshot_0000mn.png`
```
