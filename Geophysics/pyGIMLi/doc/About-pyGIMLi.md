# About pyGIMLi
https://www.pygimli.org/about.html#sec-gimli

## Introduction
pyGIMLi is an open-source library for modelling and inversion and in geophysics. The object-oriented library provides management for structured and unstructured meshes in 2D and 3D, finite-element and finite-volume solvers, various geophysical forward operators, as well as Gauss-Newton based frameworks for constrained, joint and fully-coupled inversions with flexible regularization.

## What is pyGIMLi suited for?
analyze, visualize and invert geophysical data in a reproducible manner
forward modelling of (geo)physical problems on complex 2D and 3D geometries
inversion with flexible controls on a-priori information and regularization
combination of different methods in constrained, joint and fully-coupled inversions
teaching applied geophysics (e.g. in combination with Jupyter notebooks)

## What is pyGIMLi NOT suited for?
for people that expect a ready-made GUI for interpreting their data

## Authors
We gratefully acknowledge all contributors to the pyGIMLi open-source project and look forward to your contribution!

Carsten R��cker
Berlin University of Technology, Department of Applied Geophysics, Berlin, Germany
carsten@pygimli.org

Thomas G��nther
Leibniz Institute for Applied Geophysics, Hannover, Germany
thomas@pygimli.org

Florian Wagner
RWTH Aachen University, Geophysical Imaging and Monitoring (GIM), Aachen, Germany
florian@pygimli.org

Friedrich Dinsel
Berlin University of Technology, Department of Applied Geophysics, Berlin, Germany
friedrich@pygimli.org

Maximilian Weigand
University of Bonn, Department of Geophysics, Bonn, Germany
Andrea Balza
RWTH Aachen University, Geophysical Imaging and Monitoring (GIM), Aachen, Germany

## Inversion
One main task of pyGIMli is to carry out inversion, i.e. error-weighted minimization, for given forward routines and data. Various types of regularization on meshes (1D, 2D, 3D) with regular or irregular arrangement are available. There is flexible control of all inversion parameters. The default inversion framework is based on the generalized Gauss-Newton method.

Please see Inversion for examples and more details.

## Modelling
pyGIMLi comes with various geophysical forward operators, which can directly be used for a given problem. In addition, abstract finite-element and finite-volume interfaces are available to solve custom PDEs on a given mesh. See pygimli.physics for a collection of forward operators and pygimli.solver for the solver interface.

The modelling capabilities of pyGIMLi include:

1D, 2D, 3D discretizations

linear and quadratic shape functions (automatic shape function generator for possible higher order)

Triangle, Quads, Tetrahedron, Prism and Hexahedron, mixed meshes

solver for elliptic problems (Helmholtz-type PDE)

Please see Modelling for examples and more details.

## License
pyGIMLi is distributed under the terms of the Apache 2.0 license. Details on the license agreement can be found here.

## Credits
We use or link some third-party software (beside the usual tool stack: cmake, gcc, boost, python, numpy, scipy, matplotlib) and are grateful for all the work made by the authors of these awesome open-source tools:

libkdtree++: Maybe abandoned, mirror: nvmd/libkdtree

meshio: nschloe/meshio

pyplusplus: https://pypi.org/project/pyplusplus/

pyvista: https://docs.pyvista.org/

suitesparse, umfpack: https://people.engr.tamu.edu/davis/suitesparse.html

Tetgen: http://wias-berlin.de/software/index.jsp?id=TetGen&lang=1

Triangle: https://www.cs.cmu.edu/~quake/triangle.html
