Catalyst 2.0-enabled LULESH
===========================

This is a fork of LULESH 2.0 that adds support for in situ processing using
Catalyst 2.0 (released with ParaView 5.9.0)

The codebase has the following changes:
* [`Makefile`](Makefile): updates to include new source files and add a new variable
  `CATALYST_ROOT` which must be set to point to the install prefix / root for
  Catalyst install tree.
* [`lulesh-init.cc`](lulesh-init.cc): updates to build a `conduit_node` for the mesh when Catalyst
  is enabled
* [`lulesh-util.cc`](lulesh-util.cc): updates to add a command line argument `-x` which can be
  used to pass Python scripts to execute to Catalyst.
* [`lulesh.cc`](lulesh.cc): call catalyst init/execute/finalize methods at appropriate
  locations in the simulation loop.
* [`script.py`](script.py): a sample ParaView-Catalyst analysis script


Build instructions
==================

Obtaining and Building Catalyst
-------------------------------

First, you will need to fetch and build Catalyst from the [official
repository](https://gitlab.kitware.com/paraview/catalyst).

```
> mkdir ..../catalyst
> cd ..../catalyst
> git clone --branch for/paraview-5.9 https://gitlab.kitware.com/paraview/catalyst.git src
> mkdir build
> cd build
> cmake -G Ninja -DCATALYST_BUILD_TESTING:BOOL=OFF -DCMAKE_INSTALL_PREFIX=/packages/catalyst ../src

# do the build
> ninja

# do the install
> ninja install
```

Refer to [Catalyst documentation](https://catalyst-in-situ.readthedocs.io/en/latest/build_and_install.html)
for details on variables available.

Building LULESH
----------------

* Edit Makefile to update variables such as `MPICXX`, `CXX`, `CATALYST_ROOT` to
  point to your build environment.
* `CATALYST_ROOT` must be set to the `CMAKE_INSTALL_PREFIX` specified when
  building Catalyst.
* Then, use `make` to build lulesh. This will generate `lulesh2.0` executable.


Executing LULESH
================

To run LULESH with default setup, simply use

```
> ./lulesh2.0
```

To run using MPI, if MPI was enabled in the build
```
> mpirun -np <ranks> ./lulesh2.0
```

Either of the forms will use the Catalyst stub but will do nothing
consequential.

Using ParaView-Catalyst
------------------------

To execute with ParaView-Catalyst instead of the stub, you will need a standard
ParaView build with Python support enabled. You can also use ParaView 5.9.1
binaries. However, note that in that case, if you build LULESH with MPI enabled,
you must using `MPICH` to build LULESH, since ParaView 5.9.1 binaries are
built with MPICH and MPI implementation.

```
> env LD_LIBRARY_PATH=/apps/ParaView-5.9.1/lib ./lulesh2.0 -x script.py -p -i 10
```

On my Linux system, this generates the following output

```
Running problem size 30^3 per domain until completion
Num processors: 1
Num threads: 48
Total number of elements: 27000

To run other sizes, use -s <integer>.
To run a fixed number of iterations, use -i <integer>.
To run a more or less balanced region set, use -b <integer>.
To change the relative costs of regions, use -c <integer>.
To print out progress, use -p
To write an output file for VisIt, use -v
To use a Catalyst script, use -x (requires Catalyst-enabled build)
See help (-h) for more options

cycle = 1, time = 3.876087e-06, dt=3.876087e-06
VisRTX 0.1.6, using devices:
 0: Quadro P4000 (Total: 8.5 GB, Available: 7.9 GB)
(   3.202s) [pvbatch         ]        v2_internals.py:150   WARN| Module 'script' missing Catalyst 'options', will use a default options object
saving results in '<snip>/results'
cycle = 2, time = 8.527392e-06, dt=4.651305e-06
cycle = 3, time = 1.012168e-05, dt=1.594290e-06
cycle = 4, time = 1.145304e-05, dt=1.331356e-06
cycle = 5, time = 1.263989e-05, dt=1.186852e-06
cycle = 6, time = 1.374194e-05, dt=1.102046e-06
cycle = 7, time = 1.478970e-05, dt=1.047762e-06
cycle = 8, time = 1.580172e-05, dt=1.012024e-06
cycle = 9, time = 1.679057e-05, dt=9.888451e-07
cycle = 10, time = 1.776547e-05, dt=9.749005e-07
Run completed:
   Problem size        =  30
   MPI tasks           =  1
   Iteration count     =  10
   Final Origin Energy = 7.011263e+06
   Testing Plane 0 of Energy Array on rank 0:
        MaxAbsDiff   = 4.365575e-11
        TotalAbsDiff = 4.376588e-11
        MaxRelDiff   = 1.406918e-14


Elapsed time         =       1.86 (s)
Grind time (us/z/c)  =  6.8933593 (per dom)  ( 6.8933593 overall)
FOM                  =  145.06715 (z/s)
```

Note the warning message coming from ParaView-Catalyst (nothing to worry about
at this point). And the `saving results in ...` message printed from
`script.py`.

You will see data files and images for 10 timesteps saved in `results` directory
next to the script. The script currently fails if the directory already exists
to avoid overwriting results. You can modify the `script.py`, if needed.
