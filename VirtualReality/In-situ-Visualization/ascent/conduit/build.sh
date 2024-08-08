# Installing Conduit and Third Party Dependencies
git clone --recursive https://github.com/llnl/conduit.git
cd conduit
python3 scripts/uberenv/uberenv.py --install --prefix="build"


# or Installing Conduit with pip
# If you want to use Conduit primarily in Python, another option is to build Conduit with pip.
git clone --recursive https://github.com/llnl/conduit.git
cd conduit
pip install . --user

## or If you have a system MPI and an existing HDF5 install you can add those to the build using environment variables.
git clone --recursive https://github.com/llnl/conduit.git
cd conduit
env ENABLE_MPI=ON HDF5_DIR={path/to/hdf5_dir} pip install . --user

# Using Conduit in Your Project
## CMake-based build system example (see: examples/conduit/using-with-cmake):
## Makefile-based build system example (see: examples/conduit/using-with-make):
