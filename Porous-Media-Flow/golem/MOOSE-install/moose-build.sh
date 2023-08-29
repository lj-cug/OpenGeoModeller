# Obtaining and Building MOOSE

# The master branch of MOOSE is the stable branch that will only be updated after all tests are passing. This protects you from the day-to-day changes in the MOOSE repository.
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/idaholab/moose.git
cd moose
git checkout master

# Build PETSc and libMesh
cd ~/projects/moose/scripts
export MOOSE_JOBS=6 METHODS=opt
./update_and_rebuild_petsc.sh
./update_and_rebuild_libmesh.sh

# Compile and Test MOOSE
cd ~/projects/moose/test
make -j 6
# To test MOOSE
cd ~/projects/moose/test
./run_tests -j 6

# If the installation was successful you should see most of the tests passing (some tests will be skipped depending on your system environment), and no failures.

# Now that you have a working MOOSE, proceed to 'New Users' to begin your tour of MOOSE!
