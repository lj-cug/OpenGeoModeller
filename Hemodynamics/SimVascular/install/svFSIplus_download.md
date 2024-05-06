# svFSIplus download
git clone https://github.com/SimVascular/svFSIplus

# test suite
cd svFSIplus

git lfs pull

After performing these steps once, younever need to worry about Git LFS again. All large files are handled automaticallyduring all Git operations, like push, pull, or commit.

## Running tests with pytest
cd ./tests/cases/<physics>/<test>

svFSIplus svFSI.xml

mpiexec -np 4 svFSIplus solver_params.xml

### pytest

## Create a new test


