# pre-requities
# we need gcc-10.0 g++-10.0 ninjia

# If you are on an older Ubuntu version you can install a newer compiler from the ubuntu-toolchain-r/test-repository (with the following steps e.g. you can install GCC 10.3.0 on Ubuntu 20.04):
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install gcc-10
sudo apt-get install g++-10

update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 60 \
  --slave /usr/bin/g++ g++ /usr/bin/g++-10

apt-get install ninja-build
apt-get install python3 python3-pip

# Optional: Install Qt and other dependencies for the Data Explorer
pip install aqtinstall
mkdir /opt/qt
cd /opt/qt
aqt install-qt linux desktop 5.15.2 gcc_64
aqt install-qt linux desktop 5.15.2 gcc_64 --archives qtxmlpatterns qtx11extras

# Make sure to add /opt/qt/5.15.2/gcc_64/bin to the PATH.
export PATH=/opt/qt/5.15.2/gcc_64/bin:$PATH

# Install more dependencies for VTK rendering and for NetCDF IO
apt-get install freeglut3 freeglut3-dev libglew-dev libglu1-mesa libglu1-mesa-dev \
  libgl1-mesa-glx libgl1-mesa-dev libnetcdf-c++4-dev


# download
git clone -b 6.4.4 https://gitlab.opengeosys.org/ogs/ogs.git

# instructs git to only fetch files which are smaller than 100 Kilobyte. Larger files (e.g. benchmark files, images, PDFs) are fetched on-demand only.
git clone --filter=blob:limit=100 https://gitlab.opengeosys.org/ogs/ogs.git

# or 
wget https://gitlab.opengeosys.org/ogs/ogs/-/archive/master/ogs-master.tar.gz
tar xf ogs-master.tar.gz

cd ogs

# build OGS
# Configure manually
# in ogs source-directory
mkdir -p ../build/release
cd ../build/release
cmake ../../ogs -G Ninja -DCMAKE_BUILD_TYPE=Release

# or use MPI compilers
CC=mpicc CXX=mpic++ cmake ../ogs -G Ninja -DCMAKE_BUILD_TYPE=Release -DOGS_USE_PETSC=ON

# To compile OGS with PETSc, MPI c++ compiler wrappers like OpenMPI, MPICH ( has many derivatives, e.g intelMPI) has to be installed as prerequisite.

# For CMake configuration, the option of OGS_USE_PETSC must be set to true.
cmake -DOGS_USE_PETSC=true -DCMAKE_INSTALL_PREFIX=/to/your/path ..

# Install PETSc manually
./configure PETSC_ARCH=linux-fast  COPTFLAGS='-O3  --prefix=/home/me/opt/petsc --with-debugging=0 -march=native -mtune=native' CXXOPTFLAGS='-O3 -march=native -mtune=native' --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90 --download-fblaslapack --download-metis --download-parmetis --download-superlu_dist --download-scalapack --download-mumps  --download-hypre --with-c2html=0  --with-cxx-dialect=C++11 --with-cuda=0`

# Build the project
# With ninja
cd ../build/release

# By default, ninja uses maximum parallelism which may lead to a memory consumption exceeding the available memory (especially when compiling the processes).
ninja -j4

# Waiting
# Now the build process is running… This can take some time because maybe there are external libraries which get automatically downloaded and compiled. This step is only done once per build directory, so subsequent builds will be much faster.

# Finished 
