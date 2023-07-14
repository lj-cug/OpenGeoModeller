# Obtaining the DUNE core modules
for module in common geometry grid localfunctions istl; do
  git clone -b releases/2.9 https://github.com/dune-project/dune-${module}.git
# git clone -b releases/2.9 https://gitlab.dune-project.org/core/dune-${module}.git
done

# Obtaining the Dumux source code
# Dumux code is located in the same path with dune-*
git clone -b releases/3.6 https://git.iws.uni-stuttgart.de/dumux-repositories/dumux.git

# Configure and build
#./dune-common/bin/dunecontrol --opts=dumux/cmake.opts all
# or
./dune-common/bin/dunecontrol --opts=dumux/cmake.opts --builddir=$(pwd)/build all

# or
# echo 'export PATH=/path/to/dune-common/bin:${PATH}' >> ${HOME}/.bashrc
# source ${HOME}/.bashrc

# Compiler options
# -DCMAKE_BUILD_TYPE=Release in cmake.opts in dumux/cmake.opts
#dunecontrol make -j$(nproc)


# Install external dependencies via script
#python dumux/bin/installexternal.py --help
python dumux/bin/installexternal.py alugrid

# clean the CMake cache
./dune-common/bin/dunecontrol bexec rm -r CMakeFiles CMakeCache.txt

# Reconfigure and build DuMux with the dunecontrol script
./dune-common/bin/dunecontrol --opts=./dumux/cmake.opts all