# Install Ansible etc. for build waiware
apt install -y ansible
pip install ansible      # in miniconda-base environment

apt install ninja-build  # backend of meson
pip install meson        # in miniconda-base
apt install meson        # install in Ubuntu-20.04

# Download the Waiwera source code
git clone https://github.com/waiwera/waiwera.git
# or wget -c -t 10 https://github.com/waiwera/waiwera/archive/master.zip

# Build Waiwera
cd waiwera/install

# Note that Waiwera currently requires PETSc version 3.15.2 or newer.
# waiwera使用自己的PETSc
ansible-playbook ansible/install_local.yml -e "base_dir=/home/lijian/HPC_Build/waiwera-install/"


# waiwera链接外部的PETSc
export PKG_CONFIG_PATH=/home/lijian/HPC_Build/petsc-3.15.5-install/lib/pkgconfig:$PKG_CONFIG_PATH
pkg-config --libs PETSc
pkg-config --modversion PETSc
ansible-playbook ansible/install_local.yml -e "base_dir=/home/lijian/HPC_Build/waiwera-install/"

# Linking to other libraries: FSON and Zofu


# To ensure Waiwera can be run from any directory, the Waiwera installation directory should be on the user’s PATH. 
export PATH=/home/lijian/HPC_Build/waiwera-install/bin:$PATH


# Running the unit tests
# The unit tests (which test individual routines in the Waiwera code) are created using the Zofu framework for Fortran unit testing, and run using Meson.
python unit_tests.py


# Installing Waiwera on your system
ninja -C build install
ninja -C build uninstall


