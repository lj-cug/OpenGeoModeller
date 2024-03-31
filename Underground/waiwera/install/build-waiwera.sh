# Install Ansible etc. for build waiware
apt install -y ansible
pip install ansible      # in miniconda-base environment

apt install ninja-build  # backend of meson
pip install meson        # in miniconda-base
apt install meson        # install in Ubuntu-20.04

# Download the Waiwera source code
git clone https://github.com/waiwera/waiwera.git
# or wget -c -t 10 https://github.com/waiwera/waiwera/archive/master.zip

# Installation scripts path
cd waiwera/install

# The following command builds Waiwera (and dependencies), but does not install it.
ansible-playbook ansible/build.yml
# Installing Waiwera on your system (不推荐)
ninja -C build install
ninja -C build uninstall


# Note that Waiwera currently requires PETSc version 3.15.2 or newer.
# 直接运行ansible-playbook，则waiwera使用自己的PETSc
ansible-playbook ansible/install_local.yml -e "base_dir=/home/lijian/HPC_Build/waiwera-install/"
# 或者
# waiwera也可以连接外部的PETSc
export PKG_CONFIG_PATH=/home/lijian/HPC_Build/petsc-3.15.5-install/lib/pkgconfig:$PKG_CONFIG_PATH
pkg-config --libs PETSc
pkg-config --modversion PETSc
ansible-playbook ansible/install_local.yml -e "base_dir=/home/lijian/HPC_Build/waiwera-install/"
# where base_dir is the desired Waiwera installation directory


# install_local.yml中设置其他参数
• petsc_update=true will build a new version of PETSc even if an installed version is detected
– defaults to false meaning PETSc will only be built if an installed version isn’t detected

• waiwera_update=true will build Waiwera every time even a new version isn’t pulled by git
– defaults to false

• zofu_build=true
– defaults to false and uses meson to build zofu

• fson_build=true
– defaults to false and uses meson to build zofu

• ninja_build=true
– defaults to false and only builds locally if no ninja install is detected


# Linking to other libraries: FSON and Zofu (optional)
#
# To ensure Waiwera can be run from any directory, the Waiwera installation directory should be on the user’s PATH. 
export PATH=/home/lijian/HPC_Build/waiwera-install/bin:$PATH

# Running the unit tests
# The unit tests (which test individual routines in the Waiwera code) are created using the Zofu framework for Fortran unit testing, and run using Meson.
python unit_tests.py
