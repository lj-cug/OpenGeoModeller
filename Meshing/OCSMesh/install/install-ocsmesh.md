# 在conda环境中安装OCSMesh

wget https://raw.githubusercontent.com/noaa-ocs-modeling/OCSMesh/main/environment.yml

conda env create -f environment.yml -n your-env-name

conda activate your-env-name

conda install -y -c conda-forge jigsawpy

pip install ocsmesh

# 从github的源码安装OCSMesh

git clone https://github.com/noaa-ocs-modeling/ocsmesh
cd ocsmesh
python ./setup.py install_jigsaw   # To install latest Jigsaw from GitHub
python ./setup.py install          # Installs the OCSMesh library to the current Python environment
# OR
python ./setup.py develop          # Run this if you are a developer.


## Requirements

* 3.9 <= Python
* CMake 
* C/C++ compilers

## How to Cite

Title : OCSMesh: a data-driven automated unstructured mesh generation software for coastal ocean modeling

Personal Author(s) : Mani, Soroosh;Calzada, Jaime R.;Moghimi, Saeed;Zhang, Y. Joseph;Myers, Edward;Pe’eri, Shachak;

Corporate Authors(s) : Coast Survey Development Laboratory (U.S.)

Published Date : 2021

Series : NOAA Technical Memorandum NOS CS ; 47

DOI : https://doi.org/10.25923/csba-m072
