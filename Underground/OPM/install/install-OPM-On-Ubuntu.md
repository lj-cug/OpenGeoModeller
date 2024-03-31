# Install OPM on Ubuntu Linux 20.04 or 22.04 (64 bit version only)

![OPM_Norne](./media/OPM-Flow.png)

## install the apt-add-repository command

apt-get update

apt-get install software-properties-common

## Then we add the repository, and run update again

apt-add-repository ppa:opm/ppa

apt-get update

##  To see a list of (for example) the opm-simulators packages

apt-cache search opm-simulators

## Then, to install the opm-simulators programs (including Flow) and their dependencies,Then, to install the opm-simulators programs (including Flow) and their dependencies

apt-get install mpi-default-bin

apt-get install libopm-simulators-bin
