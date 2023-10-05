# Build Original
cd $HOME/dafoam/OpenFOAM && \
wget https://sourceforge.net/projects/openfoam/files/v1812/OpenFOAM-v1812.tgz/download -O OpenFOAM-v1812.tgz && \
wget https://sourceforge.net/projects/openfoam/files/v1812/ThirdParty-v1812.tgz/download -O ThirdParty-v1812.tgz && \
tar -xvf OpenFOAM-v1812.tgz && \
tar -xvf ThirdParty-v1812.tgz && \
cd $HOME/dafoam/OpenFOAM/OpenFOAM-v1812 && \
sed -i 's/$HOME/$DAFOAM_ROOT_PATH/g' etc/bashrc && \
wget https://github.com/DAFoam/files/releases/download/v1.0.0/UPstream.C && \
mv UPstream.C src/Pstream/mpi/UPstream.C && \
echo '# OpenFOAM-v1812' >> $HOME/dafoam/loadDAFoam.sh && \
echo 'source $DAFOAM_ROOT_PATH/OpenFOAM/OpenFOAM-v1812/etc/bashrc' >> $HOME/dafoam/loadDAFoam.sh && \
echo 'export LD_LIBRARY_PATH=$DAFOAM_ROOT_PATH/OpenFOAM/sharedLibs:$LD_LIBRARY_PATH' >> $HOME/dafoam/loadDAFoam.sh && \
echo 'export PATH=$DAFOAM_ROOT_PATH/OpenFOAM/sharedBins:$PATH' >> $HOME/dafoam/loadDAFoam.sh && \
. $HOME/dafoam/loadDAFoam.sh && \
export WM_NCOMPPROCS=4 && \
./Allwmake

## check installation
simpleFoam -help

# Build Reverse Mode AD
cd $HOME/dafoam/OpenFOAM && \
wget https://github.com/DAFoam/OpenFOAM-v1812-AD/archive/v1.3.0.tar.gz -O OpenFOAM-v1812-AD.tgz && \
tar -xvf OpenFOAM-v1812-AD.tgz && mv OpenFOAM-v1812-AD-* OpenFOAM-v1812-ADR && \
cd $HOME/dafoam/OpenFOAM/OpenFOAM-v1812-ADR && \
sed -i 's/WM_PROJECT_VERSION=v1812-AD/WM_PROJECT_VERSION=v1812-ADR/g' etc/bashrc && \
sed -i 's/$HOME/$DAFOAM_ROOT_PATH/g' etc/bashrc && \
sed -i 's/export WM_CODI_AD_MODE=CODI_AD_FORWARD/export WM_CODI_AD_MODE=CODI_AD_REVERSE/g' etc/bashrc && \
. $HOME/dafoam/loadDAFoam.sh && \
source etc/bashrc && \
export WM_NCOMPPROCS=4 && \
./Allwmake 2> warningLog.txt

## check installation
DASimpleFoamReverseAD -help

## link the relative path
cd $HOME/dafoam/OpenFOAM/OpenFOAM-v1812/platforms/*/lib && \
ln -s ../../../../OpenFOAM-v1812-ADR/platforms/*/lib/*.so . && \
cd $HOME/dafoam/OpenFOAM/OpenFOAM-v1812/platforms/*/lib/dummy && \
ln -s ../../../../../OpenFOAM-v1812-ADR/platforms/*/lib/dummy/*.so . && \
cd $HOME/dafoam/OpenFOAM/OpenFOAM-v1812/platforms/*/lib/openmpi-system && \
ln -s ../../../../../OpenFOAM-v1812-ADR/platforms/*/lib/openmpi-system/*.so .


# Build Forward Mode AD
cd $HOME/dafoam/OpenFOAM && \
wget https://github.com/DAFoam/OpenFOAM-v1812-AD/archive/v1.3.0.tar.gz -O OpenFOAM-v1812-AD.tgz && \
tar -xvf OpenFOAM-v1812-AD.tgz && mv OpenFOAM-v1812-AD-* OpenFOAM-v1812-ADF && \
cd $HOME/dafoam/OpenFOAM/OpenFOAM-v1812-ADF && \
sed -i 's/WM_PROJECT_VERSION=v1812-AD/WM_PROJECT_VERSION=v1812-ADF/g' etc/bashrc && \
sed -i 's/$HOME/$DAFOAM_ROOT_PATH/g' etc/bashrc && \
. $HOME/dafoam/loadDAFoam.sh && \
source etc/bashrc && \
export WM_NCOMPPROCS=4 && \
./Allwmake 2> warningLog.txt

## link the relative path
cd $HOME/dafoam/OpenFOAM/OpenFOAM-v1812/platforms/*/lib && \
ln -s ../../../../OpenFOAM-v1812-ADF/platforms/*/lib/*.so . && \
cd $HOME/dafoam/OpenFOAM/OpenFOAM-v1812/platforms/*/lib/dummy && \
ln -s ../../../../../OpenFOAM-v1812-ADF/platforms/*/lib/dummy/*.so . && \
cd $HOME/dafoam/OpenFOAM/OpenFOAM-v1812/platforms/*/lib/openmpi-system && \
ln -s ../../../../../OpenFOAM-v1812-ADF/platforms/*/lib/openmpi-system/*.so .



unset WM_CODI_AD_MODE && \
. $HOME/dafoam/loadDAFoam.sh



