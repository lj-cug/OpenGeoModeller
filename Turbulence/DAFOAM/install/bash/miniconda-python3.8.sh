# install miniconda
#cd $HOME/dafoam/packages && \
#wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh && \
#chmod 755 Miniconda3-py38_4.10.3-Linux-x86_64.sh && \
#./Miniconda3-py38_4.10.3-Linux-x86_64.sh -b -p $HOME/dafoam/packages/miniconda3 && \

# Use the installed Miniconda3
echo '# Miniconda3' >> $HOME/dafoam/loadDAFoam.sh && \
echo 'export PATH=$DAFOAM_ROOT_PATH/packages/miniconda3/bin:$PATH' >> $HOME/dafoam/loadDAFoam.sh && \
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DAFOAM_ROOT_PATH/packages/miniconda3/lib' >> $HOME/dafoam/loadDAFoam.sh && \
echo 'export PYTHONUSERBASE=no-local-libs' >> $HOME/dafoam/loadDAFoam.sh && \
. $HOME/dafoam/loadDAFoam.sh

pip install --upgrade pip && \
pip install numpy==1.21.2 && \
pip install scipy==1.7.1 && \
pip install cython==0.29.21 && \
pip install numpy-stl==2.16.0 && \
pip install pynastran==1.3.3 && \
pip install nptyping==1.4.4 && \
pip install swig==4.1.1 && \
pip install tensorflow-cpu==2.12
