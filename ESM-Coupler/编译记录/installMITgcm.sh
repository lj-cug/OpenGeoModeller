# Download MITgcm and unzip
#wget https://github.com/MITgcm/MITgcm/archive/checkpoint67m.zip
unzip checkpoint67m.zip

# compile the code
cd MITgcm_c66e/verification/tutorial_barotropic_gyre/

# (1) Compile using GNU without MPI (default gfortran compiler)
cd build
../../../tools/genmake2 "-mods" "../code"
make depend
make
# Now, an executable fime: mitgcmuv in the build foler

# (2) Compile the code using gfortran and MPICH2
export MPI_HOME="/home/lijian/mpich-3.3/"
cd /home/lijian/ESM_lj/MITgcm_c66e/verification/exp2/build   
../../../tools/genmake2 "-mpi" "-mods" "../code"   #"-optfile" "/usr/include/netcdf"
make depend
make -j8
# Now, an executable fime: mitgcmuv in the build foler

# run
cd ../run
cp ../input/* ./
cp ../build/mitgcmuv ./
./mitgcmuv > output.txt    # run serial code without MPI
mpirun -np 4 mitgcmuv      

# Post-processing by MATLAB
ddpath('$MITGCM_DIR/utils/matlab/')
U=rdmds('U');
V=rdmds('V');
XG=rdmds('XG');
YG=rdmds('YG');
contourf(XG,YG,U(:,:))
contourf(XG,YG,V(:,:))
quiver(XG(1:5:end,1:5:end),YG(1:5:end,1:5:end),U(1:5:end,1:5:end),V(1:5:end,1:5:end))

# Python can also do the post-processing of the MITgcm results (need to install MITgcmutils in the MITgcm code):
cd $MITGCM_DIR/utils/python/MITgcmutils/
python setup.py install --user

# To plot the MITgcm results using python:
import MITgcmutils
import matplotlib.pyplot as plt
meshX = MITgcmutils.rdmds('$MITGCM_RESULTS_DIR/XC')
meshY = MITgcmutils.rdmds('$MITGCM_RESULTS_DIR/YC')
results = MITgcmutils.rdmds('$MITGCM_RESULTS_DIR/U')
plt.contourf(mitgcm_meshX,mitgcm_meshY,results[0,:,:])
