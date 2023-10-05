mkdir -p $HOME/dafoam && \
mkdir -p $HOME/dafoam/packages $HOME/dafoam/OpenFOAM $HOME/dafoam/OpenFOAM/sharedBins $HOME/dafoam/OpenFOAM/sharedLibs $HOME/dafoam/repos && \
echo '#!/bin/bash' > $HOME/dafoam/loadDAFoam.sh && \
echo '# DAFoam root path' >> $HOME/dafoam/loadDAFoam.sh && \
echo 'export DAFOAM_ROOT_PATH=$HOME/dafoam' >> $HOME/dafoam/loadDAFoam.sh && \
chmod 755 $HOME/dafoam/loadDAFoam.sh && \
. $HOME/dafoam/loadDAFoam.sh

