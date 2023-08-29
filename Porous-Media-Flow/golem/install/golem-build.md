# ����Golem����

## 1. Setting Up a MOOSE Installation

## 2. ��¡GOLEM

    cd ~/projects
    git clone https://github.com/ajacquey/Golem.git
    cd ~/projects/golem
    git checkout master
	
## 3. ����GOLEM	
	
    cd ~/projects/golem
    make -j4
	
## 4. ����GOLEM	

    cd ~/projects/golem
    ./run_tests -j2

# ����Golem

    mpiexec -n <nprocs> ~/projects/golem/golem-opt -i <input-file>

Where `<nprocs>` is the number of processors you want to use and `<input-file>` is the path to your input file (extension `.i`).  
	