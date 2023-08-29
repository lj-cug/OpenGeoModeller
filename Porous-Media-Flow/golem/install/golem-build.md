# 构建Golem步骤

## 1. Setting Up a MOOSE Installation

## 2. 克隆GOLEM

    cd ~/projects
    git clone https://github.com/ajacquey/Golem.git
    cd ~/projects/golem
    git checkout master
	
## 3. 编译GOLEM	
	
    cd ~/projects/golem
    make -j4
	
## 4. 测试GOLEM	

    cd ~/projects/golem
    ./run_tests -j2

# 运行Golem

    mpiexec -n <nprocs> ~/projects/golem/golem-opt -i <input-file>

Where `<nprocs>` is the number of processors you want to use and `<input-file>` is the path to your input file (extension `.i`).  
	