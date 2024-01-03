# 安装amgcl

git clone https://github.com/ddemidov/amgcl

cd ./amgcl

cmake -Bbuild -DAMGCL_BUILD_TESTS=ON -DAMGCL_BUILD_EXAMPLES=ON .

实际上，不需要编译安装amgcl, opm-simulator的CMAKE设置时，设置好amgcl_dir的路径到amgcl的代码路径即可
