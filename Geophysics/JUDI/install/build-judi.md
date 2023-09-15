# Install JUDI

## install Julia

wget -c https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.2-linux-x86_64.tar.gz

tar zxvf julia-1.9.2-linux-x86_64.tar.gz

gedit ~/.bashrc

export PATH=$PATH:/home/lijian/HPC_Build/Devito/julia-1.9.2/bin

source ~/.bashrc

## run julia

julia


## install JUDI

# remove error info. for libcurl

using Libdl

filter!(contains("curl"), dllist())

pkg> add JUDI

or from the command line

julia -e 'using Pkg;Pkg.add("JUDI")'