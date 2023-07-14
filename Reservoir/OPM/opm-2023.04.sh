Version="2023.04" 

# download source code
# opm-common 
# opm-material (has no version 2023.04) 
for repo in opm-grid opm-models opm-simulators opm-upscaling
do
    echo "=== downloading and building module: $repo"
    wget https://github.com/OPM/$repo/archive/release/$Version/final.tar.gz
    tar -xzvf final.tar.gz
    rm final.tar.gz
    mv ${repo}-release-${Version}-final $repo
done

# compile source code
# opm-common 
# opm-material 
for repo in opm-grid opm-models opm-simulators opm-upscaling
do
    mkdir $repo/build
    cd $repo/build
    cmake  ..
    make -j $(nproc)
    cd ../..
done


# set different versions of opm
$ cat ~/.bashrc
...
# opm
alias flow201910=/opt/opm/flow/2019.10/opm-simulators/build/bin/flow
alias flow202004=/opt/opm/flow/2020.04/opm-simulators/build/bin/flow
alias flow202010=/opt/opm/flow/2020.10/opm-simulators/build/bin/flow
...
