# you will need at least version 3.20.0 for GNU Compiler Collection
(cd $(mktemp -d) && wget -qO- https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0-Linux-x86_64.tar.gz | tar -xvz -C "." && mv "./cmake-3.20.0-linux-x86_64" "${HOME}/bin/cmake")

# use version 3.16.2 for Intel Compiler Collection
# (cd $(mktemp -d) && wget -qO- https://github.com/Kitware/CMake/releases/download/v3.16.2/cmake-3.16.2-Linux-x86_64.tar.gz | tar -xvz -C "." && mv "./cmake-3.16.2-Linux-x86_64" "${HOME}/bin/cmake")

ln -s ${HOME}/bin/cmake/bin/cmake ${HOME}/bin
