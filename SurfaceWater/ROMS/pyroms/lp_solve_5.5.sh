##Download lpsolve source code lp_solve_5.5.2.0_source.tar.gz
##Extract the archive and copy extra/Python folder into lp_solve_5.5
cd lp_solve_5.5/lpsolve55
sh ccc (on linux)
### $ sh ccc.osx (on Mac)

cd lp_solve_5.5/extra/Python/

## change lpsolve55 path in extra/Pythpn/setup.py to point to appropriate directory.
## Note: In my case, I used linux 64 bit machine so folder 'bin/ux64/' created under lpsolve55 directory when executed "sh ccc" command from terminal. The folder contains the lpsolve library files. The LPSOLVE55 path in setup.py should point to the newly generated directory which contains the required lpsolve libraries(liblpsolve55.a). 
export LPSOLVE55='../../lpsolve55/bin/ux64'  #change path to reflect appropriate path.
 
## Use following command to install lpsolve extension into site-packages.
python setup.py install
