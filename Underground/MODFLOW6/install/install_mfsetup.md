# To build modflow-setup as following

(1)  conda create -n mfsetup

(2)  conda install -c conda-forge mamba

(3)  mamba env create -f requirements.yml

(4)  mamba env update -f requirements.yml

(5) or reinstall mfsetup as :

       mamba env remove -n mfsetup
	   
       mamba env create -f requirements.yml
            
(6) conda activate mfsetup

(7) git clone https://github.com/aleaf/modflow-setup.git

    cd modflow-setup
	
    pip install -e .
    
(8) Run case error depended on the versions of 3rd party Python library, especially numpy and sfrmaker

I build and run modflow-setup-develop succefully using numpy-1.24 and sfrmaker-0.10.1, pandas==1.5.3

(9) When running to the step "m.write_input()", we met the TypeError: 'ModflowNam' object is not subscriptable

In Line 838 of ./mfsetup/mf6model.py

        files += [p[0].filename for k, p in self.simulation.package_key_dict.items()]

we should modify it to:

        files += [p.filename for k, p in self.simulation.package_key_dict.items()]

(10) install sfrmaker et al. libraries

## 安装mf-setup

Windows OS:

手动安装GDAL等依赖库，参考：https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal

Linux OS:

wget http://download.osgeo.org/gdal/3.3.3/gdal-3.3.3.tar.gz

tar -xvzf gdal-3.3.3.tar.gz

cd gdal-3.3.3

./configure --with-python='/usr/bin/python3.7'

make -j8

make install

ldconfig 

gdalinfo --version 

cd swig/python/

python setup.py build

python setup.py install




