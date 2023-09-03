To build modflow-setup as following:

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
(×¢Òâ°æ±¾ºÅ)

(9) When running to the step "m.write_input()", we met the TypeError: 'ModflowNam' object is not subscriptable

In Line 838 of ./mfsetup/mf6model.py
        files += [p[0].filename for k, p in self.simulation.package_key_dict.items()]

we should modify it to:
        files += [p.filename for k, p in self.simulation.package_key_dict.items()]

(10) install sfrmaker et al. libraries
1 Install memba
conda install mamba -n base -c conda-forge

2 install SFRMaker
mamba env create -f requirements.yml

or update SFRMaker:
conda env update -f requirements.yml

reinstall:
conda env create -f requirements.yml