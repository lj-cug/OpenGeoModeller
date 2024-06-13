# 安装spyro的步骤
## spyro源码下载
[spyro的github下载链接](https://github.com/krober10nd/spyro)

## spyro的GMD论文的zenodo
The Zenodo release for the code is available at https://doi.org/10.5281/zenodo.5164113 (Roberts et al., 2021a), 

with data and simulation scripts for FWI at https://doi.org/10.5281/zenodo.5172307 (Roberts, 2021). 

This implementation was based on Firedrake version 20210810.0 at
https://doi.org/10.5281/zenodo.5176201 (firedrake-zenodo, 2021)

## firedrake-spyro步骤
基于Zenodo的源码安装
```
python3 firedrake-install --doi 10.5281/zenodo.5176201
source firedrake/bin/activate
wget -C https://github.com/NDF-Poli-USP/spyro/archive/refs/tags/V0.1.0.tar.gz
tar xvf V0.1.0
pip install -e  <path-to-spyro-repository>
```