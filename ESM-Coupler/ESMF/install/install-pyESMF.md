# install pyESMF
## download esmf
git clone https://githubfast.com/esmf-org/esmf.git

## install ESMF

## install esmpy
```
cd /home/ESMF/src/addon/esmpy # 到esmpy路径下
#conda activate env           # 激活需要安装esmpy的环境
export ESMFMKFILE=/home/ESMF/lib/libO/Linux.intel.64.intelmpi.default/esmf.mk
make 
make install
#显示Successfully installed
```
### Check

pip list 
 
出现esmpy即安装成功