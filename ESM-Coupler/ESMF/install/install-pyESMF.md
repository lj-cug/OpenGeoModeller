# install pyESMF
## download esmf
git clone https://githubfast.com/esmf-org/esmf.git

## install ESMF

## install esmpy
```
cd /home/ESMF/src/addon/esmpy # ��esmpy·����
#conda activate env           # ������Ҫ��װesmpy�Ļ���
export ESMFMKFILE=/home/ESMF/lib/libO/Linux.intel.64.intelmpi.default/esmf.mk
make 
make install
#��ʾSuccessfully installed
```
### Check

pip list 
 
����esmpy����װ�ɹ�