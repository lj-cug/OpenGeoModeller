# pyGIMLi

pyGIMLi��һ���Ч��ʹ��Python���Ա�д�ĵ��������ݹ���

����������ݣ�����ΪFWI�ĳ�ʼ�ٶ�ģ��

[githubԴ��ֿ�����](https://github.com/gimli-org/gimli)

[��װ](https://www.pygimli.org/installation.html#sec-install)

[Tutorials-How to use pyGIMLi](https://www.pygimli.org/_tutorials_auto/index.html)

[Examples](https://www.pygimli.org/_examples_auto/index.html)
���У��е���������ݵ�����

## ��װ

conda create -n pg -c gimli -c conda-forge pygimli=1.4.3

conda activate pg

���ߣ�
�� https://anaconda.org/gimli/pygimli/filesֱ������ĳ�汾��tar.gz2�ļ���Ȼ��ִ�У�
 
   conda install package.tar.gz

���ߣ���װ���°汾

git clone https://github.com/gimli-org/gimli

cd gimli

make pygimli J=2

���û���������

export PYTHONPATH=$PYTHONPATH:$HOME/src/gimli/gimli/python

export PATH=$PATH:$HOME/src/gimli/build/lib

export PATH=$PATH:$HOME/src/gimli/build/bin

## ����

python -c "import pygimli; pygimli.test(show=False, onlydoctests=True)"