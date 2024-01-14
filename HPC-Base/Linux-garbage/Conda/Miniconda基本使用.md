# MiniConda��װ��������

Anaconda �İ�װ����ܴ��Ƽ����ʹ��MiniConda

[Anaconda ��������](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)

[MiniConda ��������](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/)


# ����MiniConda�����г��������ٶ�

Anaconda Promptʹ�����·������廪������ӵ�anaconda��ִ���������

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ 

conda config --set show_channel_urls yes

# conda��������

## ����conda�Ļ�����������������conda��

1. gedit ~/.bashrc
2. add "export PATH=$PATH:/home/root/anaconda3/bin"
3. save
4. source ~/.bashrc

## �鿴��ǰconda���л���
conda info --envs

## ��������

conda create --name envname python=version

���磺

conda create --name project_verson3.8 python=3.8

ע�������ָ��python�汾Ĭ�ϰ�װ���°�


## �������еĻ��� 

conda env export > environment.yaml

## ����yaml�ļ������µĻ��� 

conda env create -f environment.yaml

## ������ǰ��������ʹ�õİ�������requirements.txt

conda list -e > requirements.txt  #������ǰ�������е������������Ӧ�İ汾��

## ��װrequirements.txt�еİ�

conda install --yes --file requirements.txt   #���µĻ����а�װ�����İ�


## ������Ļ���

conda activate ������

## ����Ļ�������conda����pip��װ��

conda install ������

����pip install ������ -i https://pypi.tuna.tsinghua.edu.cn/simple���廪����

���磺pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple

����ͨ��conda install��pip install ����װpython�������
ִ�У�pip install -r requirements.txt      ��װ��������

## �鿴���������еİ���

conda list

pip list


## �˳���ǰ����

conda deactivate ������


## ɾ������

conda remove -n ������ --all


## �޸����⻷�����������е����⻷���Ļ���������һ���µ����⻷����

1 �½�һ����������¡Դ����
conda create --name newName --clone oldName

2 ɾ��Դ����
conda remove -n oldName --all

3�鿴����ȫ�����⻷��
conda env list
