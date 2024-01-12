# install-RStudio-in-Ubuntu 20.04

## ��װR-base

apt update -qq

apt install --no-install-recommends software-properties-common dirmngr
 
wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
 
add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"

��װR��  apt install --no-install-recommends r-base

# ��װRStudio

R Studio��վ�Ѿ���ʽ����ΪPosit����ַ��https://posit.co/

��R Studio�����ΪR Studio��������Ͻ� DOWNLOAD RSTUDIO

ҳ����ת����������ѡ��Free�汾�����DOWNLOAD

ҳ����ת���ҵ�Ubuntu 22��Ӧ�İ汾��rstudio-2022.07.2-576-amd64.deb����������ء�

���غú����deb�ļ������ļ��У��ڵ�ǰ�ļ��д��նˣ����а�װ���

sudo dpkg -i   rstudio-2022.07.2-576-amd64.deb

�����������д��������

apt --fix-broken install

����

apt-get install -f

�ٴΰ�װ��

sudo dpkg -i   rstudio-2022.07.2-576-amd64.deb

��װ�ɹ���

## ����Ϊ����Դ

��鵱ǰ��Դ��   getOption("repos")

��һ����ѡ��Toolsѡ���е�Global Optionsѡ�

�ڶ�����ѡ��Packagesѡ���е�change��ť�����о���վ���ѡ��

Ҳ�����������´��루�ʺϲ���ͼ�ν����R����ֱ���޸ģ�

options(repos=structure(c(CRAN="https://mirrors.tuna.tsinghua.edu.cn/CRAN/")))

ͨ�� getOption("repos") �������֪��Ŀǰ�ľ�����վ�������

## �޸� bioconductor �İ�װԴ

���󲿷ֵ�������Ϣ��ص�R������DESeq2, limma, clusterProfiler������ bioconductor�������ڹٷ���Դ���棬����ͨ�� install.packages() ������Ҳ�����Ӧ��R������ʹ���������װ��
```
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("DESeq2")
```

ͬ����ʹ��option�����޸�bioconductor��ԴΪ����Դ������Ҳ��������bioconductor �Ĺ����ˣ��������£�

options(BioC_mirror="https://mirrors.tuna.tsinghua.edu.cn/bioconductor")