# julia��װ���ھ���

## ʹ�÷�ʽ

ֻ��Ҫ���û������� JULIA_PKG_SERVER �����л��������ɹ��л���������ͨ�� versioninfo() ��ѯ�������Ϣ�����磺

julia> versioninfo()

�������øû���������Ĭ��ʹ�ùٷ������� pkg.julialang.org ��Ϊ���Ρ�

## ��ʱʹ��

��ͬϵͳ�������������û��������ķ�ʽ������ͬ�����������¿���ͨ�����·�ʽ����ʱ�޸Ļ�������

### Linux Bash

export JULIA_PKG_SERVER=https://mirrors.cernet.edu.cn/julia

### Windows Powershell

$env:JULIA_PKG_SERVER = 'https://mirrors.cernet.edu.cn/julia'

Ҳ�������� JuliaCN ����ά�������ı��ػ����߰� JuliaZH �������л���

using JuliaZH              # �� using ʱ���Զ��л������ڵľ���վ

JuliaZH.set_mirror("BFSU") # Ҳ����ѡ���ֶ��л��� BFSU ����

JuliaZH.mirrors             # ��ѯ��¼��������Ϣ

## ����ʹ��

��ͬϵͳ���������������趨���������ķ�ʽҲ����ͬ������ Linux Bash �¿���ͨ���޸� ~/.bashrc �ļ�ʵ�ָ�Ŀ�ģ�

# ~/.bashrc

export JULIA_PKG_SERVER=https://mirrors.cernet.edu.cn/julia
