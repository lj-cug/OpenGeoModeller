# Windows ϵͳ�°�װ

�� https://julialang.org/downloads/ ���� Windows Julia ��װ����

# Linux/FreeBSD ��װ

wget https://mirrors.tuna.tsinghua.edu.cn/julia-releases/bin/linux/x86/1.7/julia-1.7.2-linux-i686.tar.gz --no-check-certificate

��ѹ��

tar zxvf julia-1.7.2-linux-i686.tar.gz

��ѹ��ɺ� julia �Ľ�ѹĿ¼�ƶ��� /usr/local Ŀ¼�£�

mv julia-1.7.2 /usr/local/

�ƶ���ɺ����ǾͿ���ʹ�� julia ������Ŀ¼ִ�� Julia ���

# /usr/local/julia-1.7.2/bin/julia -v   
julia version 1.7.2

julia -v �������ڲ鿴�汾�š�

julia ʹ������·�����ÿ�ִ���ļ���/usr/local/julia-1.7.2/bin/julia -v

Ҳ���Խ� julia �������ӵ�����ϵͳ PATH ���������У��༭ ~/.bashrc���� ~/.bash_profile���ļ��������һ���������´��룺

export PATH="$PATH:/usr/local/julia-1.7.2/bin/"
���Ӻ�ִ����������û�������������Ч��

source ~/.bashrc 
    
��

source ~/.bash_profile�������ǾͿ���ֱ��ִ�� julia ���������Ҫ��������·����

julia -v

julia version 1.7.2
