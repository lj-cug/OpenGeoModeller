
git�ͻ������زο��� git-for-Windows.md


git�ͻ��˰�װ�ο���https://blog.csdn.net/fzx1597965407/article/details/124371720


ʲô��git
Git��һ����Դ�ķֲ�ʽ�汾����ϵͳ��������Ч�����ٵش���Ӻ�С���ǳ������Ŀ�汾����Ҳ��Linus TorvaldsΪ�˰�������Linux�ں˿�����������һ������Դ��İ汾���������

Git �ǻ��� Linux�ں˿����İ汾���ƹ��ߡ��볣�õİ汾���ƹ��� CVS, Subversion �Ȳ�ͬ���������˷ֲ�ʽ�汾��ķ�ʽ�����ط����������֧�֣�wingeddevilע����÷�����ʲô���ķ���ˣ�ʹ��httpЭ�����gitЭ��Ȳ�̫һ����������push��pull��ʱ��ͷ������˻����н����ġ�����ʹԴ����ķ����ͽ������䷽�㡣 Git ���ٶȺܿ죬��������� Linux kernel �����Ĵ���Ŀ��˵��Ȼ����Ҫ�� Git ��Ϊ��ɫ�������ĺϲ����٣�merge tracing��������

git�������������У�1����һ�γ�ʼ����2����������������3����ʼ���ֿ⣻4���鿴�ֿ⵱ǰ״̬��5���ļ���ز�����6���鿴��ʷ��¼��7������ع���8���汾����ز�����9��Զ�ֿ̲���ز�����10����֧��ز�����11��git������ã�12�������鿴������أ�13������ĳ���ύ��14����ǩ��


1����һ�γ�ʼ��
git init
git add .
git commit -m ��first commit��
git remote add origin git@github.com:�ʺ���/�ֿ���.git
git pull origin master
git push origin master # -f ǿ��
git clone git@github.com:git�ʺ���/�ֿ���.git


2��������������
git checkout master �е�����֧
git fetch origin ��ȡ���±��
git checkout -b dev origin/master ��������֧����dev��֧
git add . ��ӵ�����
git commit -m ��xxx�� �ύ�����زֿ�
git fetch origin ��ȡ���±��


3����ʼ���ֿ�
git init

4���鿴�ֿ⵱ǰ״̬
git status

5���ļ���ز���
���ļ���ӵ��ֿ⣺

git add �ļ��� ����������ĳ���ļ���ӵ��ݴ���
git add . ����ǰ�������������ļ��������ݴ���
git add -u ������б�tracked�ļ��б��޸Ļ�ɾ�����ļ���Ϣ���ݴ�����������untracked���ļ�
git add -A ������б�tracked�ļ��б��޸Ļ�ɾ�����ļ���Ϣ���ݴ���������untracked���ļ�
git add -i ���뽻������ģʽ����������ļ���������
���ݴ����ļ��ύ�����زֿ⣺

git commit -m ���ύ˵���� ���ݴ��������ύ�����زֿ�
git commit -a -m ���ύ˵���� ����������������ֱ�Ӱѹ����������ύ�����زֿ�
�Ƚ��ļ���ͬ

git diff ���������ݴ����Ĳ���
git diff ��֧�� ��������ĳ��֧�Ĳ��죬Զ�̷�֧����д��remotes/origin/��֧��
git diff HEAD ��������HEADָ��ָ������ݲ���
git diff �ύid �ļ�·�� ������ĳ�ļ���ǰ�汾����ʷ�汾�Ĳ���
git diff �Cstage �������ļ����ϴ��ύ�Ĳ���(1.6 �汾ǰ�� �Ccached)
git diff �汾TAG �鿴��ĳ���汾�󶼸Ķ�����
git diff ��֧A ��֧B �Ƚϴӷ�֧A�ͷ�֧B�Ĳ���(Ҳ֧�ֱȽ�����TAG)
git diff ��֧A����֧B �Ƚ�����֧�ڷֿ�����ԵĸĶ�
���⣺���ֻ��ͳ����Щ�ļ����Ķ��������б��Ķ���������� �Cstat ����

6���鿴��ʷ��¼
git log �鿴����commit��¼(SHA-AУ��ͣ��������ƣ����䣬�ύʱ�䣬�ύ˵��)
git log -p -���� �鿴������ٴε��ύ��¼
git log �Cstat ������ʾÿ���ύ�����ݸ���
git log �Cname-only ����ʾ���޸ĵ��ļ��嵥
git log �Cname-status ��ʾ�������޸ģ�ɾ�����ļ��嵥
git log �Coneline ���ύ��¼�Ծ����һ�����
git log �Cgraph �Call �Conline ͼ��չʾ��֧�ĺϲ���ʷ
git log �Cauthor=���� ��ѯ���ߵ��ύ��¼(��grepͬʱʹ��Ҫ��һ���Call�Cmatch����)
git log �Cgrep=������Ϣ �г��ύ��Ϣ�а���������Ϣ���ύ��¼
git log -S��ѯ���� �ͨCgrep���ƣ�S�Ͳ�ѯ���ݼ�û�пո�
git log fileName �鿴ĳ�ļ����޸ļ�¼

7������ع�
git reset HEAD^ �ָ����ϴ��ύ�İ汾
git reset HEAD^^ �ָ������ϴ��ύ�İ汾�����Ƕ��^���Դ����ƻ���~����
git reflog
git reset �Chard �汾��
�Csoft��ֻ�Ǹı�HEADָ��ָ�򣬻������͹��������䣻
�Cmixed���޸�HEADָ��ָ���ݴ������ݶ�ʧ�����������䣻
�Chard���޸�HEADָ��ָ���ݴ������ݶ�ʧ���������ָ���ǰ״̬��

8���汾����ز���
ɾ���汾���ļ���git rm �ļ���
�汾����İ汾�滻�������İ汾��git checkout �� test.txt


9��Զ�ֿ̲���ز���
ͬ��Զ�ֿ̲⣺git push -u origin master

���زֿ��������͵�Զ�ֿ̲⣺git remote add origin git@github.com:�ʺ���/�ֿ���.git

��Զ�ֿ̲��¡��Ŀ�����أ�git clone git@github.com:git�ʺ���/�ֿ���.git

�鿴Զ�̿���Ϣ��git remote

��ȡԶ�̷�֧�����زֿ⣺

git checkout -b ���ط�֧ Զ�̷�֧ # ���ڱ����½���֧�����Զ��л����÷�֧
git fetch origin Զ�̷�֧:���ط�֧ # ���ڱ����½���֧���������Զ��л�������checkout
git branch �Cset-upstream ���ط�֧ Զ�̷�֧ # �������ط�֧��Զ�̷�֧������
ͬ��Զ�ֿ̲���£���git fetch origin master

10����֧��ز���
������֧��git checkout -b dev  -b��ʾ�������л���֧
����һ�������൱��һ��Ķ�����
git branch dev  ������֧
git checkout dev  �л���֧

�鿴��֧��git branch

�ϲ���֧��

git merge dev #���ںϲ�ָ����֧����ǰ��֧
git merge �Cno-ff -m ��merge with no-ff�� dev #���ϨCno-ff�����Ϳ�������ͨģʽ�ϲ����ϲ������ʷ�з�֧���ܿ��������������ϲ�
ɾ����֧��git branch -d dev

�鿴��֧�ϲ�ͼ��git log �Cgraph �Cpretty=oneline �Cabbrev-commit

11��git�������
��װ��Git���һ��Ҫ�����£������û���Ϣ(global�ɻ���local�ڵ�����Ŀ��Ч)��

git config �Cglobal user.name ���û����� # �����û���
git config �Cglobal user.email ���û����䡱 #��������
git config �Cglobal user.name # �鿴�û����Ƿ����óɹ�
git config �Cglobal user.email # �鿴�����Ƿ�����


12�������鿴�������
git config �Cglobal �Clist # �鿴ȫ��������ز����б�
git config �Clocal �Clist # �鿴����������ز����б�
git config �Csystem �Clist # �鿴ϵͳ���ò����б�
git config �Clist # �鿴����Git������(ȫ��+����+ϵͳ)
git config �Cglobal color.ui true //��ʾgit�����ɫ


13������ĳ���ύ
git revert HEAD # ���������һ���ύ
git revert �汾�� # ����ĳ��commit


14����ǩ
git tag ��ǩ //���ǩ���Ĭ��ΪHEAD
git tag //��ʾ���б�ǩ
git tag ��ǩ �汾�� //��ĳ��commit�汾��ӱ�ǩ
git show ��ǩ //��ʾĳ����ǩ����ϸ��Ϣ
