# virtualenvָ����ͬ�汾��Python����

һ��virtualenv Python�汾

Virtualenv��һ�����ڴ���Python���⻷���Ĺ��ߣ������Խ�����PythonӦ�ó�����뿪����
ʹ��ÿ��Ӧ�ó������������Python������virtualenv�����û��ڲ�ͬ��Python�汾֮������л����Ӷ����߰�װ����ֻ��������Python�汾�°�װ����Ŀ⼴�ɡ�

# ��װ���⻷��
$ pip install virtualenv

# �л���ĳ��Python�汾��
$ virtualenv --python=/usr/bin/python3.6 env
������������У�����ָ����Python 3.6�汾��Ϊ���⻷���е�Ĭ��Python�汾��


����Python venv virtualenv
Python venvģ����Python 3.3������ģ����ṩ����virtualenv���ƵĹ��ܡ�
��Python 3.3֮ǰ���û�����ʹ�õ������������������⻷����

# �������⻷��
$ python3 -m venv env

# �л���ĳ��Python�汾��
$ source env/bin/activate

������������У�����ʹ����Python venvģ�鴴����һ����Ϊenv�����⻷����������û�����


����Python��װvirtualenv
��װvirtualenv�����ף�ֻ��ʹ��pip��װ���ɡ�

# ��װvirtualenv
$ pip install virtualenv
��ĳЩ����£��������Ҫʹ�ù���ԱȨ�ޡ�

# ʹ�ù���ԱȨ�ް�װvirtualenv
$ sudo pip install virtualenv


�ġ�Python���⻷��virtualenv
���⻷������������ͬһ̨�����ϰ�װ����汾��Python��ÿ���汾�����������Python������
�����ʹ��pip��ÿ�����⻷���а�װ��ͬ��Python���Ӧ�ó��򣬶�����Ӱ�쵽�������⻷����

������������У�����ʹ��virtualenv������һ����Ϊenv��Python 3.6���⻷����

# ����Python 3.6���⻷��
$ virtualenv --python=/usr/bin/python3.6 env
Ҫ�������⻷�����������������

# �������⻷��
$ source env/bin/activate
�����⻷���а�װPython��ǳ��򵥣�ֻ�������⻷�����������pip����ɡ�

# �����⻷���а�װPython��
$ pip install requests


�塢Python��virtualenvѡȡ
��ѡ��virtualenvʱ���м�����Ҫ��������Ҫ���ǡ�

���ȣ�����Ҫȷ��ѡ���virtualenv�汾�������ʹ�õ�Python�汾���ݡ������ʹ�õ���Python 3.6����Ӧѡ��֧��Python 3.6��virtualenv�汾��

��Σ�����Ҫ����Ҫʹ�õ����⻷�������������ֻ��һ��PythonӦ�ó��������ֻ��Ҫһ�����⻷����������ж��PythonӦ�ó����������Ҫ������⻷����

�������Ҫ����Ҫ�������⻷����Ƶ�ʡ�����㾭���л����⻷�����������Ҫʹ��Python venvģ�飬��Ϊ��������������ɵؼ�����˳����⻷����

��֮��virtualenv��һ���ǳ����õĹ��ߣ������԰���������Ա���벻ͬ��PythonӦ�ó�����ߴ���Ŀɿ��ԺͿ���ֲ�ԡ�