# AIpycker-install

cd AIPycker-master

python setup.py install

发生错误：

ImportError: cannot import name 'NavigationToolbar2TkAgg' from 'matplotlib.backends.backend_tkagg' (C:\Users\jianl\.conda\envs\segy\lib\site-packages\matplotlib\backends\backend_tkagg.py)

解决参考：
https://stackoverflow.com/questions/32188180/from-matplotlib-backends-import-tkagg-importerror-cannot-import-name-tkagg


修改gui/gui.py代码： 

替代NavigationToolbar2TkAgg 为 NavigationToolbar2Tk

第14行

第225行
