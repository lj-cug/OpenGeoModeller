# 安装pymake

[pymake的github链接](https://github.com/modflowpy/pymake)

## Installation

To install pymake using pip type:

pip install mfpymake

To install pymake directly from the git repository type:

pip install https://github.com/modflowpy/pymake/zipball/master

To update your version of pymake with the latest from the git repository type:

pip install https://github.com/modflowpy/pymake/zipball/master --upgrade


## 解决Python3 控制台输出InsecureRequestWarning的问题

问题：

使用Python3 requests发送HTTPS请求，已经关闭认证（verify=False）情况下，控制台会输出以下错误：

InsecureRequestWarning: Unverified HTTPS request is being made. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings

解决方法：

在代码中添加以下代码即可解决：

 import urllib3
 urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

Python2添加如下代码即可解决：

1 from requests.packages.urllib3.exceptions import InsecureRequestWarning
2 # 禁用安全请求警告
3 requests.packages.urllib3.disable_warnings(InsecureRequestWarning)