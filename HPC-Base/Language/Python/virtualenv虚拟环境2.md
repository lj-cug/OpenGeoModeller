# virtualenv指定不同版本的Python环境

一、virtualenv Python版本

Virtualenv是一种用于创建Python虚拟环境的工具，它可以将各种Python应用程序隔离开来，
使得每个应用程序都有其独立的Python环境。virtualenv允许用户在不同的Python版本之间进行切换，从而离线安装包，只需在所需Python版本下安装所需的库即可。

# 安装虚拟环境
$ pip install virtualenv

# 切换到某个Python版本下
$ virtualenv --python=/usr/bin/python3.6 env
在上面的例子中，我们指定了Python 3.6版本作为虚拟环境中的默认Python版本。


二、Python venv virtualenv
Python venv模块是Python 3.3中引入的，它提供了与virtualenv类似的功能。
在Python 3.3之前，用户必须使用第三方工具来创建虚拟环境。

# 创建虚拟环境
$ python3 -m venv env

# 切换到某个Python版本下
$ source env/bin/activate

在上面的例子中，我们使用了Python venv模块创建了一个名为env的虚拟环境，并进入该环境。


三、Python安装virtualenv
安装virtualenv很容易，只需使用pip安装即可。

# 安装virtualenv
$ pip install virtualenv
在某些情况下，你可能需要使用管理员权限。

# 使用管理员权限安装virtualenv
$ sudo pip install virtualenv


四、Python虚拟环境virtualenv
虚拟环境可以让你在同一台机器上安装多个版本的Python，每个版本都有其独立的Python环境。
你可以使用pip在每个虚拟环境中安装不同的Python库和应用程序，而不会影响到其他虚拟环境。

在下面的例子中，我们使用virtualenv创建了一个名为env的Python 3.6虚拟环境。

# 创建Python 3.6虚拟环境
$ virtualenv --python=/usr/bin/python3.6 env
要激活虚拟环境，请运行以下命令。

# 激活虚拟环境
$ source env/bin/activate
在虚拟环境中安装Python库非常简单，只需在虚拟环境激活后运行pip命令即可。

# 在虚拟环境中安装Python库
$ pip install requests


五、Python的virtualenv选取
在选择virtualenv时，有几个重要的因素需要考虑。

首先，你需要确保选择的virtualenv版本与你打算使用的Python版本兼容。如果你使用的是Python 3.6，则应选择支持Python 3.6的virtualenv版本。

其次，你需要考虑要使用的虚拟环境数量。如果你只有一个Python应用程序，则可能只需要一个虚拟环境。如果你有多个Python应用程序，则可能需要多个虚拟环境。

最后，你需要考虑要激活虚拟环境的频率。如果你经常切换虚拟环境，则可能需要使用Python venv模块，因为它可以让你更轻松地激活和退出虚拟环境。

总之，virtualenv是一个非常有用的工具，它可以帮助开发人员隔离不同的Python应用程序，提高代码的可靠性和可移植性。