# Python的虚拟环境设置

[原文链接](https://blog.csdn.net/Damien_J_Scott/article/details/131582357)

virtualenv是Python的一个工具，用于创建独立的Python环境。它允许你在同一台计算机上同时管理多个独立的Python环境，每个环境都可以有自己的包依赖和Python版本。

在开发Python应用程序时，常常会遇到不同项目需要使用不同的包版本或Python版本的情况。使用virtualenv可以创建隔离的Python环境，使得每个项目可以拥有自己的独立环境，并在这些环境中安装和管理所需的包。

通过使用virtualenv，你可以避免不同项目之间的依赖冲突，确保项目的环境是独立的和可重现的。这样一来，即使在同一台计算机上同时进行多个项目的开发，也能够保持它们之间的隔离性。

以下是使用virtualenv的一些常见操作：
 
## 创建虚拟环境

在你希望创建虚拟环境的位置，使用以下命令创建一个新的虚拟环境(记得提前执行pip install virtualenv)：

$ virtualenv myenv

或者

python -m venv myenv
 

## 激活虚拟环境

进入你的项目目录，并激活虚拟环境。在终端中执行以下命令：

$ source myenv/bin/activate   #(在 Linux 或 macOS 上)

$ myenv\Scripts\activate   #(在 Windows 上)

安装包：

激活虚拟环境后，在项目目录中使用pip命令安装项目所需的依赖包：

 如果报错没有pip，那就安装pip

(myenv) $ python -m ensurepip
(myenv) $ pip install package_name

查看已安装的包：

(myenv) $ pip list

## 退出虚拟环境

(myenv) $ deactivate

通过创建和使用虚拟环境，你可以更好地管理Python项目和依赖，确保项目之间的隔离性和灵活性。这在开发过程中非常有用，尤其是当你需要同时处理多个项目或者需要在不同的环境中运行相同的代码时。
