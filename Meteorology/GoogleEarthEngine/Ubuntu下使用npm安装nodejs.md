# Ubuntu下使用npm安装nodejs
使用npm的原因，直接去官网下载，一是速度慢，二是容易下载失败，

而npm 是 Node.js 的包管理工具，可以用来安装各种 Node.js 的扩展，可以避免很多安装问题。

## npm主要步骤
1.安装 npm

sudo apt install npm

2. 安装n模块(n模块是用来安装nodejs的一个工具模块)

npm install n -g

3.选择长期支持版

sudo n lts

4.检查版本

node -v

## 源码安装。

对于官网下载的源码
https://nodejs.org/dist/v12.18.0/node-v12.18.0.tar.gz
首先解压

./configure
make
make install

# node.js官网的安装脚本
```
# installs fnm (Fast Node Manager)
curl -fsSL https://fnm.vercel.app/install | bash

# download and install Node.js
fnm use --install-if-missing 20

# verifies the right Node.js version is in the environment
node -v   # should print `v20.15.0`

# verifies the right NPM version is in the environment
npm -v    # should print `10.7.0`
```