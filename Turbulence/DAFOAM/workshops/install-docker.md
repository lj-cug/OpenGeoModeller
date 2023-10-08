# Windows 下安装使用docker

[安装使用链接](https://blog.csdn.net/weixin_51351637/article/details/128006765)

# Ubuntu  下安装使用docker

sudo apt-get remove docker docker-engine docker.io containerd runc && sudo apt-get update && sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent   software-properties-common -y && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - && sudo add-apt-repository "deb [arch=amd64] https:// download. docker.com/linux/ubuntu $(lsb_release -cs) stable" && sudo apt-get update && sudo apt-get install docker docker.io -y

sudo usermod -aG docker $USER

## 查看容器

docker ps -a

## 删除容器

docker rm -f <CONTAINER_ID or CONTAINER_NAME>

## 查看安装的镜像

docker images

## 运行和停止容器

docker start/stop  <CONTAINER_ID or CONTAINER_NAME>
