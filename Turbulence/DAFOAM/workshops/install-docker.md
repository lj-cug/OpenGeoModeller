# Windows �°�װʹ��docker

[��װʹ������](https://blog.csdn.net/weixin_51351637/article/details/128006765)

# Ubuntu  �°�װʹ��docker

sudo apt-get remove docker docker-engine docker.io containerd runc && sudo apt-get update && sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent   software-properties-common -y && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - && sudo add-apt-repository "deb [arch=amd64] https:// download. docker.com/linux/ubuntu $(lsb_release -cs) stable" && sudo apt-get update && sudo apt-get install docker docker.io -y

sudo usermod -aG docker $USER

## �鿴����

docker ps -a

## ɾ������

docker rm -f <CONTAINER_ID or CONTAINER_NAME>

## �鿴��װ�ľ���

docker images

## ���к�ֹͣ����

docker start/stop  <CONTAINER_ID or CONTAINER_NAME>
