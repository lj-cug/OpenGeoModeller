# Install docker

sudo apt-get remove docker docker-engine docker.io containerd runc && sudo apt-get update && sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent   software-properties-common -y && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - && sudo add-apt-repository "deb [arch=amd64] https:// download. docker.com/linux/ubuntu $(lsb_release -cs) stable" && sudo apt-get update && sudo apt-get install docker docker.io -y

sudo usermod -aG docker $USER

docker --version

docker pull dafoam/opt-packages:v3.0.7

## 查看容器

docker ps -a

## 删除容器

docker rm -f <CONTAINER_ID or CONTAINER_NAME>

## download the workshop examples

git clone https://github.com/dafoam/workshops

cd $HOME/workshops/2022_Summer/examples/openmdao

## run this command to start a Docker container

docker run -it --rm -u dafoamuser --mount "type=bind,src=$(pwd),target=/home/dafoamuser/mount" -w /home/dafoamuser/mount dafoam/opt-packages:v3.0.7 bash

### Windows docker container

docker run -it --rm -u dafoamuser --mount "type=bind,src=%cd%,target=/home/dafoamuser/mount" -w /home/dafoamuser/mount dafoam/opt-packages:v3.0.7 bash

在Docker容器内，你会看到：

dafoamuser@cddb89839078:~/mount$

运行：   python runScript.py

# 直接测试openmdao

pip install openmdao==3.16.0

cd $HOME/workshops/2022_Summer/examples/openmdao

python runScript.py

# Run NACA0012 subsonic case

cd workshops/2022_Summer/examples/naca0012

## generate the mesh

./preProcessing.sh

## run the optimization with 2 cores

mpirun -np 2 python runScript.py 2>&1 | tee logOpt.txt

The optimization log will be printed to the screen and saved to
logOpt.txt. In addition, the optimizer will write a separate log to the
disk opt_IPOPT.txt.

## How to post-process the optimization results ?

## The N2 diagram for the NACA0012 aerodynamic optimization.

## 学习runScript.py脚本

## Summary

The runScript.py is essentially an OpenMDAO run script. So we
suggest you first learn how OpenMDAO works by going through
the OpenMDAO’s documentation (https:
//openmdao.org/newdocs/versions/latest/main.html).

We can use the above script to run any airfoil aerodynamic
optimization with DAFoam v3. If you want to change the flight
conditions, FFD points, airfoil profiles, refer to the DAFoam FAQ.
https://dafoam.github.io/mydoc_get_started_faq.html.
Note that these changes are for v2 but they also work for v3.

For 3D wing aerodynamic optimization, refer to the run script
https://github.com/DAFoam/tutorials/blob/main/MACH_
Tutorial_Wing/runScript_Aero.py

For multipoint optimization, refer to the run script
https://github.com/DAFoam/tutorials/blob/main/
NACA0012_Airfoil/multipoint/runScript.py
