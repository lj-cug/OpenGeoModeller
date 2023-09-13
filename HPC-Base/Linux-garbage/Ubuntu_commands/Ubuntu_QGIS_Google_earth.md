# install google earth pro

sudo apt install gdebi-core
wget https://dl.google.com/dl/earth/client/current/google-earth-pro-stable_current_amd64.deb
sudo gdebi google-earth-pro-stable_current_amd64.deb
google-earth-pro


# qgis安装方法

指令安装

    sudo add-apt-repository ppa:ubuntugis/ubuntugis-unstable
    sudo apt-get update
    sudo apt-get install qgis
    
卸载：

    apt-get remove qgis
    apt-get purge qgis 
  