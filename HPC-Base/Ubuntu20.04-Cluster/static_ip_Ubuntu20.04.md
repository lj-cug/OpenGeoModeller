# Ubuntu系统下设置静态Ip

https://blog.csdn.net/oNelson123/article/details/125417115

(1) Step 1

gedit /etc/netplan/01-network-manager-all.yaml

(2) Step2: edit

## Let NetworkManager manage all devices on this system
network:
  ethernets:
    enp3s0:   # 配置的网卡的名称
      addresses: [192.168.1.80/24]   # 配置的静态ip地址和掩码
      dhcp4: false                     # 关闭dhcp4
      optional: true
      gateway4: 192.168.1.1           # 网关地址
      nameservers:
        addresses: [114.114.114.114]
  version: 2
  renderer: NetworkManager

(3) Step 3

netplan apply

至此Ubuntu20.04的静态IP配置完成。经过测试，可以正常上网。
