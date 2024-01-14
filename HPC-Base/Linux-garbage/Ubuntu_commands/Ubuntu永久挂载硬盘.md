# 永久挂载

1.首先使用fdiks -l 命令查询到你要挂载的硬盘盘符例如 /dec/sdb

2.使用blkid /dev/sdb 命令查询你要挂载的硬盘的UUID

3.将UUID写入配置文件 /etc/fstab，格式如下

UUID=afab653d-7620-49df-ba66-f956c372ef93 /home/mkky/data4 ext4 defaults 0 0

第一列是 UUID? 第二列是挂载的目录 第三列是文件系统 第四列是参数 第五列0表示不备份 第六列必须为0或者2 引导区为1

4.执行 mount -a 命令

5. df -h查看硬盘是否正确挂载。

# 命令合集
```
fdisk -l
blkid /dev/sdb
echo UUID=afab653d-7620-49df-ba66-f956c372ef93 /home/mkky/data4 ext4 defaults 0 0 >> /etc/fstab
mount -a
df -h
```
