# ���ù���

1.����ʹ��fdiks -l �����ѯ����Ҫ���ص�Ӳ���̷����� /dec/sdb

2.ʹ��blkid /dev/sdb �����ѯ��Ҫ���ص�Ӳ�̵�UUID

3.��UUIDд�������ļ� /etc/fstab����ʽ����

UUID=afab653d-7620-49df-ba66-f956c372ef93 /home/mkky/data4 ext4 defaults 0 0

��һ���� UUID? �ڶ����ǹ��ص�Ŀ¼ ���������ļ�ϵͳ �������ǲ��� ������0��ʾ������ �����б���Ϊ0����2 ������Ϊ1

4.ִ�� mount -a ����

5. df -h�鿴Ӳ���Ƿ���ȷ���ء�

# ����ϼ�
```
fdisk -l
blkid /dev/sdb
echo UUID=afab653d-7620-49df-ba66-f956c372ef93 /home/mkky/data4 ext4 defaults 0 0 >> /etc/fstab
mount -a
df -h
```
