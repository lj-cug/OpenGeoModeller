# Ubuntu��ʹ��npm��װnodejs
ʹ��npm��ԭ��ֱ��ȥ�������أ�һ���ٶ�����������������ʧ�ܣ�

��npm �� Node.js �İ������ߣ�����������װ���� Node.js ����չ�����Ա���ܶలװ���⡣

## npm��Ҫ����
1.��װ npm

sudo apt install npm

2. ��װnģ��(nģ����������װnodejs��һ������ģ��)

npm install n -g

3.ѡ����֧�ְ�

sudo n lts

4.���汾

node -v

## Դ�밲װ��

���ڹ������ص�Դ��
https://nodejs.org/dist/v12.18.0/node-v12.18.0.tar.gz
���Ƚ�ѹ

./configure
make
make install

# node.js�����İ�װ�ű�
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