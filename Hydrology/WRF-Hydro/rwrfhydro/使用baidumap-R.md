# baidumap R 
https://www.jianshu.com/p/aa697cfcea7d

```
library(devtools)
install_github('badbye/baidumap')
```

## baidumap-API
Ϊ�ٶȵ�ͼAPI�ṩR�ӿڣ���ggmapһ�������Ӱٶ�api�����ǹȸ��openstreet��ȡ��ͼ��

��Կ:��Ҫ�� lbsyun.baidu.com������Կ��Ȼ����Rע�����Կ�ס�
```
library(baidumap)
options(baidumap.key = '34aVpSzoQLi1SjZLU9PJYznqPtVGAqmR')   # �ҵ�Win10�ʼǱ���IP
```

```
getLocation(location,output = "json")  # ���������ݻ�ȡλ��
����ouput�� ���÷�����������(��json��, ��xml��)
```

```
lon = matrix(c(117.93780, 24.55730, 117.93291, 24.57745, 117.23530, 24.64210, 117.05890, 24.74860), byrow=T, ncol=2)
location_json = getLocation(lon[1,], output='json') ## json 
location = getLocation(lon[1, ], formatted = T) ## formatted
```
```
�ӵ�ַ��ȡ��������
getCoordinate(address, city = NULL,output = "json",formatted = F)     #����������ʼ��Ϣ
����ouput�����÷�����������(��json��, ��xml��)

getCoordinate('������ѧ', output='xml')  # xml
getCoordinate('������ѧ', formatted = T) # character
getCoordinate(c('������ѧ', '�廪��ѧ'), formatted = T) # matrix

��ȡ�ٶȵ�ͼͼ��
getBaiduMap(location, width = 400, height = 400, zoom = 10, scale = 2,
  color = "color", messaging = TRUE)

����
location�����������λ���ַ���
width, height, zoom��ͼ���С
scale���������ĳ˷�����
color ��color or ��bw��,(color or black-and-white)
messaging����ʾ��Ϣ
getBaiduMap('�й�',zoom = 4) %>% ggmap
```