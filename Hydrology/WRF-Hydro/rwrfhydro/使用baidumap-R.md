# baidumap R 
https://www.jianshu.com/p/aa697cfcea7d

```
library(devtools)
install_github('badbye/baidumap')
```

## baidumap-API
为百度地图API提供R接口，和ggmap一样，但从百度api而不是谷歌或openstreet获取地图。

密钥:需要从 lbsyun.baidu.com申请密钥。然后在R注册你的钥匙。
```
library(baidumap)
options(baidumap.key = '34aVpSzoQLi1SjZLU9PJYznqPtVGAqmR')   # 我的Win10笔记本的IP
```

```
getLocation(location,output = "json")  # 从坐标数据获取位置
参数ouput： 设置返回数据类型(‘json’, ‘xml’)
```

```
lon = matrix(c(117.93780, 24.55730, 117.93291, 24.57745, 117.23530, 24.64210, 117.05890, 24.74860), byrow=T, ncol=2)
location_json = getLocation(lon[1,], output='json') ## json 
location = getLocation(lon[1, ], formatted = T) ## formatted
```
```
从地址获取坐标数据
getCoordinate(address, city = NULL,output = "json",formatted = F)     #返回坐标或初始信息
参数ouput：设置返回数据类型(‘json’, ‘xml’)

getCoordinate('北京大学', output='xml')  # xml
getCoordinate('北京大学', formatted = T) # character
getCoordinate(c('北京大学', '清华大学'), formatted = T) # matrix

获取百度地图图块
getBaiduMap(location, width = 400, height = 400, zoom = 10, scale = 2,
  color = "color", messaging = TRUE)

参数
location：中心坐标或位置字符，
width, height, zoom：图块大小
scale：像素数的乘法因子
color ：color or “bw”,(color or black-and-white)
messaging：提示信息
getBaiduMap('中国',zoom = 4) %>% ggmap
```