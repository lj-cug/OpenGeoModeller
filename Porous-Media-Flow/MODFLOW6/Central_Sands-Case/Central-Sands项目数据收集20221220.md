# 美国SANDS LAKES项目技术报告：数据收集和水文地质

通过该项目的数据收集项目，可作为自建数模的输入文件参考。
下文中出现的图片，可参考"Central Sands项目数据收集.pdf"

MODFLOW的建模过程可参考"Central_Sands的地下水建模.pdf"

USGS发起的研究Plainfield, Pleasant, and Long Lakes in Waushara County,
Wisconsin地下水的项目。

地下水与地表水交互建模，建立3个湖泊的地下水模型。

Hydrostratigraphy，将不同的泥沙和岩石单元分为差异化的图形域，基于格子水文地质特性建立分层。

3个湖泊周围5万口井。

hydrostratigraphic conceptualization and model layering, aquifer
properties, and interpretation of lake/groundwater interactions

## Central Sands的水文地质环境

### 沉积层厚度

图2 模型区域内沉积物厚度

厚度是由从LiDAR的DEM减去插值的基岩表面高度得到。泥沙类型间厚度差异不显著。

### 冰川地质

### 基岩地质

主要的基岩含水层是Cambrian sandstone，厚度分布极不均匀（0\~30m）

sandstone含水层下面是Crystalline基岩，可认为是不透水层。

carbonate基岩，没有出现在地下水位以下。

## 数据收集与分析

using geophysics, geologic logging, aquifer tests, groundwater-level
monitoring, a canoe-based survey of lake chemistry, temporary lakebed
piezometers, and seepage meters

使用收集的数据，创建模型分层，定义含水层特定的合理取值范围，为地下水模型提供近场水头数值。

## 井建设报告

30064个测井，描述地下水分布。图8

图9, 9982个浅水层水井，2368个基岩井，获得地下水位。

除了创建地质分层和参考水头值，还可计算水力传导度（见Aquifer
Properties讨论）。

## 地球物理探测

-   使用地震波评估到基岩的深度（在钻孔和深井稀少的区域），地震剖面主要集中在研究湖泊周围区域。

-   在重点区域，直接使用渗透仪器，测量水平和垂向水力传导度的变化。

-   使用gamma logging确认钻孔的细粒度沙区域。

-   使用穿地雷达(GPR)补充Long
    Lake的地质钻孔数据，确认浅层细粒度沙子的范围和组成。

### 地震勘探

图11(a)
地震剖面测量位置和率定点，显示了测井相对基岩的深度信息，地震数据主要集中在基岩深度很浅的区域

图11(b)研究湖泊周围的地震勘探

收集和处理了140个地震测量点数据。获得地震反演的基岩高程，以及从井、钻孔和露头获取的数据，创建表层上层基岩地形图，用于地下水模型分层创建。

### Down-hole地球物理

Gamma Log，提供岩石地层信息。

图17

### 穿地雷达(GPR)

在Long Lake西南边，图15和图16，地表以下近20m深度。

## Aquifer Properties（含水层特性）

sand和gravel含水层以及基岩含水层的水力传导和存储特性，用来评估地下水系统的水流。使用井数据评估的含水层特性，用于地下水模型，提供率定的模拟参数的合理取值。

在36个水位观测井的抽水试验，确定水力传导度。

### 含水层测试数据---文献综述

平均的水力传导度是106ft/d （slug test）

平均的水力传导度是234ft/d (pumping test)

非承压sand and gravel含水层有平均值0.17

垂向水力传导度其2.6x10^-5^ft/d，到0.56ft/d(aquifer loading)。

### 含水层特性测试---观测井

## 研究湖泊周围的水文：地下水位观测

方法：钻孔、观测井和piezometer安装

### 垂向水力梯度

## 湖泊床面水文

湖底的水力传导度

## 水文地质---概念及创建分层

Central
Sands研究区域的水文地质单元包括非固结的冰川泥沙和基岩单元。基岩分为：

1\. Paleozoic sandstones

2\. Paleozoic dolomites

3\. Precambrian crystalline bedrock

图54显示Paleozoic砂岩和dolomite

图55显示基岩高程表面

图56显示了Precambrian基岩高程表面

ArcGIS处理数据创建基岩表面

数据高程和基岩深度分布。

### 创建分层---各向异性

### 横剖面水文地质

## Hydrostratigraphy -- Aquifer Properties

使用好几种方法计算了研究区域的含水层特性（水力传导度）。

## 总结

将观测和分析数据考虑到地下水模型中有助于深入理解地下水运行。
