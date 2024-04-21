# 基于ESMF框架的耦合模式
ESMF耦合结构网格与结构网格：  
```
WRF-ROMS
WRF-MITgcm 
RegCM-MITgcm-WWMIII (RegESM）
WRF-MITgcm-WWMIII (SCRPPS) 
```

ESMF耦合结构网格(大气)与非结构网格(海洋)： 
```
WRF-FVCOM (no code)
RegCM3-FVCOM (Wei, 2018, no code)
```

ESMF耦合非结构网格(大气)与结构网格(海洋):  
ICON-GETM (ICON代码下载需要free license邮寄)

ESMF耦合非结构网格与非结构网格:  
ADCIRC-WWM

可参考代码：
```
fvcom/nuopc
schism-esmf
atmesh
```

## 研究计划

将WRF-MITgcm-WWM(SCRPPS_v2.0)的WRF-ESMF，植入 RegESM
升级RegESM系统中RegCM, Cop (Catalyst)的版本

WRF(v3.8.1, v4.1)/RegCM(v4) + SCHISM_ESMF (PDAF) + WWM + ADCIRC + MITgcm/ROMS

