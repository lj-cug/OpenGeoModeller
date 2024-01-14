# GLM-R-Tools
  
## R工具简介
```
GLMr - GLM_v2.2.0rc的R语言绑定 (GLEON研发)
GLM3r - GLM_v3.x的R语言绑定 (GLEON研发, 推荐使用!)
rLakeAnalyzer - GLM模型计算结果的后处理和评估 (GLEON, Winslow et al.,2016)
GRAPLEr  - 基于R的分布式计算工具, 管理大量的GLM运行
glmtools - USGS研发的与GLM (AED)交互,R脚本工具, 包括: 计算物理量导数、计算结果的热量属性、绘图的一些函数
glmgui - Jsta研发的基于rLakeAnalyzer, glmtools与GLMr的GUI界面化工具 (需要用R-3.6.x ???), 用于GLM2的前后处理
```

## glmtools安装
```
install.packages('remotes')
remotes::install_github('usgs-r/glmtools')
```