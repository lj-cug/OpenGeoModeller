# Rwrfhydro介绍
```
An R Package for working with WRF-Hydro 
A community-contributed toolbox for managing, analyzing, and visualizing WRF-Hydro input and output files in R. 
Rwrfhydro is currently minimally supported and rarely updated. 
https://github.com/NCAR/rwrfhydro/releases
```

# Rwrfhydro安装
## github安装
```
library(ncdf4)  ## check if the install was successful.
devtools::install_github('mccreigh/rwrfhydro')  ## one time and for updates.
library(rwrfhydro)  ## for each new R session where you want to use rwrfhydro.s
?rwrfhydro   ## 查询
```

## 本地安装
install.packages("D:/Atmosphere-Ocean/WRF-Hydro/Rwrfhydro/rwrfhydro-1.0.1.tar", repos = NULL, type="source")

## help
```
help(package='rwrfhydro')
ls('package:rwrfhydro')  # To simply print the available (exported) functions:
help.search("DART",package='rwrfhydro')
help.search(package='rwrfhydro', keyword='hplot')

GetPkgMeta(listMetaOnly = TRUE)
pkgMeta <- GetPkgMeta()
str(pkgMeta)
GetPkgMeta(listMetaOnly = TRUE, meta='concept')  #To suppress printing of the functions and see only the concepts
GetPkgMeta(concept=c('plot', 'dataGet'), keyword=c('hplot', 'manip')) # Or perhaps you just want to look at certain concepts and keywords
```