# baidumap与ggmap的区别
getBaiduMap(location, width = 400, height = 400, zoom = 10, scale = 2, color = "color", messaging = TRUE)

```
theMap <- baidumap::getBaiduMap(location, zoom = zoom, color = "color", messaging = FALSE) 		
theMap <- ggmap::get_map(location, zoom = zoom, source = source, maptype=maptype)
```

```	
ggmap::gglocator(n=1)
baidumap::getLocation
```
	    
# 搜索'google'关键词的rwrfhydro R脚本
```
visualize_channtwk.R:14:#'   this is the argument passed to the \pkg{ggmap} for its argument of the same name.
visualize_channtwk.R:17:#'   \item{\code{zoom=11}}{The zoom level for the google (or other) map. See \pkg{ggmap} for more details.}
visualize_channtwk.R:19:#'   \item{\code{source='google'}}{The source for the underlying map. See \pkg{ggmap} package for details.}
visualize_channtwk.R:21:#'   \item{\code{maptype='terrain'}}{The map type for \pkg{ggmap}.}
visualize_channtwk.R:188:    theMap <- ggmap::get_map(location, zoom = zoom, source = source, maptype=maptype)
visualize_channtwk.R:193:      ggmap::ggmap(theMap, extent='normal') +
visualize_channtwk.R:219:      clickPt <- ggmap::gglocator(n=1)

visualize_domain.R:24:##' theMap <- ggmap::get_map(location=c(lon=mean(hydroCoords$long)-.5,
visualize_domain.R:26:##' ggmap::ggmap(theMap) + geom_point(data=hydroCoords, aes(x=lonFile, y=latFile),size=1)
visualize_domain.R:27:##' ggmap::ggmap(theMap) + geom_point(data=hydroCoords, aes(x=long, y=lat),size=.5)
visualize_domain.R:141:##' (arguments are passed to ggmap and ggplot inside the function). This function (the closure) returns
visualize_domain.R:155:##'  GgMapFunction <- VisualizeDomain(hydroFile, "CHANNELGRID")
visualize_domain.R:156:##'  ggMap1 <- GgMapFunction(zoom=11, pointshape=15, pointsize=7,
visualize_domain.R:162:##'  ggMap2 <- GgMapFunction(location=c(lon=orodellLonLat$lon[1], lat=orodellLonLat$lat[1]),
visualize_domain.R:165:##' ggMap2$ggMapObj +
visualize_domain.R:243:  # The closure returns the ggMapObject - might be ways to merge or build these.
visualize_domain.R:260:      ## the package. It seems that ggmap:: should take care of this, but it dosent.
visualize_domain.R:277:      ggMapObj <- ggmap::get_map(locationEval, zoom = zoom, source = source, maptype=maptype)
visualize_domain.R:289:      ggObj <- ggmap::ggmap(ggMapObj) + ggPlotObj +
visualize_domain.R:297:                     list(plotDf=plotDf, ggMapObj=ggMapObj, ggPlotObj=ggPlotObj,

visualize_routelink.R:143:      theMap <- ggmap::get_map(location, zoom = zoom, source = source, maptype=maptype)
visualize_routelink.R:146:        ggmap::ggmap(theMap, extent='normal') +

visualize_spatial.R:9:##' (arguments are passed to ggmap and ggplot inside the function). This function (the closure) returns
visualize_spatial.R:43:  # The closure returns the ggMapObject - might be ways to merge or build these.
visualize_spatial.R:60:      ## the package. It seems that ggmap:: should take care of this, but it dosent.
visualize_spatial.R:77:      ggMapObj <- ggmap::get_map(locationEval, zoom = zoom, source = source, maptype=maptype)
visualize_spatial.R:90:        print(ggObj <- ggmap::ggmap(ggMapObj) + ggPlotObj +
visualize_spatial.R:96:                     list(plotDf=plotDf, ggMapObj=ggMapObj, ggPlotObj=ggPlotObj,
```

