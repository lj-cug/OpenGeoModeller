
stream_points=[(607899, 4.21483e+006),(610173, 4.21632e+006),(611124, 4.21716e+006),(612121, 4.2181e+006)]

cross_points=[(607706, 4.21508e+006),(608204, 4.21443e+006),(610080, 4.2163e+006),
(610499, 4.21584e+006),(611180, 4.2173e+006),(612190, 4.2163e+006)]
		
		
for i in range(len(stream_points)-1):
   AddPlot("Pseudocolor", "salinity")
   p = PseudocolorAttributes()
   p.minFlag = 1
   p.min= 0.0
   p.maxFlag = 1
   p.max= 1.0
   p.colorTableName = "hot"
   p.legendFlag=0
   SetPlotOptions(p)
   AddOperator("Transform")
   tr = TransformAttributes()
   tr.doScale = 1
   tr.scaleZ  = 100
   SetOperatorOptions(tr)
   x1 = stream_points[i][0]
   y1 = stream_points[i][1]
   x2 = stream_points[i+1][0]
   y2 = stream_points[i+1][1]
   normal0 = y1-y2
   normal1 = x2-x1
   AddOperator("Slice")
   s = SliceAttributes()
   s.originType = s.Point
   s.project2d = 0
   s.originPoint=(x1,y1,0)
   s.normal = (normal0,normal1,0)
   SetOperatorOptions(s)
   AddOperator("Clip")
   c = ClipAttributes()
   c.plane1Origin=(x1,y1,0)
   c.plane1Normal = (1,0,0)
   c.plane2Origin=(x2,y2,0)
   c.plane2Normal = (-1,0,0)
   c.planeInverse = 1
   c.plane2Status = 1
   SetOperatorOptions(c)
   DrawPlots()

for i in range(len(cross_points)/2):
   AddPlot("Pseudocolor", "salinity")
   p = PseudocolorAttributes()
   p.minFlag = 1
   p.min= 0.0
   p.maxFlag = 1
   p.max= 1.0
   p.colorTableName = "hot"
   p.legendFlag=0
   SetPlotOptions(p)
   AddOperator("Threshold")
   ts=ThresholdAttributes()
   ts.listedVarNames="salinity"
   ts.lowerBounds=0.0
   ts.zonePortions=1
   AddOperator("Transform")
   tr = TransformAttributes()
   tr.doScale = 1
   tr.scaleZ  = 100
   SetOperatorOptions(tr)
   x1 = cross_points[2*i][0]
   y1 = cross_points[2*i][1]
   x2 = cross_points[2*i+1][0]
   y2 = cross_points[2*i+1][1]
   AddOperator("Slice")
   s = SliceAttributes()
   s.originType = s.Point
   s.project2d = 0
   s.originPoint=(x1,y1,0)
   s.normal = (y1-y2,x2-x1,0)
   SetOperatorOptions(s)
   AddOperator("Clip")
   c = ClipAttributes()
   c.plane1Origin=(x2,y2,0)
   c.plane1Normal = (x2-x1,y2-y1,0)
   c.planeInverse = 0
   SetOperatorOptions(c)
   DrawPlots()

v = View3DAttributes()
v.viewNormal = (-0.0825594,-0.762568,0.641618)
v.focus = (576656,4.21771e+06,1338.23)
v.viewUp = (0.0287784,0.641723,0.766396)
v.viewAngle = 30
v.parallelScale = 109730
v.nearPlane = -219460
v.farPlane = 219460
v.imagePan=(-0.141668, 0.00512045)
v.imageZoom=47.566
v.shear=(0,0,1)
v.perspective = 1
SetView3D(v) # Set the 3D view
DrawPlots()

    


