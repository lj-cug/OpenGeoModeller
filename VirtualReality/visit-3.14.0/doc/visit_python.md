# Python script for visit
## Starting VisIt°Øs Python Interface
visit -cli

```
import sys
sys.path.append("/path/to/visit/<version>/<architecture>/lib/site-packages")
```

env PYTHONPATH=/path/to/visit/<version>/<architecture>/lib/site-packages ./myscript.py

```
from visit import *
Launch()
```

```
import visit
visit.AddArgument("-v")
visit.AddArgument("<version>") # for example: "3.2.0"
visit.Launch()
import visit
```

## Python3 vs Python2
visit -nowin -cli -py2to3 -s hello_visit.py

œ‘ æ£∫
```
Running: cli -dv -nowin -py2to3 -s hello_visit.py
VisIt CLI: Automatic Python 2to3 Conversion Enabled
Running: viewer -dv -nowin -noint -host 127.0.0.1 -port 5600
Hello from VisIt
```

## Getting started
```
OpenDatabase("/usr/local/visit/data/multi_curv3d.silo")
AddPlot("Pseudocolor", "u")
DrawPlots()
```

Example:
```
OpenDatabase("/usr/local/visit/data/globe.silo")
AddPlot("Pseudocolor", "u")
AddOperator("Slice")
p = PseudocolorAttributes()
p.colorTableName = "rainbow"
p.opacity = 0.5
SetPlotOptions(p)
a = SliceAttributes()
a.originType = a.Point
a.normal, a.upAxis = (1,1,1), (-1,1,-1)
SetOperatorOptions(a)
DrawPlots()
```

