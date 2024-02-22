# Same Python interpreter for all time steps
# We use count for one time initializations
try:
    count = count + 1
except NameError:
    count = 0

if count == 0:
    # ParaView API
    # WARNING: this does not work inside the plugin
    #          unless you have the same import in paraview-vis.py
    import paraview
    paraview.options.batch = True
    paraview.options.symmetric = True
    from paraview.simple import LoadPlugin, Show, ColorBy, \
        GetColorTransferFunction, GetActiveView, GetScalarBar, ResetCamera,\
        Render, SaveScreenshot, GetActiveCamera, GetProperty, CreateRenderView
    from paraview import catalyst
    #fname="/apps/daint/UES/Ascent/ascent-install/examples/ascent/paraview-vis/paraview_ascent_source.py"
    fname="/local/apps/Ascent/ascent-install/examples/ascent/paraview-vis/paraview_ascent_source.py"
    LoadPlugin(fname, remote=True, ns=globals())
    ascentSource = AscentSource()
    view = CreateRenderView()
    view.ViewSize = [512, 512]

    gridDisplay = Show() # show the current object as Bounding Box
    gridDisplay.Representation = 'Outline'
    gridDisplay.ColorArrayName = ['POINTS', '']

#   using the current object as Input, create a new 'Threshold' for Pressure
    threshold = Threshold(registrationName='Threshold', Input=ascentSource)
    threshold.Scalars = ['CELLS', 'p']
    threshold.LowerThreshold = 300.0
    threshold.UpperThreshold = 3000.0
    
    varname = "p"
    pLUT = GetColorTransferFunction(varname)
    pLUT.RGBPoints = [329.06873339062196, 0.231373, 0.298039, 0.752941, 6346.370878182501, 0.865003, 0.865003, 0.865003, 12363.67302297438, 0.705882, 0.0156863, 0.14902]
    pLUT.ScalarRangeInitialized = 1.0
    threshold2Display = Show(threshold)
    threshold2Display.Representation = 'Surface'
    threshold2Display.ColorArrayName = ['CELLS', 'p']
    threshold2Display.LookupTable = pLUT


    lutSB = GetScalarBar(pLUT, view)
    lutSB.Title = varname
    lutSB.ComponentTitle = ''
    # set color bar visibility
    lutSB.Visibility = 1
    # show color legend
    threshold2Display.SetScalarBarVisibility(view, True)

    cam = GetActiveCamera()
    cam.Elevation(30)
    cam.Azimuth(-120)

ascentSource.UpdateAscentData()
ascentSource.UpdatePropertyInformation()
cycle = GetProperty(ascentSource, "Cycle").GetElement(0)

imageName = "Pressure_{0:04d}.png".format(int(cycle))
threshold2Display.RescaleTransferFunctionToDataRange(False, True)
ResetCamera()

SaveScreenshot(imageName, ImageResolution=(512, 512))
