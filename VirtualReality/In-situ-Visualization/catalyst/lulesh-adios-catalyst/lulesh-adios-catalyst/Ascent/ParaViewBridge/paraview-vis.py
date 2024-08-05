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
    view.ViewSize = [1024, 1024]

    rep = Show()
    varname = "velocity"
    ColorBy(rep, ("POINTS", varname))
    lut = GetColorTransferFunction(varname)
    lut.RescaleTransferFunction(0,1e2) # if showing variable 'vtkProcessId'

    lutSB = GetScalarBar(lut, view)
    lutSB.Title = varname
    lutSB.ComponentTitle = ''
    # set color bar visibility
    lutSB.Visibility = 1
    # show color legend
    rep.SetScalarBarVisibility(view, True)

    cam = GetActiveCamera()
    cam.Elevation(-30)
    cam.Azimuth(-120)


ascentSource.UpdateAscentData()
ascentSource.UpdatePropertyInformation()
cycle = GetProperty(ascentSource, "Cycle").GetElement(0)
rank = GetProperty(ascentSource, "Rank").GetElement(0)
if rank == 0:
  for ai in ascentSource.PointData.values():
    print(cycle, ai.GetName(), ai.GetRange(0))
imageName = "MYimage_{0:04d}.png".format(int(cycle))
rep.RescaleTransferFunctionToDataRange(False, True)
ResetCamera()

SaveScreenshot(imageName, ImageResolution=(1024, 1024))

# This does not work correctly if
# topologies/topo/elements/origin/{i0,j0,k0} (optional, default = {0,0,0})
# is missing
#dataName = "lulesh_data_{0:04d}".format(int(cycle))
#writer = CreateWriter(dataName + ".pvts", ascentSource)
#writer.UpdatePipeline()


# # VTK API
# from ascent_to_vtk import AscentSource, write_vtk
# ascentSource = AscentSource()
# ascentSource.Update()
# write_vtk("vtkdata", ascentSource.GetNode(),
#           ascentSource.GetOutputDataObject(0))


# # Python API
#from ascent_to_vtk import ascent_to_vtk, write_vtk, write_json
#node = ascent_data().child(0)
#write_json("blueprint", node)
#data = ascent_to_vtk(node)
#write_vtk("pythondata", node, data)
