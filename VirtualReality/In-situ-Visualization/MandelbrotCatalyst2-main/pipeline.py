from paraview.simple import *
from paraview import catalyst
import time

# registrationName must match the channel name used in the
# 'CatalystAdaptor'.
producer = TrivialProducer(registrationName="grid")

# Load VTKmFilters plugin
LoadDistributedPlugin('VTKmFilters', remote=False, ns=globals())

contour = VTKmContour(registrationName='Isosurfacer', Input=producer)
contour.ContourBy = ['POINTS', 'iterations']
contour.ComputeScalars = True

# Create a new 'Render View'
gridView = CreateView('RenderView')
gridView.ViewSize = [3840, 2160]
gridView.CameraPosition = [-4.537188976548553, -0.012836243874583613, -1.154262528034093]
gridView.CameraFocalPoint = [3.6332860273371077, 0.053689001918238735, 2.5651828592548216]
gridView.CameraViewUp = [-0.41431456372072795, -0.0006772349835720472, 0.9101334977033205]
gridView.CameraParallelScale = 1.9202858588575382
gridView.UseColorPaletteForBackground = 0
gridView.BackgroundColorMode = 'Gradient'
gridView.Background2 = [0.0, 0.0, 0.16470588235294117]
gridView.Background = [0.32941176470588235, 0.34901960784313724, 0.42745098039215684]

# get color transfer function/color map for 'iterations'
iterationsLUT = GetColorTransferFunction('iterations')
iterationsLUT.ApplyPreset('Cool to Warm', True)
# get opacity transfer function/opacity map for 'iterations'
iterationsPWF = GetOpacityTransferFunction('iterations')

# show data from grid
gridDisplay = Show(producer, gridView, 'UniformGridRepresentation')
# set scalar coloring using an separate color/opacity maps
ColorBy(gridDisplay, ('POINTS', 'iterations'), True)
# change representation type
gridDisplay.SetRepresentationType('Volume')
gridDisplay.LookupTable = iterationsLUT
gridDisplay.OpacityTransferFunction = iterationsPWF
gridDisplay.ScalarOpacityUnitDistance = 0.0250022332321958
# get color legend/bar for iterationsLUT in view gridView
iterationsLUTColorBar = GetScalarBar(iterationsLUT, gridView)
iterationsLUTColorBar.Title = 'iterations'
iterationsLUTColorBar.ComponentTitle = ''
# set color bar visibility
iterationsLUTColorBar.Visibility = 1
iterationsLUTColorBar.AutoOrient = 0
iterationsLUTColorBar.WindowLocation = 'Upper Right Corner'
# show color legend
gridDisplay.SetScalarBarVisibility(gridView, True)

# show isosurface
contourDisplay = Show(contour, gridView, 'GeometryRepresentation')
contourDisplay.Representation = 'Surface'
contourDisplay.SelectNormalArray = 'None'

# ----------------------------------------------------------------
# setup extractors
# ----------------------------------------------------------------
# create extractor
gridViewExtractor = CreateExtractor('JPG', gridView, registrationName='gridViewExtractor')
# trace defaults for the extractor.
gridViewExtractor.Trigger = 'TimeStep'

# init the 'PNG' selected for 'Writer'
gridViewExtractor.Writer.FileName = 'screenshot_{timestep:06d}.png'
gridViewExtractor.Writer.ImageResolution = [3840, 2160]
gridViewExtractor.Writer.Format = 'PNG'

# ------------------------------------------------------------------------------
# Catalyst options
options = catalyst.Options()
if "--enable-live" in catalyst.get_args():
  options.EnableCatalystLive = 1

def catalyst_execute(info):
    print("-----------------------------------")
    print("executing (cycle={}, time={})".format(info.cycle, info.time))
    global producer
    contour.Isosurfaces = [info.time]
    contour.UpdatePipeline()
    # rescale color and/or opacity maps used to exactly fit the current data range
    gridDisplay.RescaleTransferFunctionToDataRange(True, True)

    # In a real simulation sleep is not needed. We use it here to slow down the
    # "simulation" and make sure ParaView client can catch up with the produced
    # results instead of having all of them flashing at once.
    if options.EnableCatalystLive:
        time.sleep(2)