# script-version: 2.0
# Catalyst state generated using paraview version 5.11.0-RC1-144-g016020f90f
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [996, 893]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [0.5625, 0.5625, 0.5625]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [-1.9816148830613265, -1.9288736221666583, -0.658850963847781]
renderView1.CameraFocalPoint = [0.5639588447423529, 0.5750634400281559, 0.5329896273684144]
renderView1.CameraViewUp = [-0.4304414447593504, 0.7055553399086555, -0.5629492205873146]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 0.9742785792574935
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(996, 893)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'PVTrivialProducer'
grid = PVTrivialProducer(registrationName='grid')

# create a new 'Slice'
slice1 = Slice(registrationName='Slice1', Input=grid)
slice1.SliceType = 'Plane'
slice1.HyperTreeGridSlicer = 'Plane'
slice1.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slice1.SliceType.Normal = [1.0, 0.0, -1.0]

# init the 'Plane' selected for 'HyperTreeGridSlicer'
slice1.HyperTreeGridSlicer.Origin = [0.5625, 0.5625, 0.5625]

# create a new 'Merge Blocks'
mergeBlocks1 = MergeBlocks(registrationName='MergeBlocks1', Input=slice1)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from grid
gridDisplay = Show(grid, renderView1, 'UnstructuredGridRepresentation')

# get 2D transfer function for 'velocity'
velocityTF2D = GetTransferFunction2D('velocity')

# get color transfer function/color map for 'velocity'
velocityLUT = GetColorTransferFunction('velocity')
velocityLUT.TransferFunction2D = velocityTF2D
velocityLUT.RGBPoints = [0.0, 0.23137254902, 0.298039215686, 0.752941176471, 5.878906683738906e-39, 0.865, 0.865, 0.865, 1.1757813367477812e-38, 0.705882352941, 0.0156862745098, 0.149019607843]
velocityLUT.ScalarRangeInitialized = 1.0
velocityLUT.RescaleTransferFunction(0.0, 34.0)

# get opacity transfer function/opacity map for 'velocity'
velocityPWF = GetOpacityTransferFunction('velocity')
velocityPWF.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]
velocityPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
gridDisplay.Representation = 'Surface'
gridDisplay.ColorArrayName = ['POINTS', 'velocity']
gridDisplay.LookupTable = velocityLUT
gridDisplay.SelectTCoordArray = 'None'
gridDisplay.SelectNormalArray = 'None'
gridDisplay.SelectTangentArray = 'None'
gridDisplay.OSPRayScaleArray = 'acceleration'
gridDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
gridDisplay.SelectOrientationVectors = 'None'
gridDisplay.ScaleFactor = 0.1125
gridDisplay.SelectScaleArray = 'None'
gridDisplay.GlyphType = 'Arrow'
gridDisplay.GlyphTableIndexArray = 'None'
gridDisplay.GaussianRadius = 0.005625
gridDisplay.SetScaleArray = ['POINTS', 'acceleration']
gridDisplay.ScaleTransferFunction = 'PiecewiseFunction'
gridDisplay.OpacityArray = ['POINTS', 'acceleration']
gridDisplay.OpacityTransferFunction = 'PiecewiseFunction'
gridDisplay.DataAxesGrid = 'GridAxesRepresentation'
gridDisplay.PolarAxes = 'PolarAxesRepresentation'
gridDisplay.ScalarOpacityFunction = velocityPWF
gridDisplay.ScalarOpacityUnitDistance = 0.0649519052838329
gridDisplay.OpacityArrayName = ['POINTS', 'acceleration']
gridDisplay.SelectInputVectors = ['POINTS', 'acceleration']
gridDisplay.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
gridDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
gridDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# show data from mergeBlocks1
mergeBlocks1Display = Show(mergeBlocks1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
mergeBlocks1Display.Representation = 'Surface'
mergeBlocks1Display.ColorArrayName = ['POINTS', 'velocity']
mergeBlocks1Display.LookupTable = velocityLUT
mergeBlocks1Display.SelectTCoordArray = 'None'
mergeBlocks1Display.SelectNormalArray = 'None'
mergeBlocks1Display.SelectTangentArray = 'None'
mergeBlocks1Display.OSPRayScaleArray = 'acceleration'
mergeBlocks1Display.OSPRayScaleFunction = 'PiecewiseFunction'
mergeBlocks1Display.SelectOrientationVectors = 'None'
mergeBlocks1Display.ScaleFactor = 0.1125
mergeBlocks1Display.SelectScaleArray = 'None'
mergeBlocks1Display.GlyphType = 'Arrow'
mergeBlocks1Display.GlyphTableIndexArray = 'None'
mergeBlocks1Display.GaussianRadius = 0.005625
mergeBlocks1Display.SetScaleArray = ['POINTS', 'acceleration']
mergeBlocks1Display.ScaleTransferFunction = 'PiecewiseFunction'
mergeBlocks1Display.OpacityArray = ['POINTS', 'acceleration']
mergeBlocks1Display.OpacityTransferFunction = 'PiecewiseFunction'
mergeBlocks1Display.DataAxesGrid = 'GridAxesRepresentation'
mergeBlocks1Display.PolarAxes = 'PolarAxesRepresentation'
mergeBlocks1Display.ScalarOpacityFunction = velocityPWF
mergeBlocks1Display.ScalarOpacityUnitDistance = 0.12785333546157115
mergeBlocks1Display.OpacityArrayName = ['POINTS', 'acceleration']
mergeBlocks1Display.SelectInputVectors = ['POINTS', 'acceleration']
mergeBlocks1Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
mergeBlocks1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
mergeBlocks1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup extractors
# ----------------------------------------------------------------

# create extractor
vTU1 = CreateExtractor('VTU', mergeBlocks1, registrationName='VTU1')
# trace defaults for the extractor.
vTU1.Trigger = 'TimeStep'

# init the 'VTU' selected for 'Writer'
vTU1.Writer.FileName = 'Slice_{timestep:06d}.pvtu'

# create extractor
pNG1 = CreateExtractor('PNG', renderView1, registrationName='PNG1')
# trace defaults for the extractor.
pNG1.Trigger = 'TimeStep'

# init the 'PNG' selected for 'Writer'
pNG1.Writer.FileName = 'RenderView1_{timestep:06d}{camera}.png'
pNG1.Writer.ImageResolution = [1043, 893]
pNG1.Writer.Format = 'PNG'

# create extractor
cinemaVolumetricPNG1 = CreateExtractor('CinemaVolumetricPNG', renderView1, registrationName='Cinema-Volumetric PNG1')

cinemaVolumetricPNG1.Writer.FileName = 'RenderView1_{timestep:06d}{camera}.png'
cinemaVolumetricPNG1.Writer.ImageResolution = [512,512]
cinemaVolumetricPNG1.Writer.Format = 'PNG'
cinemaVolumetricPNG1.Writer.CameraMode = 'Phi-Theta'
cinemaVolumetricPNG1.Writer.OpacityLevels = 5
cinemaVolumetricPNG1.Writer.Functions = 7
cinemaVolumetricPNG1.Writer.SingleFunctionOnly = 1
cinemaVolumetricPNG1.Writer.Range = [0.0, 0.0]
cinemaVolumetricPNG1.Writer.ExportTransferFunctions = 1

# ----------------------------------------------------------------
# restore active source
SetActiveSource(grid)
# ----------------------------------------------------------------

# ------------------------------------------------------------------------------
# Catalyst options
from paraview import catalyst
options = catalyst.Options()
options.GlobalTrigger = 'TimeStep'
options.EnableCatalystLive = 1
options.CatalystLiveTrigger = 'TimeStep'
#options.GlobalTrigger.Frequency = 10

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    from paraview.simple import SaveExtractsUsingCatalystOptions
    # Code for non in-situ environments; if executing in post-processing
    # i.e. non-Catalyst mode, let's generate extracts using Catalyst options
    SaveExtractsUsingCatalystOptions(options)
