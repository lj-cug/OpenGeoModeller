# state file generated using paraview version 5.10.1

# uncomment the following three lines to ensure this script works in future versions
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = GetRenderView()
renderView1.ViewSize = [682, 709]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [0.5625, 0.5625, 0.5625]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [-2.616544955060017, 1.668742827966123, -1.4142571660271601]
renderView1.CameraFocalPoint = [0.5625, 0.5625, 0.5625]
renderView1.CameraViewUp = [0.2540984629768661, 0.95868931378901, 0.12786231164636275]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 1.0128497253571305
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

# create a new 'XML Partitioned Dataset Reader'
grid = XMLPartitionedDatasetReader(registrationName='grid', FileName=['/home/jfavre/Projects/InSitu/InSitu-Vis-Tutorial2022/Examples/LULESH/Catalyst/datasets/lulesh_000500.vtpd'])

# create a new 'Threshold'
threshold2 = Threshold(registrationName='Threshold2', Input=grid)
threshold2.Scalars = ['CELLS', 'p']
threshold2.LowerThreshold = 300.0
threshold2.UpperThreshold = 12363.67302297438

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from grid
gridDisplay = Show(grid, renderView1, 'StructuredGridRepresentation')

# trace defaults for the display properties.
gridDisplay.Representation = 'Outline'
gridDisplay.ColorArrayName = [None, '']
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
gridDisplay.ScalarOpacityUnitDistance = 0.0649519052838329
gridDisplay.InputVectors = ['POINTS', 'acceleration']
gridDisplay.SelectInputVectors = ['POINTS', 'acceleration']
gridDisplay.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
gridDisplay.ScaleTransferFunction.Points = [-161778.91114635803, 0.0, 0.5, 0.0, 240558.38186171357, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
gridDisplay.OpacityTransferFunction.Points = [-161778.91114635803, 0.0, 0.5, 0.0, 240558.38186171357, 1.0, 0.5, 0.0]

# show data from threshold2
threshold2Display = Show(threshold2, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'p'
pLUT = GetColorTransferFunction('p')
pLUT.RGBPoints = [329.06873339062196, 0.231373, 0.298039, 0.752941, 6346.370878182501, 0.865003, 0.865003, 0.865003, 12363.67302297438, 0.705882, 0.0156863, 0.14902]
pLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'p'
pPWF = GetOpacityTransferFunction('p')
pPWF.Points = [329.06873339062196, 0.0, 0.5, 0.0, 12363.67302297438, 1.0, 0.5, 0.0]
pPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
threshold2Display.Representation = 'Surface'
threshold2Display.ColorArrayName = ['CELLS', 'p']
threshold2Display.LookupTable = pLUT
threshold2Display.SelectTCoordArray = 'None'
threshold2Display.SelectNormalArray = 'None'
threshold2Display.SelectTangentArray = 'None'
threshold2Display.OSPRayScaleArray = 'acceleration'
threshold2Display.OSPRayScaleFunction = 'PiecewiseFunction'
threshold2Display.SelectOrientationVectors = 'None'
threshold2Display.ScaleFactor = 0.04567971241500028
threshold2Display.SelectScaleArray = 'None'
threshold2Display.GlyphType = 'Arrow'
threshold2Display.GlyphTableIndexArray = 'None'
threshold2Display.GaussianRadius = 0.002283985620750014
threshold2Display.SetScaleArray = ['POINTS', 'acceleration']
threshold2Display.ScaleTransferFunction = 'PiecewiseFunction'
threshold2Display.OpacityArray = ['POINTS', 'acceleration']
threshold2Display.OpacityTransferFunction = 'PiecewiseFunction'
threshold2Display.DataAxesGrid = 'GridAxesRepresentation'
threshold2Display.PolarAxes = 'PolarAxesRepresentation'
threshold2Display.ScalarOpacityFunction = pPWF
threshold2Display.ScalarOpacityUnitDistance = 0.081115029354551
threshold2Display.OpacityArrayName = ['POINTS', 'acceleration']
threshold2Display.InputVectors = ['POINTS', 'acceleration']
threshold2Display.SelectInputVectors = ['POINTS', 'acceleration']
threshold2Display.WriteLog = ''
threshold2Display.custom_kernel = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
threshold2Display.ScaleTransferFunction.Points = [-161778.91114635803, 0.0, 0.5, 0.0, 240558.38186171357, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
threshold2Display.OpacityTransferFunction.Points = [-161778.91114635803, 0.0, 0.5, 0.0, 240558.38186171357, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for pLUT in view renderView1
pLUTColorBar = GetScalarBar(pLUT, renderView1)
pLUTColorBar.Title = 'p'
pLUTColorBar.ComponentTitle = ''

# set color bar visibility
pLUTColorBar.Visibility = 1

# show color legend
threshold2Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get color transfer function/color map for 'velocity'
velocityLUT = GetColorTransferFunction('velocity')
velocityLUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941, 56.17461471106329, 0.865003, 0.865003, 0.865003, 112.34922942212658, 0.705882, 0.0156863, 0.14902]
velocityLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'velocity'
velocityPWF = GetOpacityTransferFunction('velocity')
velocityPWF.Points = [0.0, 0.0, 0.5, 0.0, 112.34922942212658, 1.0, 0.5, 0.0]
velocityPWF.ScalarRangeInitialized = 1

# ----------------------------------------------------------------
# restore active source
SetActiveSource(threshold2)
# ----------------------------------------------------------------


if __name__ == '__main__':
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='extracts')
