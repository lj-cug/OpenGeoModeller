# script-version: 2.0
# Catalyst state generated using paraview version 5.10.0-RC2

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

renderView1 = CreateView('RenderView')
renderView1.ViewSize = [512,512]

tp = TrivialProducer(guiName="grid")
rep = Show()
ColorBy(rep, ['POINTS', 'velocity'])
lut = GetColorTransferFunction('velocity')
lut.RescaleTransferFunction(0.0, 34.0)
Render()

c = GetActiveCamera()
c.Azimuth(-135)
c.Elevation(30)
ResetCamera()

SetActiveSource(tp)

"""
recolorableImage1 = CreateExtractor('RecolorableImage', renderView1, registrationName='Recolorable Image1')
recolorableImage1.Trigger = 'TimeStep'
recolorableImage1.Trigger.Frequency = 5
recolorableImage1.Writer.FileName = 'RecolorableImage_{timestep:06d}{camera}.vtk'
recolorableImage1.Writer.ImageResolution = [512,512]
recolorableImage1.Writer.Format = 'VTK'
"""
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


# ------------------------------------------------------------------------------
# Catalyst options
from paraview import catalyst
options = catalyst.Options()
options.ExtractsOutputDirectory = 'cinema'
#options.GenerateCinemaSpecification = 1
options.GlobalTrigger = 'TimeStep'
options.CatalystLiveTrigger = 'TimeStep'
options.GlobalTrigger.Frequency = 100

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    from paraview.simple import SaveExtractsUsingCatalystOptions
    # Code for non in-situ environments; if executing in post-processing
    # i.e. non-Catalyst mode, let's generate extracts using Catalyst options
    SaveExtractsUsingCatalystOptions(options)
