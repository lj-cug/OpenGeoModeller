from paraview.simple import *
import os, os.path

# A trivial pipeline.

renderView1 = GetRenderView()
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

"""
def _get_path():
    return os.path.join(os.getcwd(), "results")

def catalyst_initialize():
    # create output directory to write results
    path = _get_path()
    #os.makedirs(path)
    print("saving results in '%s'" % path)

def catalyst_execute(params):
    Render()
    # generate results
    path = _get_path()
    WriteImage(os.path.join(path,"image-%d.png" % params.cycle))
    SaveData(os.path.join(path, "data-%d.vtpd" % params.cycle))
"""

vTP1 = CreateExtractor('VTPD', tp, registrationName='VTPD1')
vTP1.Trigger = 'TimeStep'
vTP1.Trigger.Frequency = 5
vTP1.Writer.FileName = 'particles_{timestep:06d}.vtpd'

# create extractor
pNG1 = CreateExtractor('PNG', renderView1, registrationName='PNG1')
pNG1.Trigger = 'TimeStep'
pNG1.Trigger.Frequency = 2

pNG1.Writer.FileName = 'RenderView1_{timestep:06d}{camera}.png'
pNG1.Writer.ImageResolution = [512,512]
pNG1.Writer.Format = 'PNG'

from paraview import catalyst
options = catalyst.Options()
options.GlobalTrigger = 'TimeStep'
options.EnableCatalystLive = 1
options.CatalystLiveTrigger = 'TimeStep'

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    from paraview.simple import SaveExtractsUsingCatalystOptions
    # Code for non in-situ environments; if executing in post-processing
    # i.e. non-Catalyst mode, let's generate extracts using Catalyst options
    SaveExtractsUsingCatalystOptions(options)
