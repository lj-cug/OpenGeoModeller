from paraview.simple import *
import os, os.path

# A trivial pipeline.
TrivialProducer(guiName="grid")
Show()
Render()

def _get_path():
    return os.path.join(os.getcwd(), "results")

def catalyst_initialize():
    # create output directory to write results
    path = _get_path()
    os.makedirs(path)
    print("saving results in '%s'" % path)

def catalyst_execute(params):
    Render()
    # generate results
    path = _get_path()
    WriteImage(os.path.join(path,"image-%d.png" % params.cycle))
    SaveData(os.path.join(path, "data-%d.vtpd" % params.cycle))
