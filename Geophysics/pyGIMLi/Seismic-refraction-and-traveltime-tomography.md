# Seismic refraction and traveltime tomography

https://www.pygimli.org/_examples_auto/2_seismics/index.html

¹Ø¼ü´Ê: geophone positions and a measurement scheme, shot and receiver indices,first arrival traveltime

# 2D Refraction modeling and inversion

This example shows how to use the TravelTime manager to generate the response of a three-layered sloping model and to invert the synthetic noisified data.

## Model setup

We start by creating a three-layered slope (The model is taken from the BSc thesis of Constanze Reinken conducted at the University of Bonn).

Next we define geophone positions and a measurement scheme, which consists of shot and receiver indices.

## Synthetic data generation

Now we initialize the TravelTime manager and asssign P-wave velocities to the layers. To this end, we create a map from cell markers 0 through 3 to velocities (in m/s) and generate a velocity vector. To check whether the model looks correct, we plot it along with the sensor positions.

We use this model to create noisified synthetic data and look at the traveltime data matrix. Note, we force a specific noise seed as we want reproducable results for testing purposes. 
TODO: show first arrival traveltime curves.

## Inversion

Now we invert the synthetic data. We need a new independent mesh without information about the layered structure. This mesh can be created manual or guessd automatic from the data sensor positions (in this example). We tune the maximum cell size in the parametric domain to 15m^2

The manager also holds the method showResult that is used to plot the result. Note that only covered cells are shown by default. For comparison we plot the geometry on top.

Note that internally the following is called

ax, _ = pg.show(ra.mesh, vest, label="Velocity [m/s]", **kwargs)

Another useful method is to show the model along with its response on the data.

mgr.showResultAndFit(cMin=min(vp), cMax=max(vp))

# Raypaths in layered and gradient models

This example performs raytracing for a two-layer and a vertical gradient model and compares the resulting traveltimes to existing analytical solutions. An approximation of the raypath is found by finding the shortest-path through a grid of nodes. The possible angular coverage is small when only corner points of a cell (primary nodes) are used for this purpose. The angular coverage, and hence the numerical accuracy of traveltime calculations, can be significantly improved by a few secondary nodes along the cell edges. Details can be found in Giroux & Larouche (2013).

## Two-layer model

We start by building a regular grid.

## Vertical gradient model

We first create an unstructured mesh:

# Field data inversion (¡°Koenigsee¡±)

This minimalistic example shows how to use the Refraction Manager to invert a field data set. Here, we consider the Koenigsee data set, which represents classical refraction seismics data set with slightly heterogeneous overburden and some high-velocity bedrock. The data file can be found in the pyGIMLi example data repository.

The helper function pg.getExampleData downloads the data set to a temporary location and loads it. Printing the data reveals that there are 714 data points using 63 sensors (shots and geophones) with the data columns s (shot), g (geophone), and t (traveltime). By default, there is also a validity flag.

Let¡¯s have a look at the data in the form of traveltime curves.

We initialize the refraction manager.

Finally, we call the invert method and plot the result.The mesh is created based on the sensor positions on-the-fly.

First have a look at the data fit. Plot the measured (crosses) and modelled (lines) traveltimes.