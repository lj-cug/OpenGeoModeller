
#in-situ
#export CATALYST_IMPLEMENTATION_PATHS=/home/sciviz/fmazen/in-transit/paraview_build/lib/catalyst
#DISPLAY=:1 ./bin/lulesh2.0 -x ../adios/pipeline.py -p -i 10 -s 128

#in-transit
#export CATALYST_IMPLEMENTATION_PATHS=/home/sciviz/fmazen/in-transit/adioscatalyst/build/lib/catalyst
export CATALYST_IMPLEMENTATION_NAME=adios
export CATALYST_IMPLEMENTATION_PREFER_ENV=1
./bin/lulesh2.0 -x ../adios/pipeline.py -x ../adios/adios2.xml -p -i 10 -s 128

