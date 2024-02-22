#ifndef AscentAdaptor_h
#define AscentAdaptor_h

#include <lulesh.h>
#include <ascent/ascent.hpp>
#include "conduit_blueprint.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>

namespace AscentAdaptor
{
  ascent::Ascent ascent;
  conduit::Node actions;
        
void InitializeAscent()
{
  std::cout << "AscentInitialize.........................................\n";
  conduit::Node ascent_options;
  ascent_options["mpi_comm"] = MPI_Comm_c2f(MPI_COMM_WORLD);
  ascent.open(ascent_options);

// Create an action that tells Ascent to:
//  add a scene (s1) with one plot (p1)
//  that will render a pseudocolor of the magnitude of
//  the mesh field `velocity`

  conduit::Node &add_pipeline = actions.append(); 
  add_pipeline["action"] = "add_pipelines";
  // declare a pipeline (pl1)
  conduit::Node &pipes = add_pipeline["pipelines"];
  pipes["pl1/f1/type"]        = "vector_magnitude";
  pipes["pl1/f1/params/field"]       = "velocity";
  
  conduit::Node &add_scene = actions.append(); 
  add_scene["action"] = "add_scenes";
// declare a scene (s1) and pseudocolor plot (p1)
  conduit::Node &scenes = add_scene["scenes"];
  scenes["s1/plots/p1/type"]           = "pseudocolor";
  scenes["s1/plots/p1/pipeline"]       = "pl1";
  scenes["s1/plots/p1/field"]          = "velocity_magnitude";
  
  scenes["s1/renders/r1/image_prefix"]    = "velocity.%05d";
  scenes["s1/renders/r1/camera/azimuth"] = -135.;
  scenes["s1/renders/r1/camera/elevation"] = 30.;
}

void ExecuteAscent(Domain& locDom)
{
  conduit::Node mesh = locDom.visitNode();
  mesh["state/cycle"].set(locDom.cycle());
  mesh["state/time"].set(locDom.time());
  ascent.publish(mesh);
  ascent.execute(actions);
}

void FinalizeAscent()
{
  ascent.close();
  std::cout << "AscentFinalize.........................................\n";
}

}
#endif
