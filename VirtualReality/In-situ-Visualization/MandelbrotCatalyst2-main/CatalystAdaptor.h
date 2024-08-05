#ifndef CatalystAdaptor_h
#define CatalystAdaptor_h

#include "Mandelbrot.h"
#include <catalyst.hpp>

#include <iostream>
#include <string>

/**
 * The namespace hold wrappers for the three main functions of the catalyst API
 * - catalyst_initialize
 * - catalyst_execute
 * - catalyst_finalize
 * Although not required it often helps with regards to complexity to collect
 * catalyst calls under a class /namespace.
 */
namespace CatalystAdaptor
{
static void Initialize(int argc, char* argv[])
{
  conduit_cpp::Node node;
  node["catalyst/scripts/script/filename"].set_string(argv[1]);
  for (int cc = 2; cc < argc; ++cc)
  {
    conduit_cpp::Node list_entry = node["catalyst/scripts/script/args"].append();
    list_entry.set(argv[cc]);
  }
  catalyst_status err = catalyst_initialize(conduit_cpp::c_node(&node));
  if (err != catalyst_status_ok)
  {
    std::cerr << "Failed to initialize Catalyst: " << err << std::endl;
  }
}

static void Execute(int cycle, double time, Mandelbrot& brot)
{
  conduit_cpp::Node exec_params;
  auto state = exec_params["catalyst/state"];
  state["timestep"].set(cycle);
  state["time"].set(time);
  state["multiblock"].set(1);

  auto channel = exec_params["catalyst/channels/grid"];
  channel["type"].set("mesh");
  channel["memoryspace"].set_string("cuda");

  auto mesh = channel["data"];
  conduit_cpp::Node coords = mesh["coordsets/coords"];
  coords["type"] = "uniform";
  conduit_cpp::Node dims = coords["dims"];
  dims["i"] = brot.GetDimensions()[0];
  dims["j"] = brot.GetDimensions()[1];
  dims["k"] = brot.GetDimensions()[2];
  const auto numPoints = brot.GetDimensions()[0] * brot.GetDimensions()[1] * brot.GetDimensions()[2];
  conduit_cpp::Node origin = coords["origin"];
  origin["x"] = brot.GetOrigin()[0];
  origin["y"] = brot.GetOrigin()[1];
  origin["z"] = brot.GetOrigin()[2];
  conduit_cpp::Node spacing = coords["spacing"];
  spacing["dx"] = brot.GetSpacings()[0];
  spacing["dy"] = brot.GetSpacings()[1];
  spacing["dz"] = brot.GetSpacings()[2];
  mesh["topologies/mesh/type"] = "uniform";
  mesh["topologies/mesh/coordset"] = "coords";
  auto fields = mesh["fields"];
  fields["iterations/association"].set("vertex");
  fields["iterations/topology"].set("mesh");
  fields["iterations/volume_dependent"].set("false");
  fields["iterations/values"].set_external(brot.GetIterationsArray(), numPoints);
  catalyst_status err = catalyst_execute(conduit_cpp::c_node(&exec_params));
  if (err != catalyst_status_ok)
  {
    std::cerr << "Failed to execute Catalyst: " << err << std::endl;
  }
}

static void Finalize()
{
  conduit_cpp::Node node;
  catalyst_status err = catalyst_finalize(conduit_cpp::c_node(&node));
  if (err != catalyst_status_ok)
  {
    std::cerr << "Failed to finalize Catalyst: " << err << std::endl;
  }
}
}

#endif
