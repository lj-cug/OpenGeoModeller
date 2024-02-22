#ifndef CatalystAdaptor_h
#define CatalystAdaptor_h

#include <catalyst.h>
#include <lulesh.h>
#include <sstream>
#include <iostream>
#include <string>

namespace CatalystAdaptor
{
void InitializeCatalyst(const cmdLineOpts& opts)
{
  std::cout << "CatalystInitialize.........................................\n";
  conduit_node* node = conduit_node_create();
  for (size_t cc=0; cc < opts.scripts.size(); ++cc)
    {
    //std::ostringstream str;
    //str << "catalyst/scripts/script/filename" << cc;
    //conduit_node_set_path_char8_str(node, str.str().c_str(), opts.scripts[cc].c_str());
    if (strstr(opts.scripts[cc].c_str(), "xml"))
    {
      conduit_node_set_path_char8_str(node, "adios/config_filepath", opts.scripts[cc].c_str());
    }
    else
    conduit_node_set_path_char8_str(node, "catalyst/scripts/script/filename", opts.scripts[cc].c_str());
    }
// indicate that we want to load ParaView-Catalyst
  conduit_node_set_path_char8_str(node, "catalyst_load/implementation", "paraview");
  conduit_node_set_path_char8_str(node, "catalyst_load/search_paths/paraview", PARAVIEW_IMPL_DIR);

  catalyst_initialize(node);

  conduit_node_destroy(node);
}

void ExecuteCatalyst(Domain& locDom)
{
  conduit_node* node = conduit_node_create();
  conduit_node_set_path_int64(node, "catalyst/state/cycle", locDom.cycle());
  conduit_node_set_path_float64(node, "catalyst/state/time", locDom.time());
  
  conduit_node_set_path_char8_str(node, "catalyst/channels/grid/type", "mesh");
  conduit_node_set_path_external_node(node, "catalyst/channels/grid/data", locDom.node());
  
  catalyst_execute(node);

  conduit_node_destroy(node);
}

void FinalizeCatalyst()
{
  conduit_node* node = conduit_node_create();
  
  catalyst_finalize(node);

  conduit_node_destroy(node);
  std::cout << "CatalystFinalize.........................................\n";
}

}

#endif

