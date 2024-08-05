#if VIZ_CATALYST

#include <catalyst.h>
#include <lulesh.h>
#include <sstream>

void InitializeCatalyst(const cmdLineOpts& opts)
{
  conduit_node* node = conduit_node_create();
	for (size_t cc=0; cc < opts.scripts.size(); ++cc)
	{
		std::ostringstream str;
		str << "catalyst/scripts/sample" << cc;
  	conduit_node_set_path_char8_str(node, str.str().c_str(), opts.scripts[cc].c_str());
	}
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
}

#else

void InitializeCatalyst(const cmdLineOpts& opts)
{
}

void ExecuteCatalyst(Domain& locDom)
{
}

void FinalizeCatalyst()
{
}

#endif
