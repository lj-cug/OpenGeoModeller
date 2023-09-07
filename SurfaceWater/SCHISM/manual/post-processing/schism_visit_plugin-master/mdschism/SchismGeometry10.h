#ifndef _SCHISMGEOMETRY10_H_
#define _SCHISMGEOMETRY10_H_


long  meshSideNum(long    *     a_mesh_node,
                  const long &  a_number_ele,
				  const long &  a_number_node);


void  meshSideNodeNeiborEle(long    *      a_mesh_node,
                            long    *     &a_node_neighbor_ele,
							long    *     &a_side_neighbor_ele,
							int     &      a_max_ele_at_node,
							const long &  a_number_ele,
				            const long &  a_number_node,
							const long &  a_number_side);


void   meshSideNode(long    *     a_side_node,
				    long    *     a_mesh_node,
				    const long &  a_number_side,
                    const long &  a_number_ele,
				    const long &  a_number_node);

#endif